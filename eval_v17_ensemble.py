#!/usr/bin/env python3
"""v17 앙상블 테스트: v17 + v16 + v13 조합별 NMS/WBF 평가

각 모델 SAHI 추론 1회 → NMS/WBF 앙상블 → 후처리 파이프라인 → per-class conf sweep
평가: 729장 combined set (3k val + verified helmet_off)

조합:
  1. v17 단독 (baseline with post-processing)
  2. v17 + v16 NMS
  3. v17 + v13 NMS
  4. v17 + v16 + v13 NMS
  5. v17 + v16 WBF (ensemble_boxes)
"""
import os, sys, time, logging
from collections import defaultdict
from PIL import Image

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ── 설정 ──
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"

MODELS = {
    "v17": {
        "path": "/home/lay/hoban/hoban_go3k_v17/weights/best.pt",
        "imgsz": 1280,
        "desc": "1280px, COCO pt, F1=0.918",
    },
    "v16": {
        "path": "/home/lay/hoban/hoban_go3k_v16_640/weights/best.pt",
        "imgsz": 640,
        "desc": "640px, v13 pt, F1=0.885",
    },
    "v13": {
        "path": "/home/lay/hoban/hoban_v13_stage2/weights/best.pt",
        "imgsz": 1280,
        "desc": "curriculum learning",
    },
}

CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}

# v17 최적 후처리 파이프라인
BEST_PIPELINE = [
    ("cross_class_nms", {"iou_thresh": 0.3}),
    ("min_area", {"min_area": 5e-05}),
    ("gate", {"gate_conf": 0.20, "radius": 30}),
]
BEST_PER_CLASS = {0: 0.40, 1: 0.15}

# ── 유틸리티 ──

def load_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, (cx-w/2)*img_w, (cy-h/2)*img_h,
                          (cx+w/2)*img_w, (cy+h/2)*img_h))
    return boxes


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def evaluate(all_gt, all_preds, image_set, per_class_conf=None, conf_thresh=None):
    tp = fp = fn = 0
    ctp, cfp, cfn = defaultdict(int), defaultdict(int), defaultdict(int)
    for fname in image_set:
        gts = all_gt.get(fname, [])
        raw = all_preds.get(fname, [])
        if per_class_conf:
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in raw
                     if s >= per_class_conf.get(c, 0.5)]
        elif conf_thresh is not None:
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in raw
                     if s >= conf_thresh]
        else:
            preds = raw
        matched = set()
        for _, (pc, ps, px1, py1, px2, py2) in sorted(enumerate(preds), key=lambda x: -x[1][1]):
            bi, bv = -1, 0
            for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
                if gi in matched or gc != pc:
                    continue
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > bv:
                    bv, bi = iou, gi
            if bv >= 0.5 and bi >= 0:
                tp += 1; ctp[gts[bi][0]] += 1; matched.add(bi)
            else:
                fp += 1; cfp[pc] += 1
        for gi in range(len(gts)):
            if gi not in matched:
                fn += 1; cfn[gts[gi][0]] += 1
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn


def print_class_detail(ctp, cfp, cfn):
    for cls_id, name in CLASS_NAMES.items():
        ct, cf, cm = ctp[cls_id], cfp[cls_id], cfn[cls_id]
        cp = ct/(ct+cf) if ct+cf else 0
        cr = ct/(ct+cm) if ct+cm else 0
        cf1 = 2*cp*cr/(cp+cr) if cp+cr else 0
        print(f"    {name}: P={cp:.3f} R={cr:.3f} F1={cf1:.3f} (TP={ct} FP={cf} FN={cm})")


# ── 후처리 함수 ──

def cross_class_nms(preds, iou_thresh=0.3):
    if len(preds) <= 1:
        return preds
    sorted_preds = sorted(preds, key=lambda x: -x[1])
    keep, suppressed = [], set()
    for i, (c1, s1, x1a, y1a, x2a, y2a) in enumerate(sorted_preds):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            c2 = sorted_preds[j][0]
            if c1 != c2:
                iou = compute_iou((x1a, y1a, x2a, y2a),
                                  (sorted_preds[j][2], sorted_preds[j][3],
                                   sorted_preds[j][4], sorted_preds[j][5]))
                if iou >= iou_thresh:
                    suppressed.add(j)
    return keep


def filter_min_area(preds, min_area, img_w, img_h):
    img_area = img_w * img_h
    return [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in preds
            if ((x2-x1)*(y2-y1)) / img_area >= min_area]


def apply_gate(sahi_preds, full_raw, gate_conf, radius):
    gates = [(x1,y1,x2,y2) for conf,x1,y1,x2,y2 in full_raw if conf >= gate_conf]
    if not gates:
        return sahi_preds
    filtered = []
    for c,s,x1,y1,x2,y2 in sahi_preds:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        for gx1,gy1,gx2,gy2 in gates:
            gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
            if abs(cx-gcx) <= radius and abs(cy-gcy) <= radius:
                filtered.append((c,s,x1,y1,x2,y2))
                break
    return filtered


def apply_pipeline(preds, full_raw, img_w, img_h):
    """v17 최적 후처리 파이프라인 적용"""
    filtered = cross_class_nms(preds, 0.3)
    filtered = filter_min_area(filtered, 5e-05, img_w, img_h)
    filtered = apply_gate(filtered, full_raw, 0.20, 30)
    return filtered


# ── 앙상블 함수 ──

def ensemble_nms(preds_list, iou_thresh=0.5):
    """다중 모델 predictions를 NMS로 앙상블
    preds_list: [[(cls, conf, x1, y1, x2, y2), ...], ...]
    """
    # 모든 모델의 predictions 합치기
    all_preds = []
    for preds in preds_list:
        all_preds.extend(preds)

    if len(all_preds) <= 1:
        return all_preds

    # conf 내림차순 정렬 → 같은 클래스끼리 NMS
    sorted_preds = sorted(all_preds, key=lambda x: -x[1])
    keep, suppressed = [], set()

    for i in range(len(sorted_preds)):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        c1 = sorted_preds[i][0]
        b1 = sorted_preds[i][2:]
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            c2 = sorted_preds[j][0]
            if c1 == c2:  # 같은 클래스만 NMS
                iou = compute_iou(b1, sorted_preds[j][2:])
                if iou >= iou_thresh:
                    suppressed.add(j)
    return keep


def ensemble_wbf(preds_list, img_w, img_h, iou_thresh=0.5, skip_box_thr=0.01):
    """Weighted Boxes Fusion (ensemble_boxes 라이브러리 사용)"""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print("  [WBF] ensemble_boxes 미설치 → pip install ensemble_boxes")
        return None

    boxes_list, scores_list, labels_list = [], [], []

    for preds in preds_list:
        boxes, scores, labels = [], [], []
        for cls, conf, x1, y1, x2, y2 in preds:
            # ensemble_boxes는 0~1 정규화 좌표 사용
            boxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
            scores.append(conf)
            labels.append(cls)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if not any(boxes_list):
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thresh, skip_box_thr=skip_box_thr)

    result = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h
        result.append((int(label), float(score), x1, y1, x2, y2))
    return result


# ── SAHI 추론 ──

def run_sahi_inference(model_name, model_info, combined, img_paths):
    """단일 모델 SAHI 추론"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    imgsz = model_info["imgsz"]
    # SAHI 슬라이스 크기: 모델 imgsz에 맞춤
    if imgsz == 1280:
        slice_w, slice_h = 1280, 720
    else:
        slice_w, slice_h = 640, 640

    print(f"\n[{model_name}] SAHI 추론 ({slice_w}x{slice_h}, conf=0.05)...")
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_info["path"],
        confidence_threshold=0.05, device="0", image_size=imgsz)

    preds = {}
    t0 = time.time()
    for i, f in enumerate(combined):
        if i % 50 == 0:
            print(f"  {model_name}: {i}/{len(combined)}...", end="\r")
        r = get_sliced_prediction(
            img_paths[f], model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", verbose=0)
        preds[f] = [(p.category.id, p.score.value,
                     p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                    for p in r.object_prediction_list]
    elapsed = time.time() - t0
    total = sum(len(v) for v in preds.values())
    print(f"  {model_name} 완료: {total} preds ({elapsed:.0f}s)")
    return preds


def run_full_inference(model_path, combined, img_paths):
    """Full-image 추론 (Gate용)"""
    from ultralytics import YOLO
    print(f"\n[v17] Full-image 추론 (Gate용, conf=0.01)...")
    yolo_model = YOLO(model_path)
    full_preds = {}
    t0 = time.time()
    for i, f in enumerate(combined):
        if i % 100 == 0:
            print(f"  Full: {i}/{len(combined)}...", end="\r")
        results = yolo_model.predict(img_paths[f], conf=0.01, imgsz=1280,
                                     device="0", verbose=False)
        boxes = results[0].boxes
        full_preds[f] = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                         for j in range(len(boxes))]
    print(f"  Full 완료: {sum(len(v) for v in full_preds.values())} preds ({time.time()-t0:.0f}s)")
    return full_preds


# ── 평가 헬퍼 ──

def eval_with_pipeline(name, all_gt, raw_preds, full_preds, img_sizes, combined):
    """후처리 파이프라인 + per-class conf sweep"""
    # 파이프라인 적용
    processed = {}
    for f in combined:
        preds = raw_preds.get(f, [])
        full_raw = full_preds.get(f, [])
        img_w, img_h = img_sizes[f]
        processed[f] = apply_pipeline(preds, full_raw, img_w, img_h)

    # per-class conf sweep
    best_f1, best_conf = 0, {}
    for c0 in [0.30, 0.35, 0.40, 0.45]:
        for c1 in [0.10, 0.15, 0.20, 0.25]:
            pc = {0: c0, 1: c1}
            tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
                all_gt, processed, combined, per_class_conf=pc)
            if f1 > best_f1:
                best_f1, best_conf = f1, pc
                best_result = (tp, fp, fn, p, r, f1, ctp, cfp, cfn)

    tp, fp, fn, p, r, f1, ctp, cfp, cfn = best_result
    print(f"  [{name}] F1={f1:.3f} P={p:.3f} R={r:.3f} "
          f"(TP={tp} FP={fp} FN={fn}) @c0={best_conf[0]}, c1={best_conf[1]}")
    print_class_detail(ctp, cfp, cfn)
    return best_f1, best_conf

def eval_simple(name, all_gt, raw_preds, combined):
    """단순 conf sweep (후처리 없이)"""
    best_f1, best_conf = 0, 0
    for conf in [0.30, 0.35, 0.40, 0.45, 0.50]:
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, raw_preds, combined, conf_thresh=conf)
        if f1 > best_f1:
            best_f1, best_conf = f1, conf
    tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
        all_gt, raw_preds, combined, conf_thresh=best_conf)
    print(f"  [{name} raw] F1={f1:.3f} P={p:.3f} R={r:.3f} "
          f"(TP={tp} FP={fp} FN={fn}) @conf={best_conf}")
    return best_f1


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.time()

    # ── GT 로드 ──
    val_imgs = sorted(f for f in os.listdir(VAL_IMG) if f.endswith(".jpg"))
    all_gt, img_sizes, img_paths = {}, {}, {}
    for f in val_imgs:
        path = os.path.join(VAL_IMG, f)
        img = Image.open(path)
        img_sizes[f] = img.size
        img_paths[f] = path
        all_gt[f] = load_gt(os.path.join(VAL_LBL, f.replace(".jpg", ".txt")), *img.size)

    extra_imgs = []
    if os.path.isdir(EXTRA_IMG):
        for f in sorted(os.listdir(EXTRA_IMG)):
            if f.endswith(".jpg") and f not in all_gt:
                path = os.path.join(EXTRA_IMG, f)
                img = Image.open(path)
                img_sizes[f] = img.size
                img_paths[f] = path
                all_gt[f] = load_gt(os.path.join(EXTRA_LBL, f.replace(".jpg", ".txt")), *img.size)
                extra_imgs.append(f)

    combined = val_imgs + extra_imgs
    gt_total = sum(len(all_gt[f]) for f in combined)
    gt_off = sum(1 for f in combined for g in all_gt[f] if g[0] == 1)
    print(f"평가 세트: {len(combined)}장 ({len(val_imgs)} val + {len(extra_imgs)} extra)")
    print(f"GT bbox: {gt_total} (helmet_off: {gt_off})")

    # ── 각 모델 SAHI 추론 ──
    model_preds = {}
    for name, info in MODELS.items():
        model_preds[name] = run_sahi_inference(name, info, combined, img_paths)

    # ── v17 Full-image (Gate용) ──
    full_preds = run_full_inference(MODELS["v17"]["path"], combined, img_paths)

    # ══════════════════════════════════════════════
    # 1. 각 모델 단독 평가 (baseline)
    # ══════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  1. 각 모델 단독 평가")
    print(f"{'='*80}")

    for name in MODELS:
        eval_simple(name, all_gt, model_preds[name], combined)

    print(f"\n  [v17 + 후처리 파이프라인]")
    v17_f1, v17_conf = eval_with_pipeline(
        "v17+pipeline", all_gt, model_preds["v17"], full_preds, img_sizes, combined)

    # ══════════════════════════════════════════════
    # 2. NMS 앙상블 (IoU sweep)
    # ══════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  2. NMS 앙상블")
    print(f"{'='*80}")

    ensemble_combos = [
        ("v17+v16", ["v17", "v16"]),
        ("v17+v13", ["v17", "v13"]),
        ("v17+v16+v13", ["v17", "v16", "v13"]),
    ]

    best_overall_f1 = v17_f1
    best_overall_name = "v17+pipeline"

    for combo_name, model_names in ensemble_combos:
        print(f"\n  --- {combo_name} ---")
        for nms_iou in [0.4, 0.5, 0.6]:
            # NMS 앙상블
            ens_preds = {}
            for f in combined:
                preds_list = [model_preds[m].get(f, []) for m in model_names]
                ens_preds[f] = ensemble_nms(preds_list, iou_thresh=nms_iou)

            # 후처리 파이프라인 적용
            processed = {}
            for f in combined:
                preds = ens_preds.get(f, [])
                full_raw = full_preds.get(f, [])
                img_w, img_h = img_sizes[f]
                processed[f] = apply_pipeline(preds, full_raw, img_w, img_h)

            # per-class conf sweep
            best_f1 = 0
            for c0 in [0.35, 0.40, 0.45]:
                for c1 in [0.10, 0.15, 0.20]:
                    pc = {0: c0, 1: c1}
                    tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
                        all_gt, processed, combined, per_class_conf=pc)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_result = (tp, fp, fn, p, r, f1, ctp, cfp, cfn, pc)

            tp, fp, fn, p, r, f1, ctp, cfp, cfn, pc = best_result
            marker = " ***" if f1 > best_overall_f1 else ""
            print(f"  NMS(IoU={nms_iou}): F1={f1:.3f} P={p:.3f} R={r:.3f} "
                  f"(TP={tp} FP={fp} FN={fn}) @c0={pc[0]},c1={pc[1]}{marker}")
            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_overall_name = f"{combo_name} NMS(IoU={nms_iou})"
                print_class_detail(ctp, cfp, cfn)

    # ══════════════════════════════════════════════
    # 3. WBF 앙상블 (ensemble_boxes)
    # ══════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  3. WBF 앙상블")
    print(f"{'='*80}")

    wbf_available = True
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        print("  ensemble_boxes 미설치. pip install ensemble_boxes 후 재실행")
        wbf_available = False

    if wbf_available:
        wbf_combos = [
            ("v17+v16", ["v17", "v16"]),
            ("v17+v16+v13", ["v17", "v16", "v13"]),
        ]

        for combo_name, model_names in wbf_combos:
            print(f"\n  --- {combo_name} WBF ---")
            for wbf_iou in [0.4, 0.5, 0.6]:
                ens_preds = {}
                for f in combined:
                    img_w, img_h = img_sizes[f]
                    preds_list = [model_preds[m].get(f, []) for m in model_names]
                    result = ensemble_wbf(preds_list, img_w, img_h,
                                          iou_thresh=wbf_iou, skip_box_thr=0.01)
                    ens_preds[f] = result if result is not None else []

                # 후처리 파이프라인
                processed = {}
                for f in combined:
                    preds = ens_preds.get(f, [])
                    full_raw = full_preds.get(f, [])
                    img_w, img_h = img_sizes[f]
                    processed[f] = apply_pipeline(preds, full_raw, img_w, img_h)

                # per-class conf sweep
                best_f1 = 0
                for c0 in [0.30, 0.35, 0.40]:
                    for c1 in [0.10, 0.15, 0.20]:
                        pc = {0: c0, 1: c1}
                        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
                            all_gt, processed, combined, per_class_conf=pc)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_result = (tp, fp, fn, p, r, f1, ctp, cfp, cfn, pc)

                tp, fp, fn, p, r, f1, ctp, cfp, cfn, pc = best_result
                marker = " ***" if f1 > best_overall_f1 else ""
                print(f"  WBF(IoU={wbf_iou}): F1={f1:.3f} P={p:.3f} R={r:.3f} "
                      f"(TP={tp} FP={fp} FN={fn}) @c0={pc[0]},c1={pc[1]}{marker}")
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    best_overall_name = f"{combo_name} WBF(IoU={wbf_iou})"
                    print_class_detail(ctp, cfp, cfn)

    # ══════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════
    elapsed = time.time() - t_total
    print(f"\n{'='*80}")
    print(f"  SUMMARY (소요: {elapsed/60:.1f}분)")
    print(f"{'='*80}")
    print(f"  v17 단독 (후처리): F1={v17_f1:.3f}")
    print(f"  최고 앙상블:       F1={best_overall_f1:.3f} ({best_overall_name})")
    if best_overall_f1 > v17_f1:
        print(f"  개선: +{best_overall_f1 - v17_f1:.3f}")
    else:
        print(f"  앙상블 효과 없음 (v17 단독이 최고)")
