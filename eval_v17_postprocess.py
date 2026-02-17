#!/usr/bin/env python3
"""v17 SAHI 오탐 제거: 7-Phase 과학적 후처리 실험

Phase 1: Cross-Class NMS (ON+OFF 중복 제거)
Phase 2: 최소 bbox 면적 필터
Phase 3: 이미지 경계 필터
Phase 4: 종횡비 필터
Phase 5: Full-Image Gate
Phase 6: 복합 최적화 (Phase 1~5 누적)
Phase 7: SAHI 파라미터 재탐색

1회 추론 → 후처리 조합별 post-hoc 평가
"""
import os, sys, time, logging
from collections import defaultdict
from itertools import product
from PIL import Image

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ── 설정 ──
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
MODEL = "/home/lay/hoban/hoban_go3k_v17/weights/best.pt"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}


# ══════════════════════════════════════════════
# 유틸리티 함수
# ══════════════════════════════════════════════

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


def evaluate(all_gt, all_preds, image_set, conf_thresh=None, per_class_conf=None):
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


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_class_detail(ctp, cfp, cfn):
    for cls_id, name in CLASS_NAMES.items():
        ct, cf, cm = ctp[cls_id], cfp[cls_id], cfn[cls_id]
        cp = ct/(ct+cf) if ct+cf else 0
        cr = ct/(ct+cm) if ct+cm else 0
        cf1 = 2*cp*cr/(cp+cr) if cp+cr else 0
        print(f"    {name}: P={cp:.3f} R={cr:.3f} F1={cf1:.3f} (TP={ct} FP={cf} FN={cm})")


# ══════════════════════════════════════════════
# 후처리 함수
# ══════════════════════════════════════════════

def cross_class_nms(preds, iou_thresh=0.5):
    """클래스 간 NMS: 같은 위치에 ON+OFF가 있으면 높은 conf만 유지"""
    if len(preds) <= 1:
        return preds
    sorted_preds = sorted(preds, key=lambda x: -x[1])  # conf 내림차순
    keep = []
    suppressed = set()
    for i, (c1, s1, x1a, y1a, x2a, y2a) in enumerate(sorted_preds):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            c2, s2, x1b, y1b, x2b, y2b = sorted_preds[j]
            if c1 != c2:  # 다른 클래스일 때만 cross-class 억제
                iou = compute_iou((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b))
                if iou >= iou_thresh:
                    suppressed.add(j)
    return keep


def filter_min_area(preds, min_area, img_w, img_h):
    """최소 면적 필터 (정규화 면적 기준)"""
    img_area = img_w * img_h
    return [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in preds
            if ((x2-x1)*(y2-y1)) / img_area >= min_area]


def filter_edge(preds, margin_ratio, img_w, img_h):
    """이미지 경계 필터: bbox가 가장자리에 너무 가까우면 제거"""
    mx = margin_ratio * img_w
    my = margin_ratio * img_h
    result = []
    for c, s, x1, y1, x2, y2 in preds:
        # bbox의 한 변이 이미지 경계에 있고, 반대편은 아닌 경우 (잘린 bbox)
        at_left = x1 < mx
        at_right = x2 > img_w - mx
        at_top = y1 < my
        at_bottom = y2 > img_h - my
        edge_count = sum([at_left, at_right, at_top, at_bottom])
        if edge_count >= 2:  # 2변 이상 경계에 접함 = 모서리에 있는 bbox
            continue
        result.append((c, s, x1, y1, x2, y2))
    return result


def filter_aspect_ratio(preds, min_ratio, max_ratio):
    """종횡비 필터: w/h 비율이 범위 밖이면 제거"""
    result = []
    for c, s, x1, y1, x2, y2 in preds:
        w, h = x2-x1, y2-y1
        if h <= 0:
            continue
        ratio = w / h
        if min_ratio <= ratio <= max_ratio:
            result.append((c, s, x1, y1, x2, y2))
    return result


def point_near_any(px, py, gates, radius):
    """SAHI bbox 중심이 gate 영역 근처에 있는지"""
    for gx1, gy1, gx2, gy2 in gates:
        gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
        if abs(px - gcx) <= radius and abs(py - gcy) <= radius:
            return True
    return False


def apply_gate(sahi_preds, full_raw, gate_conf, radius):
    """풀이미지 gate: 풀이미지에서 사람이 탐지된 곳만 SAHI 결과 유지"""
    gates = [(x1, y1, x2, y2) for conf, x1, y1, x2, y2 in full_raw if conf >= gate_conf]
    if not gates:
        return sahi_preds  # gate 없으면 전부 통과
    filtered = []
    for c, s, x1, y1, x2, y2 in sahi_preds:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if point_near_any(cx, cy, gates, radius):
            filtered.append((c, s, x1, y1, x2, y2))
    return filtered


def apply_pipeline(preds_dict, img_sizes, pipeline, full_preds=None):
    """파이프라인 순차 적용하여 필터링된 predictions 반환"""
    result = {}
    for fname, preds in preds_dict.items():
        img_w, img_h = img_sizes[fname]
        filtered = list(preds)

        for step_name, step_params in pipeline:
            if step_name == "cross_class_nms":
                filtered = cross_class_nms(filtered, **step_params)
            elif step_name == "min_area":
                filtered = filter_min_area(filtered, img_w=img_w, img_h=img_h, **step_params)
            elif step_name == "edge":
                filtered = filter_edge(filtered, img_w=img_w, img_h=img_h, **step_params)
            elif step_name == "aspect_ratio":
                filtered = filter_aspect_ratio(filtered, **step_params)
            elif step_name == "gate" and full_preds:
                full_raw = full_preds.get(fname, [])
                filtered = apply_gate(filtered, full_raw, **step_params)

        result[fname] = filtered
    return result


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
    gt_off = sum(1 for f in combined for g in all_gt[f] if g[0] == 1)
    print(f"평가 세트: {len(combined)}장 ({len(val_imgs)} val + {len(extra_imgs)} extra)")
    print(f"GT bbox: {sum(len(all_gt[f]) for f in combined)} (helmet_off: {gt_off})")

    # ── SAHI 추론 (conf=0.05, 1회) ──
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    print(f"\n[v17] SAHI 추론 (1280x720, overlap=0.15, conf=0.05)...")
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=MODEL,
        confidence_threshold=0.05, device="0", image_size=1280)

    all_preds = {}
    t0 = time.time()
    for i, f in enumerate(combined):
        if i % 50 == 0:
            print(f"  SAHI: {i}/{len(combined)}...", end="\r")
        r = get_sliced_prediction(
            img_paths[f], model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", postprocess_class_agnostic=False,
            verbose=0)
        all_preds[f] = [(p.category.id, p.score.value,
                         p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                        for p in r.object_prediction_list]
    print(f"  SAHI 완료: {sum(len(v) for v in all_preds.values())} preds ({time.time()-t0:.0f}s)")

    # ── Full-image 추론 (Gate용) ──
    from ultralytics import YOLO
    print(f"\n[v17] Full-image 추론 (Gate용, conf=0.01)...")
    yolo_model = YOLO(MODEL)
    full_preds = {}
    t0 = time.time()
    for i, f in enumerate(combined):
        if i % 100 == 0:
            print(f"  Full: {i}/{len(combined)}...", end="\r")
        results = yolo_model.predict(img_paths[f], conf=0.01, imgsz=1280,
                                     device="0", verbose=False)
        boxes = results[0].boxes
        # (conf, x1, y1, x2, y2) - 클래스 무관, gate는 "사람 존재" 확인용
        full_preds[f] = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                         for j in range(len(boxes))]
    print(f"  Full 완료: {sum(len(v) for v in full_preds.values())} preds ({time.time()-t0:.0f}s)")

    # ── Baseline ──
    print_header("BASELINE (후처리 없음)")
    confs = [0.30, 0.35, 0.40, 0.45, 0.50]
    best_base_f1, best_base_conf = 0, 0
    for conf in confs:
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, all_preds, combined, conf_thresh=conf)
        marker = ""
        if f1 > best_base_f1:
            best_base_f1, best_base_conf = f1, conf
            marker = " <--"
        print(f"  conf={conf:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn}){marker}")
    print(f"\n  Baseline best: F1={best_base_f1:.3f} @conf={best_base_conf}")
    tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, all_preds, combined, conf_thresh=best_base_conf)
    print_class_detail(ctp, cfp, cfn)
    baseline_fp = fp

    # ══════════════════════════════════════════
    # Phase 1: Cross-Class NMS
    # ══════════════════════════════════════════
    print_header("Phase 1: Cross-Class NMS")
    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    conf_sweeps = [0.30, 0.35, 0.40, 0.45, 0.50]
    best_p1_f1, best_p1_iou, best_p1_conf = 0, 0, 0

    for iou_th in iou_thresholds:
        pipeline = [("cross_class_nms", {"iou_thresh": iou_th})]
        filtered = apply_pipeline(all_preds, img_sizes, pipeline)
        for conf in conf_sweeps:
            tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, filtered, combined, conf_thresh=conf)
            marker = ""
            if f1 > best_p1_f1:
                best_p1_f1, best_p1_iou, best_p1_conf = f1, iou_th, conf
                marker = " <--"
            print(f"  IoU={iou_th:.1f} conf={conf:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f} FP={fp}({fp-baseline_fp:+d}){marker}")

    print(f"\n  Phase 1 best: F1={best_p1_f1:.3f} @IoU={best_p1_iou}, conf={best_p1_conf}")
    p1_pipeline = [("cross_class_nms", {"iou_thresh": best_p1_iou})]
    p1_filtered = apply_pipeline(all_preds, img_sizes, p1_pipeline)
    tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, p1_filtered, combined, conf_thresh=best_p1_conf)
    print_class_detail(ctp, cfp, cfn)
    print(f"  효과: FP {baseline_fp}→{fp} ({fp-baseline_fp:+d})")

    # ══════════════════════════════════════════
    # Phase 2: 최소 면적 필터
    # ══════════════════════════════════════════
    print_header("Phase 2: 최소 bbox 면적 필터 (+ Phase 1 최적)")
    min_areas = [0.00005, 0.0001, 0.0002, 0.0005, 0.001]
    best_p2_f1, best_p2_area = 0, 0

    for area in min_areas:
        pipeline = p1_pipeline + [("min_area", {"min_area": area})]
        filtered = apply_pipeline(all_preds, img_sizes, pipeline)
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, filtered, combined, conf_thresh=best_p1_conf)
        marker = ""
        if f1 > best_p2_f1:
            best_p2_f1, best_p2_area = f1, area
            marker = " <--"
        print(f"  min_area={area:.5f}: P={p:.3f} R={r:.3f} F1={f1:.3f} FP={fp} FN={fn}{marker}")

    print(f"\n  Phase 2 best: F1={best_p2_f1:.3f} @min_area={best_p2_area}")
    # 효과 없으면 스킵
    p2_pipeline = p1_pipeline + ([("min_area", {"min_area": best_p2_area})] if best_p2_f1 > best_p1_f1 else [])

    # ══════════════════════════════════════════
    # Phase 3: 경계 필터
    # ══════════════════════════════════════════
    print_header("Phase 3: 이미지 경계 필터 (+ Phase 1-2 최적)")
    margins = [0.005, 0.01, 0.02, 0.03]
    best_p3_f1, best_p3_margin = 0, 0

    for margin in margins:
        pipeline = p2_pipeline + [("edge", {"margin_ratio": margin})]
        filtered = apply_pipeline(all_preds, img_sizes, pipeline)
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, filtered, combined, conf_thresh=best_p1_conf)
        marker = ""
        if f1 > best_p3_f1:
            best_p3_f1, best_p3_margin = f1, margin
            marker = " <--"
        print(f"  margin={margin:.3f}: P={p:.3f} R={r:.3f} F1={f1:.3f} FP={fp} FN={fn}{marker}")

    p2_best = max(best_p1_f1, best_p2_f1)
    p3_pipeline = p2_pipeline + ([("edge", {"margin_ratio": best_p3_margin})] if best_p3_f1 > p2_best else [])
    print(f"\n  Phase 3 best: F1={best_p3_f1:.3f} @margin={best_p3_margin}")

    # ══════════════════════════════════════════
    # Phase 4: 종횡비 필터
    # ══════════════════════════════════════════
    print_header("Phase 4: 종횡비 필터 (+ Phase 1-3 최적)")
    ratios = [(0.2, 2.0), (0.25, 1.8), (0.3, 1.5), (0.15, 2.5)]
    best_p4_f1, best_p4_ratio = 0, (0, 0)

    for min_r, max_r in ratios:
        pipeline = p3_pipeline + [("aspect_ratio", {"min_ratio": min_r, "max_ratio": max_r})]
        filtered = apply_pipeline(all_preds, img_sizes, pipeline)
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, filtered, combined, conf_thresh=best_p1_conf)
        marker = ""
        if f1 > best_p4_f1:
            best_p4_f1, best_p4_ratio = f1, (min_r, max_r)
            marker = " <--"
        print(f"  ratio=[{min_r:.2f}, {max_r:.2f}]: P={p:.3f} R={r:.3f} F1={f1:.3f} FP={fp} FN={fn}{marker}")

    p3_best = max(best_p1_f1, best_p2_f1, best_p3_f1)
    p4_pipeline = p3_pipeline + ([("aspect_ratio", {"min_ratio": best_p4_ratio[0], "max_ratio": best_p4_ratio[1]})]
                                  if best_p4_f1 > p3_best else [])
    print(f"\n  Phase 4 best: F1={best_p4_f1:.3f} @ratio={best_p4_ratio}")

    # ══════════════════════════════════════════
    # Phase 5: Full-Image Gate
    # ══════════════════════════════════════════
    print_header("Phase 5: Full-Image Gate (+ Phase 1-4 최적)")
    gate_confs = [0.05, 0.10, 0.15, 0.20]
    gate_radii = [30, 50, 80, 120]
    best_p5_f1, best_p5_gc, best_p5_gr = 0, 0, 0

    for gc in gate_confs:
        for gr in gate_radii:
            pipeline = p4_pipeline + [("gate", {"gate_conf": gc, "radius": gr})]
            filtered = apply_pipeline(all_preds, img_sizes, pipeline, full_preds=full_preds)
            tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, filtered, combined, conf_thresh=best_p1_conf)
            marker = ""
            if f1 > best_p5_f1:
                best_p5_f1, best_p5_gc, best_p5_gr = f1, gc, gr
                marker = " <--"
            print(f"  gate_conf={gc:.2f} radius={gr:>3}px: P={p:.3f} R={r:.3f} F1={f1:.3f} FP={fp} FN={fn}{marker}")

    p4_best = max(best_p1_f1, best_p2_f1, best_p3_f1, best_p4_f1)
    p5_pipeline = p4_pipeline + ([("gate", {"gate_conf": best_p5_gc, "radius": best_p5_gr})]
                                  if best_p5_f1 > p4_best else [])
    print(f"\n  Phase 5 best: F1={best_p5_f1:.3f} @gate_conf={best_p5_gc}, radius={best_p5_gr}px")

    # ══════════════════════════════════════════
    # Phase 6: 복합 최적화 + Per-Class Conf
    # ══════════════════════════════════════════
    print_header("Phase 6: 복합 최적화 + Per-Class Confidence")

    # 누적 효과 비교
    print("\n[누적 효과 비교]")
    pipelines = {
        "A) Baseline (없음)": [],
        "B) Cross-class NMS": p1_pipeline,
        "C) B + 면적": p2_pipeline,
        "D) C + 경계": p3_pipeline,
        "E) D + 종횡비": p4_pipeline,
        "F) E + Gate": p5_pipeline,
    }
    for name, pipeline in pipelines.items():
        filtered = apply_pipeline(all_preds, img_sizes, pipeline, full_preds=full_preds)
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, filtered, combined, conf_thresh=best_p1_conf)
        print(f"  {name}: P={p:.3f} R={r:.3f} F1={f1:.3f} (FP={fp}, FN={fn})")

    # Per-class conf sweep on best pipeline
    print(f"\n[Per-Class Conf Sweep on best pipeline]")
    best_pipeline = p5_pipeline
    filtered_all = apply_pipeline(all_preds, img_sizes, best_pipeline, full_preds=full_preds)

    c0_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    c1_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    best_p6_f1, best_p6_c0, best_p6_c1 = 0, 0, 0

    for c0, c1 in product(c0_range, c1_range):
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, filtered_all, combined, per_class_conf={0: c0, 1: c1})
        if f1 > best_p6_f1:
            best_p6_f1, best_p6_c0, best_p6_c1 = f1, c0, c1

    tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
        all_gt, filtered_all, combined, per_class_conf={0: best_p6_c0, 1: best_p6_c1})
    print(f"  Best: c0={best_p6_c0:.2f}, c1={best_p6_c1:.2f}")
    print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")
    print_class_detail(ctp, cfp, cfn)

    # ══════════════════════════════════════════
    # Phase 7: SAHI 파라미터 재탐색
    # ══════════════════════════════════════════
    print_header("Phase 7: SAHI 파라미터 재탐색 (최적 후처리 적용)")

    sahi_configs = [
        ("agnostic=True, th=0.4, ov=0.15", True, 0.4, 0.15),
        ("agnostic=False, th=0.3, ov=0.15", False, 0.3, 0.15),
        ("agnostic=False, th=0.5, ov=0.15", False, 0.5, 0.15),
        ("agnostic=False, th=0.4, ov=0.20", False, 0.4, 0.20),
        ("agnostic=False, th=0.4, ov=0.25", False, 0.4, 0.25),
    ]

    best_p7_f1, best_p7_config = 0, ""
    for config_name, agnostic, match_th, overlap in sahi_configs:
        print(f"\n  [{config_name}] 추론 중...", end=" ", flush=True)
        t0 = time.time()
        temp_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8", model_path=MODEL,
            confidence_threshold=0.05, device="0", image_size=1280)

        temp_preds = {}
        for f in combined:
            r = get_sliced_prediction(
                img_paths[f], temp_model,
                slice_height=720, slice_width=1280,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                perform_standard_pred=True,
                postprocess_type="NMS", postprocess_match_threshold=match_th,
                postprocess_match_metric="IOS", postprocess_class_agnostic=agnostic,
                verbose=0)
            temp_preds[f] = [(p.category.id, p.score.value,
                              p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                             for p in r.object_prediction_list]
        elapsed = time.time() - t0
        print(f"({elapsed:.0f}s)")

        # 최적 후처리 적용
        filtered = apply_pipeline(temp_preds, img_sizes, best_pipeline, full_preds=full_preds)
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, filtered, combined, per_class_conf={0: best_p6_c0, 1: best_p6_c1})
        marker = ""
        if f1 > best_p7_f1:
            best_p7_f1, best_p7_config = f1, config_name
            marker = " <--"
        print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} (FP={fp} FN={fn}){marker}")
        print_class_detail(ctp, cfp, cfn)

    # ══════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════
    elapsed_total = time.time() - t_total
    print_header(f"FINAL SUMMARY (소요: {elapsed_total/60:.1f}분)")

    print(f"\n  Baseline:        F1={best_base_f1:.3f} @conf={best_base_conf}")
    print(f"  P1 Cross-NMS:    F1={best_p1_f1:.3f} @IoU={best_p1_iou}, conf={best_p1_conf}")
    print(f"  P2 면적 필터:    F1={best_p2_f1:.3f} @min_area={best_p2_area}")
    print(f"  P3 경계 필터:    F1={best_p3_f1:.3f} @margin={best_p3_margin}")
    print(f"  P4 종횡비:       F1={best_p4_f1:.3f} @ratio={best_p4_ratio}")
    print(f"  P5 Gate:         F1={best_p5_f1:.3f} @gc={best_p5_gc}, gr={best_p5_gr}")
    print(f"  P6 Per-class:    F1={best_p6_f1:.3f} @c0={best_p6_c0}, c1={best_p6_c1}")
    print(f"  P7 SAHI재탐색:   F1={best_p7_f1:.3f} @{best_p7_config}")

    print(f"\n  최종 파이프라인:")
    for step_name, params in best_pipeline:
        print(f"    {step_name}: {params}")
    print(f"    per_class_conf: c0={best_p6_c0}, c1={best_p6_c1}")
    if best_p7_f1 > best_p6_f1:
        print(f"    SAHI: {best_p7_config}")
    print(f"\n  개선: F1 {best_base_f1:.3f} → {max(best_p6_f1, best_p7_f1):.3f} "
          f"({max(best_p6_f1, best_p7_f1)-best_base_f1:+.3f})")
