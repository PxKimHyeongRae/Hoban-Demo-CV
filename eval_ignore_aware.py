#!/usr/bin/env python3
"""Ignore-aware SAHI F1 평가

area < IGNORE_THRESH인 GT/Pred는 ignore 처리:
  - ignore GT: 미탐지해도 FN 아님, 탐지해도 TP 아님
  - pred가 ignore GT에만 매칭 → FP도 아님 (무시)
  - pred area < IGNORE_THRESH이고 active GT 매칭 없으면 → FP 아님

COCO 스타일 ignore region 적용.

사용법:
  python eval_ignore_aware.py                          # v19 기본
  python eval_ignore_aware.py --model path/to/best.pt  # 다른 모델
  python eval_ignore_aware.py --ignore-thresh 0.00015  # threshold 변경
  python eval_ignore_aware.py --compare                # v17/v19/v21/v23 전체 비교
"""
import os, sys, time, logging, argparse
from collections import defaultdict
from PIL import Image

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ── 설정 ──
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
DEFAULT_MODEL = "/home/lay/hoban/hoban_go3k_v19/weights/best.pt"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}

IGNORE_THRESH = 0.00020  # area < 이 값이면 ignore (~20px at 1280px)

MODELS_COMPARE = {
    "v19": "/home/lay/hoban/hoban_go3k_v19/weights/best.pt",
    "v17": "/home/lay/hoban/hoban_go3k_v17/weights/best.pt",
    "v21-l": "/home/lay/hoban/hoban_go3k_v21_l/weights/best.pt",
    "v23": "/home/lay/hoban/hoban_go3k_v23/weights/best.pt",
}


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


def bbox_area_ratio(x1, y1, x2, y2, img_w, img_h):
    return ((x2-x1) * (y2-y1)) / (img_w * img_h)


def evaluate_ignore(all_gt, all_preds, image_set, img_sizes,
                    per_class_conf=None, conf_thresh=None,
                    ignore_thresh=IGNORE_THRESH):
    """Ignore-aware 평가.

    - active GT (area >= ignore_thresh): 정상 매칭
    - ignore GT (area < ignore_thresh): 매칭되면 무시, 미매칭도 FN 아님
    - pred가 ignore GT에만 매칭 → FP도 아님
    """
    tp = fp = fn = 0
    ignored_preds = ignored_gts = 0
    ctp, cfp, cfn = defaultdict(int), defaultdict(int), defaultdict(int)

    for fname in image_set:
        gts = all_gt.get(fname, [])
        raw = all_preds.get(fname, [])
        img_w, img_h = img_sizes[fname]

        # conf 필터링
        if per_class_conf:
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in raw
                     if s >= per_class_conf.get(c, 0.5)]
        elif conf_thresh is not None:
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in raw
                     if s >= conf_thresh]
        else:
            preds = raw

        # GT를 active / ignore로 분리
        active_gts = []  # (idx, cls, x1, y1, x2, y2)
        ignore_gts = []
        for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
            area = bbox_area_ratio(gx1, gy1, gx2, gy2, img_w, img_h)
            if area >= ignore_thresh:
                active_gts.append((gi, gc, gx1, gy1, gx2, gy2))
            else:
                ignore_gts.append((gi, gc, gx1, gy1, gx2, gy2))
                ignored_gts += 1

        # Pred 매칭 (conf 내림차순)
        matched_active = set()
        matched_ignore = set()

        sorted_preds = sorted(enumerate(preds), key=lambda x: -x[1][1])

        for pi, (pc, ps, px1, py1, px2, py2) in sorted_preds:
            # 1. active GT 매칭 시도
            best_iou, best_idx = 0, -1
            for ai, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(active_gts):
                if ai in matched_active or gc != pc:
                    continue
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou, best_idx = iou, ai

            if best_iou >= 0.5 and best_idx >= 0:
                # TP: active GT 매칭
                tp += 1
                ctp[pc] += 1
                matched_active.add(best_idx)
                continue

            # 2. ignore GT 매칭 시도 → 매칭되면 FP도 아님
            ignore_match = False
            for ii, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(ignore_gts):
                if ii in matched_ignore or gc != pc:
                    continue
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou >= 0.5:
                    ignore_match = True
                    matched_ignore.add(ii)
                    break

            if ignore_match:
                ignored_preds += 1
                continue

            # 3. pred 자체가 tiny이면 → FP로 세지 않음
            pred_area = bbox_area_ratio(px1, py1, px2, py2, img_w, img_h)
            if pred_area < ignore_thresh:
                ignored_preds += 1
                continue

            # 4. FP
            fp += 1
            cfp[pc] += 1

        # FN: 미매칭된 active GT만 카운트
        for ai in range(len(active_gts)):
            if ai not in matched_active:
                fn += 1
                cfn[active_gts[ai][1]] += 1

    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn, ignored_preds, ignored_gts


def print_class_detail(ctp, cfp, cfn):
    for cls_id, name in CLASS_NAMES.items():
        ct, cf, cm = ctp[cls_id], cfp[cls_id], cfn[cls_id]
        cp = ct/(ct+cf) if ct+cf else 0
        cr = ct/(ct+cm) if ct+cm else 0
        cf1 = 2*cp*cr/(cp+cr) if cp+cr else 0
        print(f"    {name}: P={cp:.3f} R={cr:.3f} F1={cf1:.3f} (TP={ct} FP={cf} FN={cm})")


# ── 후처리 ──

def cross_class_nms(preds, iou_thresh=0.3):
    if len(preds) <= 1:
        return preds
    sorted_preds = sorted(preds, key=lambda x: -x[1])
    keep, suppressed = [], set()
    for i in range(len(sorted_preds)):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        c1, b1 = sorted_preds[i][0], sorted_preds[i][2:]
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            if c1 != sorted_preds[j][0]:
                if compute_iou(b1, sorted_preds[j][2:]) >= iou_thresh:
                    suppressed.add(j)
    return keep


def apply_pipeline(preds, full_raw, img_w, img_h):
    """v17 최적 후처리 파이프라인"""
    filtered = cross_class_nms(preds, 0.3)
    img_area = img_w * img_h
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if ((x2-x1)*(y2-y1)) / img_area >= 5e-05]
    gates = [(x1,y1,x2,y2) for conf,x1,y1,x2,y2 in full_raw if conf >= 0.20]
    if gates:
        gated = []
        for c,s,x1,y1,x2,y2 in filtered:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            for gx1,gy1,gx2,gy2 in gates:
                gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
                if abs(cx-gcx) <= 30 and abs(cy-gcy) <= 30:
                    gated.append((c,s,x1,y1,x2,y2))
                    break
        filtered = gated
    return filtered


def run_sahi_inference(model_path, combined, img_paths):
    """SAHI + Full-image 추론"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO

    print(f"  SAHI 추론 (1280x720, overlap=0.15, conf=0.05)...")
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.05, device="0", image_size=1280)

    all_preds = {}
    t0 = time.time()
    for i, f in enumerate(combined):
        if i % 50 == 0:
            print(f"    SAHI: {i}/{len(combined)}...", end="\r")
        r = get_sliced_prediction(
            img_paths[f], model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", verbose=0)
        all_preds[f] = [(p.category.id, p.score.value,
                         p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                        for p in r.object_prediction_list]
    print(f"    SAHI 완료: {sum(len(v) for v in all_preds.values())} preds ({time.time()-t0:.0f}s)")

    print(f"  Full-image 추론 (Gate용, conf=0.01)...")
    yolo_model = YOLO(model_path)
    full_preds = {}
    t0 = time.time()
    for i, f in enumerate(combined):
        if i % 100 == 0:
            print(f"    Full: {i}/{len(combined)}...", end="\r")
        results = yolo_model.predict(img_paths[f], conf=0.01, imgsz=1280,
                                     device="0", verbose=False)
        boxes = results[0].boxes
        full_preds[f] = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                         for j in range(len(boxes))]
    print(f"    Full 완료 ({time.time()-t0:.0f}s)")

    return all_preds, full_preds


def eval_single_model(model_path, model_name, all_gt, combined, img_sizes, img_paths,
                      ignore_thresh, verbose=True):
    """단일 모델 평가 (기존 + ignore-aware 비교)"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"  모델: {model_name} ({model_path})")
        print(f"  Ignore threshold: area < {ignore_thresh} (~{int((ignore_thresh * 1280*720)**0.5)}px)")
        print(f"{'='*80}")

    all_preds, full_preds = run_sahi_inference(model_path, combined, img_paths)

    # 후처리 적용
    processed = {}
    for f in combined:
        img_w, img_h = img_sizes[f]
        processed[f] = apply_pipeline(all_preds[f], full_preds.get(f, []), img_w, img_h)

    # ── 기존 평가 (ignore 없음) ──
    if verbose:
        print(f"\n  --- 기존 평가 (ignore 없음) ---")
    best_f1_old, best_conf_old = 0, {}
    for c0 in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for c1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
            pc = {0: c0, 1: c1}
            tp, fp, fn, p, r, f1, ctp, cfp, cfn, _, _ = evaluate_ignore(
                all_gt, processed, combined, img_sizes,
                per_class_conf=pc, ignore_thresh=0)  # ignore 없음
            if f1 > best_f1_old:
                best_f1_old, best_conf_old = f1, pc
                best_old = (tp, fp, fn, p, r, f1, ctp, cfp, cfn)

    if verbose:
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = best_old
        print(f"  Best: F1={f1:.3f} P={p:.3f} R={r:.3f} "
              f"(TP={tp} FP={fp} FN={fn}) @c0={best_conf_old[0]}, c1={best_conf_old[1]}")
        print_class_detail(ctp, cfp, cfn)

    # ── Ignore-aware 평가 ──
    if verbose:
        print(f"\n  --- Ignore-aware 평가 (area < {ignore_thresh} 무시) ---")
    best_f1_new, best_conf_new = 0, {}
    all_results = []
    for c0 in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for c1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
            pc = {0: c0, 1: c1}
            tp, fp, fn, p, r, f1, ctp, cfp, cfn, ig_p, ig_g = evaluate_ignore(
                all_gt, processed, combined, img_sizes,
                per_class_conf=pc, ignore_thresh=ignore_thresh)
            all_results.append((f1, pc, tp, fp, fn, p, r, ctp, cfp, cfn, ig_p, ig_g))
            if f1 > best_f1_new:
                best_f1_new, best_conf_new = f1, pc
                best_new = (tp, fp, fn, p, r, f1, ctp, cfp, cfn, ig_p, ig_g)

    if verbose:
        tp, fp, fn, p, r, f1, ctp, cfp, cfn, ig_p, ig_g = best_new
        print(f"  Best: F1={f1:.3f} P={p:.3f} R={r:.3f} "
              f"(TP={tp} FP={fp} FN={fn}) @c0={best_conf_new[0]}, c1={best_conf_new[1]}")
        print(f"  Ignored: {ig_g} GT bbox, {ig_p} pred bbox")
        print_class_detail(ctp, cfp, cfn)

        # 비교
        diff = best_f1_new - best_f1_old
        print(f"\n  F1 변화: {best_f1_old:.3f} → {best_f1_new:.3f} ({'+' if diff >= 0 else ''}{diff:.3f})")

        # 상위 5개 conf 조합
        all_results.sort(key=lambda x: -x[0])
        print(f"\n  Top 5 conf 조합:")
        for rank, (f1, pc, tp, fp, fn, p, r, *_) in enumerate(all_results[:5]):
            print(f"    #{rank+1}: F1={f1:.3f} P={p:.3f} R={r:.3f} "
                  f"(TP={tp} FP={fp} FN={fn}) @c0={pc[0]}, c1={pc[1]}")

    return {
        "model": model_name,
        "old_f1": best_f1_old, "old_conf": best_conf_old,
        "new_f1": best_f1_new, "new_conf": best_conf_new,
        "old_detail": best_old,
        "new_detail": best_new,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ignore-thresh", type=float, default=IGNORE_THRESH,
                        help=f"Ignore threshold (default: {IGNORE_THRESH})")
    parser.add_argument("--compare", action="store_true",
                        help="v17/v19/v21/v23 전체 비교")
    args = parser.parse_args()

    t_total = time.time()

    # GT 로드
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

    # GT 통계
    total_gt = sum(len(all_gt[f]) for f in combined)
    ignore_count = 0
    active_count = 0
    for f in combined:
        img_w, img_h = img_sizes[f]
        for cls, x1, y1, x2, y2 in all_gt[f]:
            area = bbox_area_ratio(x1, y1, x2, y2, img_w, img_h)
            if area < args.ignore_thresh:
                ignore_count += 1
            else:
                active_count += 1

    gt_off = sum(1 for f in combined for g in all_gt[f] if g[0] == 1)
    print(f"평가 세트: {len(combined)}장 ({len(val_imgs)} val + {len(extra_imgs)} extra)")
    print(f"GT bbox 전체: {total_gt} (helmet_off: {gt_off})")
    print(f"Ignore threshold: area < {args.ignore_thresh}")
    print(f"  Active GT: {active_count} ({active_count/total_gt*100:.1f}%)")
    print(f"  Ignore GT: {ignore_count} ({ignore_count/total_gt*100:.1f}%)")

    if args.compare:
        # 전체 모델 비교
        results = []
        for name, path in MODELS_COMPARE.items():
            if not os.path.exists(path):
                print(f"\n경고: {name} 모델 없음 ({path})")
                continue
            r = eval_single_model(path, name, all_gt, combined, img_sizes, img_paths,
                                  args.ignore_thresh)
            results.append(r)

        # 종합 비교 테이블
        print(f"\n{'='*80}")
        print(f"  종합 비교 (ignore threshold = {args.ignore_thresh})")
        print(f"{'='*80}")
        print(f"  {'Model':<8} {'기존F1':>8} {'Ignore F1':>10} {'변화':>8} {'Best Conf':>12}")
        print(f"  {'-'*48}")
        for r in sorted(results, key=lambda x: -x["new_f1"]):
            diff = r["new_f1"] - r["old_f1"]
            conf_str = f"c0={r['new_conf'][0]},c1={r['new_conf'][1]}"
            print(f"  {r['model']:<8} {r['old_f1']:>8.3f} {r['new_f1']:>10.3f} "
                  f"{'+' if diff >= 0 else ''}{diff:>7.3f} {conf_str:>12}")

        # 상세 비교
        print(f"\n  상세 (Ignore-aware):")
        print(f"  {'Model':<8} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7} {'Ign_P':>7} {'Ign_G':>7}")
        print(f"  {'-'*68}")
        for r in sorted(results, key=lambda x: -x["new_f1"]):
            tp, fp, fn, p, rec, f1, ctp, cfp, cfn, ig_p, ig_g = r["new_detail"]
            print(f"  {r['model']:<8} {tp:>6} {fp:>6} {fn:>6} {p:>7.3f} {rec:>7.3f} {f1:>7.3f} {ig_p:>7} {ig_g:>7}")

    else:
        # 단일 모델
        if not os.path.exists(args.model):
            print(f"모델 없음: {args.model}")
            sys.exit(1)
        model_name = os.path.basename(os.path.dirname(os.path.dirname(args.model)))
        eval_single_model(args.model, model_name, all_gt, combined, img_sizes, img_paths,
                          args.ignore_thresh)

    elapsed = time.time() - t_total
    print(f"\n총 소요: {elapsed/60:.1f}분")
