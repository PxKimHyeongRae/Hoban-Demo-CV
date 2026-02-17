#!/usr/bin/env python3
"""v18 SAHI F1 평가 (v17 평가 스크립트 기반)

v17 최적 후처리 파이프라인 적용:
  cross_class_nms(IoU=0.3) → min_area(5e-05) → gate(conf=0.20, r=30)
  → per_class_conf(helmet_on=0.40, helmet_off=0.15)

SET A: 3k val (641장)
SET B: 3k val + verified helmet_off (729장)

사용법:
  python eval_go3k_v18.py
  python eval_go3k_v18.py --model /path/to/other/best.pt
  python eval_go3k_v18.py --no-pipeline  # 후처리 없이 비교
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
DEFAULT_MODEL = "/home/lay/hoban/hoban_go3k_v18/weights/best.pt"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}


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
    # 1. Cross-class NMS
    filtered = cross_class_nms(preds, 0.3)
    # 2. Min area
    img_area = img_w * img_h
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if ((x2-x1)*(y2-y1)) / img_area >= 5e-05]
    # 3. Gate
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-pipeline", action="store_true", help="후처리 없이 평가")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"모델 없음: {args.model}")
        sys.exit(1)

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
    gt_off = sum(1 for f in combined for g in all_gt[f] if g[0] == 1)
    print(f"모델: {args.model}")
    print(f"평가 세트: {len(combined)}장 ({len(val_imgs)} val + {len(extra_imgs)} extra)")
    print(f"GT bbox: {sum(len(all_gt[f]) for f in combined)} (helmet_off: {gt_off})")

    # SAHI 추론
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    print(f"\nSAHI 추론 (1280x720, overlap=0.15, conf=0.05)...")
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=args.model,
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
            postprocess_match_metric="IOS", verbose=0)
        all_preds[f] = [(p.category.id, p.score.value,
                         p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                        for p in r.object_prediction_list]
    print(f"  SAHI 완료: {sum(len(v) for v in all_preds.values())} preds ({time.time()-t0:.0f}s)")

    # Full-image (Gate용)
    full_preds = {}
    if not args.no_pipeline:
        from ultralytics import YOLO
        print(f"\nFull-image 추론 (Gate용, conf=0.01)...")
        yolo_model = YOLO(args.model)
        t0 = time.time()
        for i, f in enumerate(combined):
            if i % 100 == 0:
                print(f"  Full: {i}/{len(combined)}...", end="\r")
            results = yolo_model.predict(img_paths[f], conf=0.01, imgsz=1280,
                                         device="0", verbose=False)
            boxes = results[0].boxes
            full_preds[f] = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                             for j in range(len(boxes))]
        print(f"  Full 완료 ({time.time()-t0:.0f}s)")

    # ── 평가 ──
    print(f"\n{'='*80}")

    # Baseline (후처리 없음)
    print(f"  BASELINE (후처리 없음)")
    print(f"{'='*80}")
    for conf in [0.30, 0.35, 0.40, 0.45, 0.50]:
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, all_preds, combined, conf_thresh=conf)
        print(f"  conf={conf:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")

    # 후처리 파이프라인 + per-class sweep
    if not args.no_pipeline:
        print(f"\n{'='*80}")
        print(f"  PIPELINE + Per-Class Conf Sweep")
        print(f"{'='*80}")

        processed = {}
        for f in combined:
            img_w, img_h = img_sizes[f]
            processed[f] = apply_pipeline(
                all_preds.get(f, []), full_preds.get(f, []), img_w, img_h)

        best_f1, best_conf = 0, {}
        for c0 in [0.30, 0.35, 0.40, 0.45]:
            for c1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
                pc = {0: c0, 1: c1}
                tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
                    all_gt, processed, combined, per_class_conf=pc)
                if f1 > best_f1:
                    best_f1, best_conf = f1, pc
                    best_result = (tp, fp, fn, p, r, f1, ctp, cfp, cfn)

        tp, fp, fn, p, r, f1, ctp, cfp, cfn = best_result
        print(f"\n  Best: F1={f1:.3f} P={p:.3f} R={r:.3f} "
              f"(TP={tp} FP={fp} FN={fn}) @c0={best_conf[0]}, c1={best_conf[1]}")
        print_class_detail(ctp, cfp, cfn)

        # SET A (val only)
        print(f"\n  --- SET A (val {len(val_imgs)}장) ---")
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, processed, val_imgs, per_class_conf=best_conf)
        print(f"  F1={f1:.3f} P={p:.3f} R={r:.3f} (TP={tp} FP={fp} FN={fn})")
        print_class_detail(ctp, cfp, cfn)

    elapsed = time.time() - t_total
    print(f"\n소요: {elapsed/60:.1f}분")

    # v17 비교
    print(f"\n{'='*80}")
    print(f"  v17 참고: F1=0.918 (pipeline, c0=0.40, c1=0.15)")
    if not args.no_pipeline:
        diff = best_f1 - 0.918
        print(f"  v18 vs v17: {'+' if diff >= 0 else ''}{diff:.3f}")
    print(f"{'='*80}")
