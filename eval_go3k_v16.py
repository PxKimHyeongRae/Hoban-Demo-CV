#!/usr/bin/env python3
"""v16 SAHI 평가: 3k val 641장 (clean, leakage 없음)"""
import os, time
from collections import defaultdict
from itertools import product
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

IMG_DIR = "/home/lay/hoban/datasets/3k_finetune/val/images"
LBL_DIR = "/home/lay/hoban/datasets/3k_finetune/val/labels"
MODEL = "/home/lay/hoban/hoban_go3k_v16_640/weights/best.pt"


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
            boxes.append((cls, (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                          (cx + w / 2) * img_w, (cy + h / 2) * img_h))
    return boxes


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def evaluate(all_gt, all_preds, image_set, per_class_conf=None, conf_thresh=None):
    tp = fp = fn = 0
    ctp, cfp, cfn = defaultdict(int), defaultdict(int), defaultdict(int)
    for fname in image_set:
        gts = all_gt.get(fname, [])
        raw = all_preds.get(fname, [])
        if per_class_conf:
            preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in raw
                     if s >= per_class_conf.get(c, 0.5)]
        elif conf_thresh is not None:
            preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in raw
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
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn


if __name__ == "__main__":
    t_start = time.time()

    # Load images & GT
    all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    print(f"Eval images: {len(all_imgs)}")

    all_gt, img_sizes = {}, {}
    total_bbox = 0
    for f in all_imgs:
        img = Image.open(os.path.join(IMG_DIR, f))
        img_sizes[f] = img.size
        gt = load_gt(os.path.join(LBL_DIR, f.replace(".jpg", ".txt")), *img.size)
        all_gt[f] = gt
        total_bbox += len(gt)
    print(f"Total GT bbox: {total_bbox}")

    # SAHI inference
    print(f"\n[v16] SAHI inference (1280x720 tiles)...")
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=MODEL,
        confidence_threshold=0.05, device="0")

    all_preds = {}
    for i, f in enumerate(all_imgs):
        if i % 50 == 0:
            print(f"  {i}/{len(all_imgs)}...", end="\r")
        r = get_sliced_prediction(
            os.path.join(IMG_DIR, f), model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", postprocess_class_agnostic=False)
        all_preds[f] = [(p.category.id, p.score.value,
                         p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                        for p in r.object_prediction_list]
    print(f"  Done: {sum(len(v) for v in all_preds.values())} total preds")

    # Uniform conf sweep
    print(f"\n{'='*60}")
    print("Uniform Confidence Sweep")
    print(f"{'='*60}")
    confs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_f1u, best_cu = 0, 0
    for conf in confs:
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, all_preds, all_imgs, conf_thresh=conf)
        marker = ""
        if f1 > best_f1u:
            best_f1u, best_cu = f1, conf
            marker = " <-- best"
        print(f"  conf={conf:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn}){marker}")

    # Per-class conf sweep
    print(f"\n{'='*60}")
    print("Per-Class Confidence Sweep")
    print(f"{'='*60}")
    c0_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    c1_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    best_f1p, best_c0, best_c1 = 0, 0, 0
    for c0, c1 in product(c0_range, c1_range):
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, all_preds, all_imgs, per_class_conf={0: c0, 1: c1})
        if f1 > best_f1p:
            best_f1p, best_c0, best_c1 = f1, c0, c1

    # Show best per-class result with details
    tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
        all_gt, all_preds, all_imgs, per_class_conf={0: best_c0, 1: best_c1})
    print(f"  Best: c0={best_c0:.2f}, c1={best_c1:.2f}")
    print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")
    for cls_id, cls_name in {0: "helmet_on", 1: "helmet_off"}.items():
        ct, cf, cm = ctp[cls_id], cfp[cls_id], cfn[cls_id]
        cp = ct / (ct + cf) if ct + cf else 0
        cr = ct / (ct + cm) if ct + cm else 0
        cf1 = 2 * cp * cr / (cp + cr) if cp + cr else 0
        print(f"    {cls_name}: P={cp:.3f} R={cr:.3f} F1={cf1:.3f} (TP={ct} FP={cf} FN={cm})")

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Summary (소요: {elapsed/60:.1f}분)")
    print(f"{'='*60}")
    print(f"  Uniform best:   F1={best_f1u:.3f} @conf={best_cu}")
    print(f"  Per-class best: F1={best_f1p:.3f} @c0={best_c0},c1={best_c1}")
    print(f"  Eval set: 3k val {len(all_imgs)}장, {total_bbox} bbox (clean)")
