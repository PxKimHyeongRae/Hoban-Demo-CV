#!/usr/bin/env python3
"""v17 SAHI 평가: 3k val + verified helmet_off (통합)

v17: yolo26m.pt, 1280px 학습, best epoch 51, mAP50=0.9611
평가 세트:
  A) 3k val 641장 (v16과 동일 비교)
  B) 3k val + verified helmet_off 86장 = ~727장 (helmet_off bbox 강화)
"""
import os, sys, time, logging
from collections import defaultdict
from itertools import product
from PIL import Image

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Paths
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
MODEL = "/home/lay/hoban/hoban_go3k_v17/weights/best.pt"


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
            boxes.append((cls, (cx - w/2)*img_w, (cy - h/2)*img_h,
                          (cx + w/2)*img_w, (cy + h/2)*img_h))
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


def print_result(tp, fp, fn, p, r, f1, ctp, cfp, cfn, label=""):
    print(f"  {label}P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")
    for cls_id, cls_name in {0: "helmet_on", 1: "helmet_off"}.items():
        ct, cf, cm = ctp[cls_id], cfp[cls_id], cfn[cls_id]
        cp = ct/(ct+cf) if ct+cf else 0
        cr = ct/(ct+cm) if ct+cm else 0
        cf1 = 2*cp*cr/(cp+cr) if cp+cr else 0
        print(f"    {cls_name}: P={cp:.3f} R={cr:.3f} F1={cf1:.3f} (TP={ct} FP={cf} FN={cm})")


def sahi_infer(model, img_path):
    from sahi.predict import get_sliced_prediction
    r = get_sliced_prediction(
        img_path, model,
        slice_height=720, slice_width=1280,
        overlap_height_ratio=0.15, overlap_width_ratio=0.15,
        perform_standard_pred=True,
        postprocess_type="NMS", postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS", postprocess_class_agnostic=False,
        verbose=0)
    return [(p.category.id, p.score.value,
             p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
            for p in r.object_prediction_list]


if __name__ == "__main__":
    t_start = time.time()

    # ── Load GT: 3k val ──
    val_imgs = sorted(f for f in os.listdir(VAL_IMG) if f.endswith(".jpg"))
    all_gt, img_paths = {}, {}
    for f in val_imgs:
        path = os.path.join(VAL_IMG, f)
        img = Image.open(path)
        img_paths[f] = path
        all_gt[f] = load_gt(os.path.join(VAL_LBL, f.replace(".jpg", ".txt")), *img.size)

    # ── Load GT: verified helmet_off ──
    extra_imgs = []
    if os.path.isdir(EXTRA_IMG):
        for f in sorted(os.listdir(EXTRA_IMG)):
            if f.endswith(".jpg") and f not in all_gt:
                path = os.path.join(EXTRA_IMG, f)
                img = Image.open(path)
                img_paths[f] = path
                all_gt[f] = load_gt(os.path.join(EXTRA_LBL, f.replace(".jpg", ".txt")), *img.size)
                extra_imgs.append(f)

    combined_imgs = val_imgs + extra_imgs
    gt_bbox_val = sum(len(all_gt[f]) for f in val_imgs)
    gt_bbox_extra = sum(len(all_gt[f]) for f in extra_imgs)
    gt_off_val = sum(1 for f in val_imgs for g in all_gt[f] if g[0] == 1)
    gt_off_extra = sum(1 for f in extra_imgs for g in all_gt[f] if g[0] == 1)

    print(f"3k val: {len(val_imgs)}장, {gt_bbox_val} bbox (helmet_off: {gt_off_val})")
    print(f"Extra verified: {len(extra_imgs)}장, {gt_bbox_extra} bbox (helmet_off: {gt_off_extra})")
    print(f"Combined: {len(combined_imgs)}장, {gt_bbox_val+gt_bbox_extra} bbox (helmet_off: {gt_off_val+gt_off_extra})")

    # ── SAHI inference (v17, image_size=1280) ──
    from sahi import AutoDetectionModel
    print(f"\n[v17] SAHI inference (1280x720 tiles, image_size=1280)...")
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=MODEL,
        confidence_threshold=0.05, device="0",
        image_size=1280)

    all_preds = {}
    for i, f in enumerate(combined_imgs):
        if i % 50 == 0:
            print(f"  {i}/{len(combined_imgs)}...", end="\r")
        all_preds[f] = sahi_infer(model, img_paths[f])
    total_preds = sum(len(v) for v in all_preds.values())
    print(f"  Done: {total_preds} total preds ({time.time()-t_start:.0f}s)")

    # ══════════════════════════════════════════════
    # SET A: 3k val only (v16 비교용)
    # ══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"SET A: 3k val only ({len(val_imgs)}장) — v16 비교")
    print(f"{'='*60}")

    # Uniform conf sweep
    print("\n[A] Uniform Confidence Sweep")
    confs = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_f1a, best_ca = 0, 0
    for conf in confs:
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, all_preds, val_imgs, conf_thresh=conf)
        marker = ""
        if f1 > best_f1a:
            best_f1a, best_ca = f1, conf
            marker = " <-- best"
        print(f"  conf={conf:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn}){marker}")

    # Per-class conf sweep
    print("\n[A] Per-Class Confidence Sweep")
    c0_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    c1_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    best_f1ap, best_c0a, best_c1a = 0, 0, 0
    for c0, c1 in product(c0_range, c1_range):
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, all_preds, val_imgs, per_class_conf={0: c0, 1: c1})
        if f1 > best_f1ap:
            best_f1ap, best_c0a, best_c1a = f1, c0, c1

    tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
        all_gt, all_preds, val_imgs, per_class_conf={0: best_c0a, 1: best_c1a})
    print(f"  Best: c0={best_c0a:.2f}, c1={best_c1a:.2f}")
    print_result(tp, fp, fn, p, r, f1, ctp, cfp, cfn)

    # ══════════════════════════════════════════════
    # SET B: Combined (3k val + verified helmet_off)
    # ══════════════════════════════════════════════
    if extra_imgs:
        print(f"\n{'='*60}")
        print(f"SET B: Combined ({len(combined_imgs)}장, helmet_off bbox: {gt_off_val+gt_off_extra})")
        print(f"{'='*60}")

        # Uniform
        print("\n[B] Uniform Confidence Sweep")
        best_f1b, best_cb = 0, 0
        for conf in confs:
            tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, all_preds, combined_imgs, conf_thresh=conf)
            marker = ""
            if f1 > best_f1b:
                best_f1b, best_cb = f1, conf
                marker = " <-- best"
            print(f"  conf={conf:.2f}: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp} FP={fp} FN={fn}){marker}")

        # Per-class
        print("\n[B] Per-Class Confidence Sweep")
        best_f1bp, best_c0b, best_c1b = 0, 0, 0
        for c0, c1 in product(c0_range, c1_range):
            tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
                all_gt, all_preds, combined_imgs, per_class_conf={0: c0, 1: c1})
            if f1 > best_f1bp:
                best_f1bp, best_c0b, best_c1b = f1, c0, c1

        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, all_preds, combined_imgs, per_class_conf={0: best_c0b, 1: best_c1b})
        print(f"  Best: c0={best_c0b:.2f}, c1={best_c1b:.2f}")
        print_result(tp, fp, fn, p, r, f1, ctp, cfp, cfn)

    # ══════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUMMARY (소요: {elapsed/60:.1f}분)")
    print(f"{'='*60}")
    print(f"  v17 model: best epoch 51, mAP50=0.9611 (1280px)")
    print(f"")
    print(f"  [SET A] 3k val {len(val_imgs)}장:")
    print(f"    Uniform best:   F1={best_f1a:.3f} @conf={best_ca}")
    print(f"    Per-class best: F1={best_f1ap:.3f} @c0={best_c0a},c1={best_c1a}")
    if extra_imgs:
        print(f"")
        print(f"  [SET B] Combined {len(combined_imgs)}장 (helmet_off: {gt_off_val+gt_off_extra} bbox):")
        print(f"    Uniform best:   F1={best_f1b:.3f} @conf={best_cb}")
        print(f"    Per-class best: F1={best_f1bp:.3f} @c0={best_c0b},c1={best_c1b}")
    print(f"\n  v16 baseline: F1=0.885 (uniform), F1=0.898 (per-class)")
