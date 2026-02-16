#!/usr/bin/env python3
"""v4, v6, v7 Clean 125장 평가 + 기존 v2/v3/v5와 앙상블 비교"""
import os, time, numpy as np
from collections import defaultdict
from itertools import product
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion

CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
TRAIN_DIR = "/home/lay/hoban/datasets_go2k_v2/train/images"

MODELS = {
    "v2": "/home/lay/hoban/hoban_go2k_v2/weights/best.pt",
    "v3": "/home/lay/hoban/hoban_go2k_v3/weights/best.pt",
    "v5": "/home/lay/hoban/hoban_go2k_v5/weights/best.pt",
    "v4": "/home/lay/hoban/hoban_go2k_v4/weights/best.pt",
    "v6": "/home/lay/hoban/hoban_go2k_v6/weights/best.pt",
    "v7": "/home/lay/hoban/hoban_go2k_v7/weights/best.pt",
}

def load_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path): return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h))
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
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in raw if s >= per_class_conf.get(c, 0.5)]
        elif conf_thresh is not None:
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in raw if s >= conf_thresh]
        else:
            preds = raw
        matched = set()
        for _, (pc, ps, px1, py1, px2, py2) in sorted(enumerate(preds), key=lambda x: -x[1][1]):
            bi, bv = -1, 0
            for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
                if gi in matched or gc != pc: continue
                iou = compute_iou((px1,py1,px2,py2), (gx1,gy1,gx2,gy2))
                if iou > bv: bv, bi = iou, gi
            if bv >= 0.5 and bi >= 0:
                tp += 1; ctp[gts[bi][0]] += 1; matched.add(bi)
            else:
                fp += 1; cfp[pc] += 1
        for gi in range(len(gts)):
            if gi not in matched: fn += 1; cfn[gts[gi][0]] += 1
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1

def run_sahi(model_path, images, image_size=None):
    kwargs = {"image_size": image_size} if image_size else {}
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.05, device="0", **kwargs)
    preds = {}
    for i, f in enumerate(images):
        if i % 100 == 0: print(f"  {i}/{len(images)}...", end="\r")
        r = get_sliced_prediction(
            os.path.join(IMG_DIR, f), model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", postprocess_class_agnostic=False)
        preds[f] = [(p.category.id, p.score.value, p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                     for p in r.object_prediction_list]
    print(f"  Done: {sum(len(v) for v in preds.values())} preds")
    return preds

def wbf(preds_list, img_sizes, iou_thr=0.4, weights=None):
    merged = {}
    for f in preds_list[0]:
        w, h = img_sizes[f]
        bl, sl, ll = [], [], []
        for p in preds_list:
            raw = p.get(f, [])
            if not raw: bl.append([]); sl.append([]); ll.append([]); continue
            bl.append([[x1/w,y1/h,x2/w,y2/h] for _,_,x1,y1,x2,y2 in raw])
            sl.append([s for _,s,_,_,_,_ in raw])
            ll.append([c for c,_,_,_,_,_ in raw])
        if all(len(b)==0 for b in bl): merged[f] = []; continue
        mb, ms, ml = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=iou_thr, skip_box_thr=0.0001)
        merged[f] = [(int(l), float(s), b[0]*w, b[1]*h, b[2]*w, b[3]*h) for b,s,l in zip(mb,ms,ml)]
    return merged

if __name__ == "__main__":
    t_start = time.time()

    # Split
    train_orig = set(f for f in os.listdir(TRAIN_DIR)
                     if f.startswith("cam") and "_x" not in f and f.endswith(".jpg"))
    all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    clean = sorted(f for f in all_imgs if f not in train_orig)
    print(f"Images: {len(all_imgs)} total, {len(clean)} clean\n")

    # Load GT
    all_gt, img_sizes = {}, {}
    for f in all_imgs:
        img = Image.open(os.path.join(IMG_DIR, f))
        img_sizes[f] = img.size
        all_gt[f] = load_gt(os.path.join(LBL_DIR, f.replace(".jpg",".txt")), *img.size)

    # ── SAHI inference for all 6 models ──
    cache = {}
    for name in ["v2", "v3", "v4", "v5", "v6", "v7"]:
        isz = 1280 if name == "v3" else None
        print(f"[{name}] SAHI inference..." + (f" (image_size={isz})" if isz else ""))
        t0 = time.time()
        cache[name] = run_sahi(MODELS[name], all_imgs, image_size=isz)
        print(f"  Time: {time.time()-t0:.0f}s\n")

    # ── Phase 1: 모델별 단독 성능 ──
    print("=" * 80)
    print("단독 모델 성능 (Clean 125장)")
    print("=" * 80)

    confs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    c0_range = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    c1_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    single_best = {}
    for name in ["v2", "v3", "v4", "v5", "v6", "v7"]:
        # uniform conf
        best_f1u, best_confu = 0, 0
        for conf in confs:
            tp,fp,fn,p,r,f1 = evaluate(all_gt, cache[name], clean, conf_thresh=conf)
            if f1 > best_f1u: best_f1u, best_confu = f1, conf
        # per-class
        best_f1p, best_c0, best_c1 = 0, 0, 0
        for c0, c1 in product(c0_range, c1_range):
            tp,fp,fn,p,r,f1 = evaluate(all_gt, cache[name], clean, per_class_conf={0:c0, 1:c1})
            if f1 > best_f1p: best_f1p, best_c0, best_c1 = f1, c0, c1
        single_best[name] = max(best_f1u, best_f1p)
        print(f"  {name}: uniform best F1={best_f1u:.3f}@{best_confu} | per-class best F1={best_f1p:.3f}@c0={best_c0},c1={best_c1}")

    # ── Phase 2: 2-model 앙상블 (모든 조합) ──
    print(f"\n{'=' * 80}")
    print("2-model WBF 앙상블 (Clean 125장)")
    print("=" * 80)

    models = ["v2", "v3", "v4", "v5", "v6", "v7"]
    ens_c0 = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    ens_c1 = [0.15, 0.20, 0.25, 0.30, 0.35]

    pair_results = {}
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1, m2 = models[i], models[j]
            ens_preds = wbf([cache[m1], cache[m2]], img_sizes, iou_thr=0.4)
            best_f1, best_c0, best_c1 = 0, 0, 0
            for c0, c1 in product(ens_c0, ens_c1):
                tp,fp,fn,p,r,f1 = evaluate(all_gt, ens_preds, clean, per_class_conf={0:c0, 1:c1})
                if f1 > best_f1: best_f1, best_c0, best_c1 = f1, c0, c1
            pair_results[f"{m1}+{m2}"] = best_f1
            print(f"  {m1}+{m2}: F1={best_f1:.3f} @c0={best_c0},c1={best_c1}")

    # ── Phase 3: 3-model 앙상블 (상위 조합) ──
    print(f"\n{'=' * 80}")
    print("3-model WBF 앙상블 (Clean 125장)")
    print("=" * 80)

    # Test key 3-model combos
    triple_combos = [
        ("v2+v3+v5", ["v2","v3","v5"], [1,1,0.5]),
        ("v3+v5+v7", ["v3","v5","v7"], [1,1,0.5]),
        ("v3+v5+v4", ["v3","v5","v4"], [1,1,0.5]),
        ("v3+v5+v6", ["v3","v5","v6"], [1,1,0.5]),
        ("v3+v4+v5+v6", ["v3","v4","v5","v6"], [1,0.5,1,0.5]),
        ("v2+v3+v4+v5", ["v2","v3","v4","v5"], [1,1,0.5,1]),
        ("all6", ["v2","v3","v4","v5","v6","v7"], [1,1,0.5,1,0.5,0.5]),
    ]

    for label, mnames, weights in triple_combos:
        ens_preds = wbf([cache[m] for m in mnames], img_sizes, iou_thr=0.4, weights=weights)
        best_f1, best_c0, best_c1 = 0, 0, 0
        for c0, c1 in product(ens_c0, ens_c1):
            tp,fp,fn,p,r,f1 = evaluate(all_gt, ens_preds, clean, per_class_conf={0:c0, 1:c1})
            if f1 > best_f1: best_f1, best_c0, best_c1 = f1, c0, c1
        print(f"  {label}: F1={best_f1:.3f} @c0={best_c0},c1={best_c1}")

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"종합 요약 (소요: {elapsed/60:.1f}분)")
    print(f"{'=' * 80}")
    print(f"\n  단독 모델 순위:")
    for name, f1 in sorted(single_best.items(), key=lambda x: -x[1]):
        print(f"    {name}: F1={f1:.3f}")
    print(f"\n  2-model 앙상블 Top 5:")
    for label, f1 in sorted(pair_results.items(), key=lambda x: -x[1])[:5]:
        print(f"    {label}: F1={f1:.3f}")
    print(f"\nDone.")
