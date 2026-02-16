#!/usr/bin/env python3
"""
종합 Clean 재평가: 모든 핵심 설정을 Clean 125장에서 평가
- 모델별 단독 성능 (v2, v3, v5)
- Per-class conf sweep
- WBF vs NMS 앙상블 비교
- 다양한 앙상블 조합 (2-model, 3-model)
- Overlap sweep (SAHI 파라미터)

SAHI 추론은 1회만 하고, conf threshold는 post-hoc 평가.
"""
import os, sys, time
import numpy as np
from collections import defaultdict
from itertools import product
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion, nms

CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
TRAIN_DIR = "/home/lay/hoban/datasets_go2k_v2/train/images"

MODELS = {
    "v2": "/home/lay/hoban/hoban_go2k_v2/weights/best.pt",
    "v3": "/home/lay/hoban/hoban_go2k_v3/weights/best.pt",
    "v5": "/home/lay/hoban/hoban_go2k_v5/weights/best.pt",
}

# ── helpers ──────────────────────────────────────────────

def load_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls,
                          (cx - w/2) * img_w, (cy - h/2) * img_h,
                          (cx + w/2) * img_w, (cy + h/2) * img_h))
    return boxes

def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

def evaluate(all_gt, all_preds, image_set, per_class_conf=None, conf_thresh=None, iou_thresh=0.5):
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
            if bv >= iou_thresh and bi >= 0:
                tp += 1; ctp[gts[bi][0]] += 1; matched.add(bi)
            else:
                fp += 1; cfp[pc] += 1
        for gi in range(len(gts)):
            if gi not in matched: fn += 1; cfn[gts[gi][0]] += 1
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn

def run_sahi(model_path, images, tile_w=1280, tile_h=720, overlap=0.15, image_size=None):
    kwargs = {"image_size": image_size} if image_size else {}
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.05, device="0", **kwargs)
    preds = {}
    for i, f in enumerate(images):
        if i % 50 == 0: print(f"  {i}/{len(images)}...", end="\r")
        r = get_sliced_prediction(
            os.path.join(IMG_DIR, f), model,
            slice_height=tile_h, slice_width=tile_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", postprocess_class_agnostic=False)
        preds[f] = [(p.category.id, p.score.value, p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                     for p in r.object_prediction_list]
    n = sum(len(v) for v in preds.values())
    print(f"  Done: {n} preds ({n/len(images):.1f}/img)")
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

def nms_ensemble(preds_list, img_sizes, iou_thr=0.4, weights=None):
    merged = {}
    for f in preds_list[0]:
        w, h = img_sizes[f]
        bl, sl, ll = [], [], []
        for p in preds_list:
            raw = p.get(f, [])
            if not raw: bl.append(np.empty((0, 4))); sl.append([]); ll.append([]); continue
            bl.append([[x1/w,y1/h,x2/w,y2/h] for _,_,x1,y1,x2,y2 in raw])
            sl.append([s for _,s,_,_,_,_ in raw])
            ll.append([c for c,_,_,_,_,_ in raw])
        if all(len(s)==0 for s in sl): merged[f] = []; continue
        mb, ms, ml = nms(bl, sl, ll, weights=weights, iou_thr=iou_thr)
        merged[f] = [(int(l), float(s), b[0]*w, b[1]*h, b[2]*w, b[3]*h) for b,s,l in zip(mb,ms,ml)]
    return merged


# ── main ─────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    # ── Split images ──
    train_orig = set(f for f in os.listdir(TRAIN_DIR)
                     if f.startswith("cam") and "_x" not in f and f.endswith(".jpg"))
    all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    clean = sorted(f for f in all_imgs if f not in train_orig)
    leaked = sorted(f for f in all_imgs if f in train_orig)
    print(f"Images: {len(all_imgs)} total, {len(clean)} clean, {len(leaked)} leaked\n")

    # ── Load GT ──
    all_gt, img_sizes = {}, {}
    for f in all_imgs:
        img = Image.open(os.path.join(IMG_DIR, f))
        img_sizes[f] = img.size
        all_gt[f] = load_gt(os.path.join(LBL_DIR, f.replace(".jpg",".txt")), *img.size)

    gt_clean = sum(len(all_gt.get(f, [])) for f in clean)
    gt_all = sum(len(all_gt.get(f, [])) for f in all_imgs)
    print(f"GT: Clean {gt_clean} bbox, All {gt_all} bbox\n")

    # ══════════════════════════════════════════════════════
    # PHASE 1: SAHI inference (3 models, default overlap=0.15)
    # ══════════════════════════════════════════════════════
    print("=" * 80)
    print("PHASE 1: SAHI 추론 (3 models)")
    print("=" * 80)

    cache = {}
    for name, path in MODELS.items():
        isz = 1280 if name == "v3" else None
        print(f"\n[{name}] SAHI 1280x720, overlap=0.15" + (f", image_size={isz}" if isz else ""))
        t0 = time.time()
        cache[name] = run_sahi(path, all_imgs, image_size=isz)
        print(f"  Time: {time.time()-t0:.0f}s")

    # ══════════════════════════════════════════════════════
    # PHASE 2: 모델별 단독 성능 (conf sweep)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 2: 모델별 단독 성능 - conf sweep (Clean 125장)")
    print("=" * 80)

    confs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    print(f"\n{'모델':<6} {'conf':>5} | {'TP':>4} {'FP':>4} {'FN':>4} | {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 55)

    best_single = {}
    for name in ["v2", "v3", "v5"]:
        best_f1, best_conf = 0, 0
        for conf in confs:
            tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, cache[name], clean, conf_thresh=conf)
            marker = ""
            if f1 > best_f1: best_f1, best_conf = f1, conf
            print(f"{name:<6} {conf:>5.2f} | {tp:>4} {fp:>4} {fn:>4} | {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
        best_single[name] = (best_f1, best_conf)
        print(f"  >> {name} best: F1={best_f1:.3f} @conf={best_conf}")
        print()

    # ══════════════════════════════════════════════════════
    # PHASE 3: Per-class conf sweep (Clean 125장)
    # ══════════════════════════════════════════════════════
    print("=" * 80)
    print("PHASE 3: Per-class conf sweep (Clean 125장)")
    print("=" * 80)

    c0_range = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    c1_range = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    for name in ["v2", "v3", "v5"]:
        print(f"\n[{name}] per-class conf sweep:")
        print(f"  {'c0':>5} {'c1':>5} | {'TP':>4} {'FP':>4} {'FN':>4} | {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print(f"  " + "-" * 52)
        best_f1, best_c0, best_c1 = 0, 0, 0
        for c0, c1 in product(c0_range, c1_range):
            tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, cache[name], clean, per_class_conf={0:c0, 1:c1})
            if f1 > best_f1: best_f1, best_c0, best_c1 = f1, c0, c1
            if f1 >= best_f1 - 0.005:  # only print top results
                print(f"  {c0:>5.2f} {c1:>5.2f} | {tp:>4} {fp:>4} {fn:>4} | {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
        print(f"  >> {name} best per-class: F1={best_f1:.3f} @c0={best_c0}, c1={best_c1}")

    # ══════════════════════════════════════════════════════
    # PHASE 4: 앙상블 비교 (WBF vs NMS, 다양한 조합)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 4: 앙상블 비교 - WBF vs NMS (Clean 125장)")
    print("=" * 80)

    combos = [
        ("v2+v3",    ["v2","v3"],      [1, 1]),
        ("v2+v5",    ["v2","v5"],      [1, 1]),
        ("v3+v5",    ["v3","v5"],      [1, 1]),
        ("v2+v3+v5", ["v2","v3","v5"], [1, 1, 0.5]),
        ("v2+v3+v5 eq", ["v2","v3","v5"], [1, 1, 1]),
    ]

    # Per-class conf for ensemble eval
    ens_c0_range = [0.40, 0.45, 0.50, 0.55, 0.60]
    ens_c1_range = [0.20, 0.25, 0.30, 0.35, 0.40]

    print(f"\n{'앙상블':<16} {'방법':>4} {'iou':>4} {'c0':>5} {'c1':>5} | {'TP':>4} {'FP':>4} {'FN':>4} | {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 82)

    best_ens_f1 = 0
    best_ens_label = ""

    for combo_name, model_names, weights in combos:
        preds_list = [cache[m] for m in model_names]

        for method_name, method_fn in [("WBF", wbf), ("NMS", nms_ensemble)]:
            for iou_thr in [0.3, 0.4, 0.5]:
                ens_preds = method_fn(preds_list, img_sizes, iou_thr=iou_thr, weights=weights)

                local_best_f1 = 0
                for c0, c1 in product(ens_c0_range, ens_c1_range):
                    tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, ens_preds, clean, per_class_conf={0:c0, 1:c1})
                    if f1 > local_best_f1:
                        local_best_f1 = f1
                        if f1 > best_ens_f1:
                            best_ens_f1 = f1
                            best_ens_label = f"{combo_name} {method_name}/iou={iou_thr} c0={c0} c1={c1}"

                # Print best for this combo+method+iou
                for c0, c1 in product(ens_c0_range, ens_c1_range):
                    tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, ens_preds, clean, per_class_conf={0:c0, 1:c1})
                    if f1 >= local_best_f1 - 0.002:
                        print(f"{combo_name:<16} {method_name:>4} {iou_thr:>4.1f} {c0:>5.2f} {c1:>5.2f} | {tp:>4} {fp:>4} {fn:>4} | {p:>6.3f} {r:>6.3f} {f1:>6.3f}")

    print(f"\n>> 앙상블 최고: F1={best_ens_f1:.3f} ({best_ens_label})")

    # ══════════════════════════════════════════════════════
    # PHASE 5: Overlap sweep (SAHI 파라미터, v2 기준)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PHASE 5: SAHI Overlap sweep (v2 모델, Clean 125장)")
    print("=" * 80)

    overlap_results = {}
    for ov in [0.05, 0.10, 0.20, 0.25, 0.30]:
        print(f"\n[v2] overlap={ov:.2f} ...")
        t0 = time.time()
        preds_ov = run_sahi(MODELS["v2"], all_imgs, overlap=ov)
        elapsed = time.time() - t0
        # Find best conf
        best_f1_ov, best_conf_ov = 0, 0
        for conf in confs:
            tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, preds_ov, clean, conf_thresh=conf)
            if f1 > best_f1_ov: best_f1_ov, best_conf_ov = f1, conf
        # Find best per-class
        best_f1_pc, best_c0_pc, best_c1_pc = 0, 0, 0
        for c0, c1 in product(c0_range, c1_range):
            tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, preds_ov, clean, per_class_conf={0:c0, 1:c1})
            if f1 > best_f1_pc: best_f1_pc, best_c0_pc, best_c1_pc = f1, c0, c1

        overlap_results[ov] = (best_f1_ov, best_conf_ov, best_f1_pc, best_c0_pc, best_c1_pc)
        print(f"  Time: {elapsed:.0f}s | uniform best: F1={best_f1_ov:.3f}@{best_conf_ov} | per-class best: F1={best_f1_pc:.3f}@c0={best_c0_pc},c1={best_c1_pc}")

    # Add default overlap=0.15 result
    best_f1_def, best_conf_def = best_single["v2"]
    best_f1_pc_def = 0
    for c0, c1 in product(c0_range, c1_range):
        tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, cache["v2"], clean, per_class_conf={0:c0, 1:c1})
        if f1 > best_f1_pc_def: best_f1_pc_def = f1
    overlap_results[0.15] = (best_f1_def, best_conf_def, best_f1_pc_def, 0, 0)

    print(f"\n  Overlap 비교 요약:")
    print(f"  {'overlap':>8} | {'uniform F1':>11} | {'per-class F1':>13}")
    print(f"  " + "-" * 40)
    for ov in sorted(overlap_results.keys()):
        uf1, _, pf1, _, _ = overlap_results[ov]
        marker = " <-- default" if ov == 0.15 else ""
        print(f"  {ov:>8.2f} | {uf1:>11.3f} | {pf1:>13.3f}{marker}")

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"종합 요약 (Clean 125장 기준, 총 소요: {elapsed_total/60:.1f}분)")
    print(f"{'=' * 80}")

    print(f"\n  [단독 모델 best]")
    for name in ["v2", "v3", "v5"]:
        f1, conf = best_single[name]
        print(f"    {name}: F1={f1:.3f} @conf={conf}")

    print(f"\n  [앙상블 best]")
    print(f"    {best_ens_label}: F1={best_ens_f1:.3f}")

    print(f"\n  [기존 기준선 (참고)]")
    print(f"    v2 단독 conf=0.55: F1=0.845")
    print(f"    v2 per-class c0=0.65 c1=0.35: F1=0.856")
    print(f"    v2+v3+v5 WBF c0=0.50 c1=0.30: F1=0.891")

    print(f"\nDone.")
