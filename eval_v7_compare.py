#!/usr/bin/env python3
"""
v7 vs v2 공정 비교: v7의 clean eval set (438장)에서 SAHI 평가
- v7: leakage-free 학습
- v2: leakage 있는 학습 (기존 최고 모델)
"""
import os, time
from collections import defaultdict
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion

CLASS_NAMES = {0: "person_with_helmet", 1: "person_without_helmet"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
TRAIN_DIR = "/home/lay/hoban/datasets_go2k_v2/train/images"

MODELS = {
    "v2": "/home/lay/hoban/hoban_go2k_v2/weights/best.pt",
    "v3": "/home/lay/hoban/hoban_go2k_v3/weights/best.pt",
    "v5": "/home/lay/hoban/hoban_go2k_v5/weights/best.pt",
    "v7": "/home/lay/hoban/hoban_go2k_v7/weights/best.pt",
}

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
            x1, y1 = (cx - w/2) * img_w, (cy - h/2) * img_h
            x2, y2 = (cx + w/2) * img_w, (cy + h/2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
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
        elif conf_thresh:
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
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn

def run_sahi(model_path, images, img_sizes, image_size=None):
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
    # Get clean eval set (v7 valid = not in v2 train)
    train_orig = set(f for f in os.listdir(TRAIN_DIR)
                     if f.startswith("cam") and "_x" not in f and f.endswith(".jpg"))
    all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    clean = sorted(f for f in all_imgs if f not in train_orig)  # 125장
    leaked = sorted(f for f in all_imgs if f in train_orig)      # 479장

    print(f"Clean: {len(clean)}장, Leaked: {len(leaked)}장, 전체: {len(all_imgs)}장")

    # Load GT & sizes
    all_gt, img_sizes = {}, {}
    for f in all_imgs:
        img = Image.open(os.path.join(IMG_DIR, f))
        img_sizes[f] = img.size
        all_gt[f] = load_gt(os.path.join(LBL_DIR, f.replace(".jpg",".txt")), *img.size)

    # SAHI inference
    cache = {}
    for name, path in [("v2", MODELS["v2"]), ("v3", MODELS["v3"]),
                         ("v5", MODELS["v5"]), ("v7", MODELS["v7"])]:
        isz = 1280 if name == "v3" else None
        print(f"\n[{name}] SAHI inference...")
        t0 = time.time()
        cache[name] = run_sahi(path, all_imgs, img_sizes, image_size=isz)
        print(f"  Time: {time.time()-t0:.0f}s")

    # Ensembles
    wbf_v235 = wbf([cache["v2"], cache["v3"], cache["v5"]], img_sizes, weights=[1,1,0.5])
    wbf_v735 = wbf([cache["v7"], cache["v3"], cache["v5"]], img_sizes, weights=[1,1,0.5])
    wbf_v2v7 = wbf([cache["v2"], cache["v7"]], img_sizes, weights=[1,1])

    # Evaluate
    configs = [
        ("v2 단독 conf=0.55", cache["v2"], {"conf_thresh": 0.55}),
        ("v7 단독 conf=0.55", cache["v7"], {"conf_thresh": 0.55}),
        ("v2 per-class c0=0.65 c1=0.35", cache["v2"], {"per_class_conf": {0:0.65, 1:0.35}}),
        ("v7 per-class c0=0.65 c1=0.35", cache["v7"], {"per_class_conf": {0:0.65, 1:0.35}}),
        ("v2+v3+v5 WBF c0=0.50 c1=0.30", wbf_v235, {"per_class_conf": {0:0.50, 1:0.30}}),
        ("v7+v3+v5 WBF c0=0.50 c1=0.30", wbf_v735, {"per_class_conf": {0:0.50, 1:0.30}}),
        ("v2+v7 WBF c0=0.50 c1=0.30", wbf_v2v7, {"per_class_conf": {0:0.50, 1:0.30}}),
    ]

    for group_name, group in [("Clean (125장)", clean), ("Leaked (479장)", leaked), ("전체 (604장)", all_imgs)]:
        print(f"\n{'='*80}")
        print(f" {group_name}")
        print(f"{'='*80}")
        print(f"{'설정':<40} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
        print("-"*80)
        for label, preds, kw in configs:
            tp,fp,fn,p,r,f1,_,_,_ = evaluate(all_gt, preds, group, **kw)
            print(f"{label:<40} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")

    print("\nDone.")
