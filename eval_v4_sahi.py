#!/usr/bin/env python3
"""v5 모델 SAHI 평가 (go2k_manual GT 대비, v2 최적 설정)"""
import os, time
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
MODEL = "/home/lay/hoban/hoban_go2k_v5/weights/best.pt"


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def load_gt(lbl, w, h):
    boxes = []
    if not os.path.exists(lbl):
        return boxes
    for line in open(lbl):
        p = line.strip().split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = [float(x) for x in p[1:5]]
        boxes.append((c, (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return boxes


def evaluate(gt, preds, conf):
    tp = fp = fn = 0
    for f in gt:
        gb = gt[f]
        pb = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in preds.get(f, []) if s >= conf]
        mg = set()
        for pc, ps, px1, py1, px2, py2 in sorted(pb, key=lambda x: -x[1]):
            bi, bg = 0, -1
            for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gb):
                if gi in mg or gc != pc:
                    continue
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > bi:
                    bi, bg = iou, gi
            if bi >= 0.5 and bg >= 0:
                tp += 1
                mg.add(bg)
            else:
                fp += 1
        fn += len(gb) - len(mg)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return tp + fp, tp, fp, fn, p, r, f1


# GT 로드
images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
all_gt = {}
for f in images:
    img = Image.open(os.path.join(IMG_DIR, f))
    all_gt[f] = load_gt(os.path.join(LBL_DIR, f.replace(".jpg", ".txt")), *img.size)

total_gt = sum(len(v) for v in all_gt.values())
print(f"GT: {len(images)}장, {total_gt}개")
print(f"모델: {MODEL}\n")

model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=MODEL,
    confidence_threshold=0.05, device="0",
)

CONFS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# 1) SAHI + perform_standard_pred (v2 최적 설정)
print("=" * 80)
print("v5 SAHI 1280x720 + perform_standard_pred")
print("=" * 80)
t0 = time.time()
preds_std = {}
for i, f in enumerate(images):
    if i % 100 == 0:
        print(f"  {i}/{len(images)}...")
    r = get_sliced_prediction(
        os.path.join(IMG_DIR, f), model,
        slice_height=720, slice_width=1280,
        overlap_height_ratio=0.15, overlap_width_ratio=0.15,
        postprocess_type="NMS", postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS",
        perform_standard_pred=True,
    )
    preds_std[f] = [(p.category.id, p.score.value,
                     p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                    for p in r.object_prediction_list]
print(f"  완료 ({time.time()-t0:.0f}s)\n")

print(f"{'conf':>5} {'예측':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7}")
print("-" * 60)
for c in CONFS:
    n, tp, fp, fn, p, r, f1 = evaluate(all_gt, preds_std, c)
    print(f"{c:>5.2f} {n:>6} {tp:>6} {fp:>6} {fn:>6} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")

# 비교 기준
print(f"\n{'='*80}")
print("비교 기준")
print("=" * 80)
print("v2 +std_pred  conf=0.55: F1=0.914, P=0.893, R=0.935, FP=188")
print("v3 +std_pred  conf=0.25: F1=0.896, P=0.939, R=0.857, FP=94")
