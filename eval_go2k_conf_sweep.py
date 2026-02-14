#!/usr/bin/env python3
"""
go2k_v2: conf 임계값별 + SAHI NMS 비교 테스트
낮은 conf로 한번만 추론 → 여러 임계값에서 필터링 평가

실행: python eval_go2k_conf_sweep.py
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="0")
parser.add_argument("--model", default="/home/lay/hoban/hoban_go2k_v2/weights/best.pt")
args = parser.parse_args()

CLASS_NAMES = {0: "person_with_helmet", 1: "person_without_helmet"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"

CONF_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


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
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate(all_gt, all_preds, conf_thresh, iou_thresh=0.5):
    sum_tp = sum_fp = sum_fn = 0
    cls_tp = defaultdict(int)
    cls_fp = defaultdict(int)
    cls_fn = defaultdict(int)

    for fname in all_gt:
        gt_boxes = all_gt[fname]
        pred_boxes = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in all_preds.get(fname, []) if s >= conf_thresh]

        matched_gt = set()
        pred_sorted = sorted(enumerate(pred_boxes), key=lambda x: -x[1][1])

        for pi, (p_cls, p_conf, p_x1, p_y1, p_x2, p_y2) in pred_sorted:
            best_iou = 0
            best_gi = -1
            for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if gi in matched_gt or g_cls != p_cls:
                    continue
                iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thresh and best_gi >= 0:
                cls_tp[p_cls] += 1
                matched_gt.add(best_gi)
            else:
                cls_fp[p_cls] += 1

        for gi, (g_cls, *_) in enumerate(gt_boxes):
            if gi not in matched_gt:
                cls_fn[g_cls] += 1

    sum_tp = sum(cls_tp.values())
    sum_fp = sum(cls_fp.values())
    sum_fn = sum(cls_fn.values())
    n_pred = sum_tp + sum_fp

    prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return n_pred, sum_tp, sum_fp, sum_fn, prec, rec, f1


# 모델 로드 (conf=0.05로 최대한 많이 수집)
print(f"모델: {args.model}")
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=0.05,
    device=args.device,
)

images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"go2k_manual: {len(images)}장 추론 중 (conf=0.05로 전체 수집)...\n")

all_gt = {}
all_preds = {}

for i, fname in enumerate(images):
    if i % 100 == 0:
        print(f"  {i}/{len(images)}...")

    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, fname.replace(".jpg", ".txt"))

    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size

    all_gt[fname] = load_gt(lbl_path, img_w, img_h)

    result = get_sliced_prediction(
        img_path, model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
    )

    preds = []
    for p in result.object_prediction_list:
        bbox = p.bbox
        preds.append((p.category.id, p.score.value, bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))
    all_preds[fname] = preds

total_gt = sum(len(v) for v in all_gt.values())
total_raw = sum(len(v) for v in all_preds.values())
print(f"\n추론 완료! GT: {total_gt}개, 원시 예측(conf≥0.05): {total_raw}개\n")

# conf별 평가
print(f"{'=' * 75}")
print(f"go2k_v2 + SAHI: conf 임계값별 결과 (IoU≥0.5)")
print(f"{'=' * 75}")
print(f"{'conf':>6} {'예측':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
print("-" * 75)

for conf in CONF_THRESHOLDS:
    n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, all_preds, conf)
    marker = " <<<" if f1 == max(evaluate(all_gt, all_preds, c)[6] for c in CONF_THRESHOLDS) else ""
    print(f"{conf:>6.2f} {n_pred:>7} {tp:>6} {fp:>6} {fn:>6} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}{marker}")

print("-" * 75)
print("<<< = best F1")
