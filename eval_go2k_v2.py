#!/usr/bin/env python3
"""
go2k_v2 best.pt → go2k_manual GT 대비 정확도 평가 (SAHI)

실행: python eval_go2k_v2.py
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="0")
parser.add_argument("--conf", type=float, default=0.15)
parser.add_argument("--iou-thresh", type=float, default=0.5)
parser.add_argument("--model", default="/home/lay/hoban/hoban_go2k_v2/weights/best.pt")
args = parser.parse_args()

CLASS_NAMES = {0: "person_with_helmet", 1: "person_without_helmet"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"


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


def match_predictions(gt_boxes, pred_boxes, iou_thresh):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    matched_gt = set()

    pred_sorted = sorted(enumerate(pred_boxes), key=lambda x: -x[1][1])

    for pi, (p_cls, p_conf, p_x1, p_y1, p_x2, p_y2) in pred_sorted:
        best_iou = 0
        best_gi = -1
        for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if g_cls != p_cls:
                continue
            iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= iou_thresh and best_gi >= 0:
            tp[p_cls] += 1
            matched_gt.add(best_gi)
        else:
            fp[p_cls] += 1

    for gi, (g_cls, *_) in enumerate(gt_boxes):
        if gi not in matched_gt:
            fn[g_cls] += 1

    return tp, fp, fn


# 모델 로드
print(f"모델: {args.model}")
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=args.conf,
    device=args.device,
)

images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"go2k_manual 평가: {len(images)}장 (IoU≥{args.iou_thresh}, conf≥{args.conf})")

total_tp = defaultdict(int)
total_fp = defaultdict(int)
total_fn = defaultdict(int)
total_gt = 0
total_pred = 0

for i, fname in enumerate(images):
    if i % 50 == 0:
        print(f"  {i}/{len(images)}...")

    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, fname.replace(".jpg", ".txt"))

    from PIL import Image
    img = Image.open(img_path)
    img_w, img_h = img.size

    gt_boxes = load_gt(lbl_path, img_w, img_h)
    total_gt += len(gt_boxes)

    result = get_sliced_prediction(
        img_path, model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    )

    pred_boxes = []
    for p in result.object_prediction_list:
        bbox = p.bbox
        pred_boxes.append((
            p.category.id, p.score.value,
            bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        ))
    total_pred += len(pred_boxes)

    tp, fp, fn = match_predictions(gt_boxes, pred_boxes, args.iou_thresh)
    for cls in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
        total_tp[cls] += tp[cls]
        total_fp[cls] += fp[cls]
        total_fn[cls] += fn[cls]

# 결과
print(f"\n{'=' * 60}")
print(f"go2k_v2 + SAHI 평가 결과 (vs 수동 라벨링 GT)")
print(f"{'=' * 60}")
print(f"GT bbox: {total_gt}개 / 예측 bbox: {total_pred}개")
print(f"IoU 임계값: {args.iou_thresh}\n")

all_classes = sorted(set(list(total_tp.keys()) + list(total_fp.keys()) + list(total_fn.keys())))

print(f"{'클래스':<25} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
print("-" * 70)

sum_tp = sum_fp = sum_fn = 0
for cls in all_classes:
    tp = total_tp[cls]
    fp_val = total_fp[cls]
    fn_val = total_fn[cls]
    prec = tp / (tp + fp_val) if (tp + fp_val) > 0 else 0
    rec = tp / (tp + fn_val) if (tp + fn_val) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{CLASS_NAMES.get(cls, str(cls)):<25} {tp:>6} {fp_val:>6} {fn_val:>6} {prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")
    sum_tp += tp
    sum_fp += fp_val
    sum_fn += fn_val

total_prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
total_rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0
print("-" * 70)
print(f"{'전체':<25} {sum_tp:>6} {sum_fp:>6} {sum_fn:>6} {total_prec:>8.3f} {total_rec:>8.3f} {total_f1:>8.3f}")
