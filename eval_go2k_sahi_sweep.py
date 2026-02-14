#!/usr/bin/env python3
"""
SAHI 후처리 파라미터 sweep: postprocess_type, match_threshold, match_metric
각 조합별 conf sweep까지 한번에 평가

실행: python eval_go2k_sahi_sweep.py
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from collections import defaultdict
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="0")
parser.add_argument("--model", default="/home/lay/hoban/hoban_go2k_v2/weights/best.pt")
args = parser.parse_args()

CLASS_NAMES = {0: "person_with_helmet", 1: "person_without_helmet"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"

# SAHI 후처리 조합
SAHI_CONFIGS = [
    # (postprocess_type, match_threshold, match_metric, label)
    ("NMS",  0.5, "IOU", "NMS/0.5/IOU (현재)"),
    ("NMS",  0.4, "IOU", "NMS/0.4/IOU"),
    ("NMS",  0.3, "IOU", "NMS/0.3/IOU"),
    ("NMM",  0.5, "IOU", "NMM/0.5/IOU"),
    ("NMM",  0.4, "IOU", "NMM/0.4/IOU"),
    ("NMM",  0.3, "IOU", "NMM/0.3/IOU"),
    ("NMS",  0.3, "IOS", "NMS/0.3/IOS"),
    ("NMS",  0.4, "IOS", "NMS/0.4/IOS"),
    ("NMS",  0.5, "IOS", "NMS/0.5/IOS"),
    ("NMM",  0.3, "IOS", "NMM/0.3/IOS"),
]

CONF_THRESHOLDS = [0.15, 0.25, 0.35, 0.50]


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

    for fname in all_gt:
        gt_boxes = all_gt[fname]
        pred_boxes = [(c, s, x1, y1, x2, y2)
                      for c, s, x1, y1, x2, y2 in all_preds.get(fname, [])
                      if s >= conf_thresh]

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
                sum_tp += 1
                matched_gt.add(best_gi)
            else:
                sum_fp += 1

        for gi, (g_cls, *_) in enumerate(gt_boxes):
            if gi not in matched_gt:
                sum_fn += 1

    n_pred = sum_tp + sum_fp
    prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return n_pred, sum_tp, sum_fp, sum_fn, prec, rec, f1


# GT 로드 (한번만)
from PIL import Image
images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
all_gt = {}
for fname in images:
    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, fname.replace(".jpg", ".txt"))
    img = Image.open(img_path)
    img_w, img_h = img.size
    all_gt[fname] = load_gt(lbl_path, img_w, img_h)

total_gt = sum(len(v) for v in all_gt.values())
print(f"모델: {args.model}")
print(f"이미지: {len(images)}장, GT bbox: {total_gt}개")
print(f"SAHI 조합: {len(SAHI_CONFIGS)}개 × conf {len(CONF_THRESHOLDS)}개\n")

# 모델 로드
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=0.05,
    device=args.device,
)

# 결과 저장
results = []

for ci, (pp_type, pp_thresh, pp_metric, label) in enumerate(SAHI_CONFIGS):
    print(f"[{ci+1}/{len(SAHI_CONFIGS)}] {label} ...")
    t0 = time.time()

    all_preds = {}
    for i, fname in enumerate(images):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{len(images)}...")

        img_path = os.path.join(IMG_DIR, fname)
        result = get_sliced_prediction(
            img_path, model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            postprocess_type=pp_type,
            postprocess_match_threshold=pp_thresh,
            postprocess_match_metric=pp_metric,
            postprocess_class_agnostic=False,
        )

        preds = []
        for p in result.object_prediction_list:
            bbox = p.bbox
            preds.append((p.category.id, p.score.value, bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))
        all_preds[fname] = preds

    elapsed = time.time() - t0
    raw_count = sum(len(v) for v in all_preds.values())
    print(f"  완료 ({elapsed:.0f}s, 원시 예측: {raw_count}개)")

    for conf in CONF_THRESHOLDS:
        n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, all_preds, conf)
        results.append((label, conf, n_pred, tp, fp, fn, prec, rec, f1))

# 최종 결과 출력
print(f"\n{'=' * 90}")
print(f"SAHI 후처리 파라미터 sweep 결과")
print(f"{'=' * 90}")
print(f"{'SAHI 설정':<22} {'conf':>5} {'예측':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 90)

best_f1 = max(r[8] for r in results)
for label, conf, n_pred, tp, fp, fn, prec, rec, f1 in results:
    marker = " <<<" if f1 == best_f1 else ""
    print(f"{label:<22} {conf:>5.2f} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}{marker}")

# conf=0.35 기준 비교 (가장 실용적)
print(f"\n{'=' * 90}")
print(f"conf=0.35 기준 SAHI 설정 비교 (실용 기준)")
print(f"{'=' * 90}")
print(f"{'SAHI 설정':<22} {'예측':>6} {'FP':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 60)
for label, conf, n_pred, tp, fp, fn, prec, rec, f1 in results:
    if conf == 0.35:
        print(f"{label:<22} {n_pred:>6} {fp:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")
