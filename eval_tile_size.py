#!/usr/bin/env python3
"""
SAHI 타일 크기별 성능 비교
640x640 → 960x540 → 1280x720 → 1920x1080(풀이미지)
go2k_manual GT 대비 Precision/Recall/F1

실행: python eval_tile_size.py
"""
import os
import time
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "/home/lay/hoban/hoban_go2k_v2/weights/best.pt"
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
SAHI_CONF = 0.50

# 타일 크기 설정 (width x height)
TILE_CONFIGS = [
    (640, 640, 0.2, "640x640 (현재)"),
    (800, 800, 0.2, "800x800"),
    (960, 540, 0.2, "960x540 (2x2)"),
    (960, 960, 0.2, "960x960"),
    (1280, 720, 0.15, "1280x720 (HD절반)"),
    (1920, 1080, 0.0, "1920x1080 (풀이미지)"),
]


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


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
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            boxes.append((cls, (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h))
    return boxes


def evaluate(all_gt, all_preds):
    tp = fp = fn = 0
    for fname in all_gt:
        gt_boxes = all_gt[fname]
        pred_boxes = all_preds.get(fname, [])
        matched_gt = set()
        for p_cls, p_conf, p_x1, p_y1, p_x2, p_y2 in sorted(pred_boxes, key=lambda x: -x[1]):
            best_iou, best_gi = 0, -1
            for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if gi in matched_gt or g_cls != p_cls:
                    continue
                iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= 0.5 and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp + fp, tp, fp, fn, prec, rec, f1


# GT 로드
images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
all_gt = {}
for fname in images:
    img = Image.open(os.path.join(IMG_DIR, fname))
    img_w, img_h = img.size
    all_gt[fname] = load_gt(os.path.join(LBL_DIR, fname.replace(".jpg", ".txt")), img_w, img_h)

total_gt = sum(len(v) for v in all_gt.values())
print(f"이미지: {len(images)}장, GT: {total_gt}개")
print(f"모델: {MODEL_PATH}, SAHI conf={SAHI_CONF}\n")

# 풀이미지 추론 (SAHI 없이, 비교용)
print("풀이미지 640px 추론 (baseline)...")
full_model = YOLO(MODEL_PATH)
full_model.to("cuda:0")

all_full_preds = {}
for fname in images:
    img_path = os.path.join(IMG_DIR, fname)
    results = full_model.predict(img_path, imgsz=640, conf=SAHI_CONF, device="0", verbose=False)
    preds = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            preds.append((cls_id, conf, float(x1), float(y1), float(x2), float(y2)))
    all_full_preds[fname] = preds

n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, all_full_preds)

print(f"\n{'='*95}")
print(f"{'타일 크기':<25} {'예측':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'시간':>7}")
print("-" * 95)
print(f"{'풀이미지 640px (no SAHI)':<25} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {'---':>7}")
print("-" * 95)

# 타일 크기별 SAHI 추론
for tile_w, tile_h, overlap, label in TILE_CONFIGS:
    print(f"\n{label} 추론 중...", end=" ", flush=True)

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=MODEL_PATH,
        confidence_threshold=SAHI_CONF, device="0",
    )

    all_preds = {}
    t0 = time.time()

    for fname in images:
        img_path = os.path.join(IMG_DIR, fname)

        result = get_sliced_prediction(
            img_path, sahi_model,
            slice_height=tile_h, slice_width=tile_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS",
        )

        preds = []
        for p in result.object_prediction_list:
            preds.append((
                p.category.id, p.score.value,
                p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
            ))
        all_preds[fname] = preds

    elapsed = time.time() - t0
    n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, all_preds)
    print(f"완료 ({elapsed:.0f}s)")
    print(f"{label:<25} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {elapsed:>6.0f}s")

print(f"\n{'='*95}")
print("타일 클수록: FP 감소 + FN 증가 (소형 객체 놓침)")
print("타일 작을수록: FP 증가 + FN 감소 (소형 객체 탐지)")
