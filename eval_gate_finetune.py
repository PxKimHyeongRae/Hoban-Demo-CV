#!/usr/bin/env python3
"""
풀이미지 게이트 파라미터 정밀 튜닝
conf: 0.05~0.30 x radius: 30~150 = 48개 조합

1차 결과 캐시 활용 (SAHI + 풀이미지 재추론)

실행: python eval_gate_finetune.py
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
# 정밀 탐색 범위
GATE_CONFS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
GATE_RADII = [30, 40, 50, 60, 75, 100]


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
            if len(parts) < 5: continue
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
                if gi in matched_gt or g_cls != p_cls: continue
                iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                if iou > best_iou: best_iou, best_gi = iou, gi
            if best_iou >= 0.5 and best_gi >= 0:
                tp += 1; matched_gt.add(best_gi)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp + fp, tp, fp, fn, prec, rec, f1


def point_near_any(px, py, gates, radius):
    for gx1, gy1, gx2, gy2 in gates:
        gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
        if abs(px - gcx) <= radius and abs(py - gcy) <= radius:
            return True
    return False


# 모델 로드
print(f"모델: {MODEL_PATH}")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=MODEL_PATH,
    confidence_threshold=SAHI_CONF, device="0",
)
full_model = YOLO(MODEL_PATH)
full_model.to("cuda:0")

images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"이미지: {len(images)}장\n")

# 1단계: SAHI 추론 + 풀이미지 추론 (모든 conf에서 재사용하기 위해 conf=0.01로)
print("추론 중 (SAHI + 풀이미지 conf=0.01)...")
t0 = time.time()

all_gt = {}
all_sahi_preds = {}
all_full_raw = {}  # conf, bbox 모두 저장

for i, fname in enumerate(images):
    if i % 100 == 0 and i > 0:
        print(f"  {i}/{len(images)}...")

    img_path = os.path.join(IMG_DIR, fname)
    img = Image.open(img_path)
    img_w, img_h = img.size

    all_gt[fname] = load_gt(os.path.join(LBL_DIR, fname.replace(".jpg", ".txt")), img_w, img_h)

    # SAHI
    result = get_sliced_prediction(
        img_path, sahi_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS",
    )
    all_sahi_preds[fname] = [
        (p.category.id, p.score.value, p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
        for p in result.object_prediction_list
    ]

    # 풀이미지 (매우 낮은 conf로 모든 후보 수집)
    full_results = full_model.predict(img_path, imgsz=640, conf=0.01, device="0", verbose=False)
    full_dets = []
    for r in full_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            full_dets.append((conf, float(x1), float(y1), float(x2), float(y2)))
    all_full_raw[fname] = full_dets

print(f"  완료 ({time.time()-t0:.1f}s)\n")

# 2단계: conf x radius 그리드 서치
print(f"{'=' * 95}")
print(f"풀이미지 게이트 파라미터 그리드 서치 (SAHI conf={SAHI_CONF})")
print(f"{'=' * 95}")
print(f"{'gate_conf':>10} {'radius':>7} {'예측':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FP감소':>7}")
print("-" * 95)

# baseline
n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, all_sahi_preds)
baseline_fp = fp
print(f"{'baseline':>10} {'---':>7} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {'---':>7}")
print("-" * 95)

best_f1 = 0
best_config = None
results = []

for gate_conf in GATE_CONFS:
    for radius in GATE_RADII:
        gated_preds = {}
        for fname in all_sahi_preds:
            # 풀이미지 결과를 gate_conf로 필터
            gates = [(x1, y1, x2, y2) for c, x1, y1, x2, y2 in all_full_raw[fname] if c >= gate_conf]

            filtered = []
            for cls_id, conf, x1, y1, x2, y2 in all_sahi_preds[fname]:
                cx, cy = (x1+x2)/2, (y1+y2)/2
                if not gates or point_near_any(cx, cy, gates, radius):
                    filtered.append((cls_id, conf, x1, y1, x2, y2))
            gated_preds[fname] = filtered

        n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, gated_preds)
        fp_reduction = baseline_fp - fp
        results.append((gate_conf, radius, n_pred, tp, fp, fn, prec, rec, f1, fp_reduction))

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_config = (gate_conf, radius)
            marker = " <<<"

        print(f"{gate_conf:>10.2f} {radius:>7} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {fp_reduction:>+7}{marker}")

print(f"\n{'=' * 95}")
print(f"최적: gate_conf={best_config[0]}, radius={best_config[1]}px → F1={best_f1:.3f}")

# Recall >= 0.89 조건에서 최고 F1
print(f"\n[Recall >= 0.89 조건 최적]")
filtered_results = [(gc, r, np_, tp, fp, fn, p, rec, f1, fpr) for gc, r, np_, tp, fp, fn, p, rec, f1, fpr in results if rec >= 0.89]
if filtered_results:
    best = max(filtered_results, key=lambda x: x[8])
    print(f"  conf={best[0]}, radius={best[1]}px → P={best[6]:.3f}, R={best[7]:.3f}, F1={best[8]:.3f}, FP={best[4]} (-{best[9]})")

# Recall >= 0.895 조건
print(f"\n[Recall >= 0.895 조건 최적]")
filtered_results = [(gc, r, np_, tp, fp, fn, p, rec, f1, fpr) for gc, r, np_, tp, fp, fn, p, rec, f1, fpr in results if rec >= 0.895]
if filtered_results:
    best = max(filtered_results, key=lambda x: x[8])
    print(f"  conf={best[0]}, radius={best[1]}px → P={best[6]:.3f}, R={best[7]:.3f}, F1={best[8]:.3f}, FP={best[4]} (-{best[9]})")
