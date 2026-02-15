#!/usr/bin/env python3
"""
풀이미지 게이트 효과 측정:
  1) 풀이미지 640px 추론 → 대략적 사람 위치 후보 (low conf)
  2) SAHI 타일 추론 → 정밀 탐지
  3) SAHI 결과 중 풀이미지 후보 근처만 채택

go2k_manual GT 대비 정확도 비교 (gate ON vs OFF)

실행: python eval_fullimage_gate.py
"""
import os
import time
import argparse
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/home/lay/hoban/hoban_go2k_v2/weights/best.pt")
args = parser.parse_args()

IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
CLASS_NAMES = {0: "person_with_helmet", 1: "person_without_helmet"}

# 풀이미지 게이트 설정
GATE_CONF = 0.05        # 풀이미지 최소 conf (낮게 → 사람 후보 넓게 잡기)
GATE_RADIUS = 100       # 풀이미지 탐지 중심에서 반경 (px)
SAHI_CONF = 0.50        # SAHI 최종 conf
GATE_RADII = [50, 75, 100, 150, 200]  # 반경별 테스트


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
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
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def evaluate(all_gt, all_preds, iou_thresh=0.5):
    tp = fp = fn = 0
    for fname in all_gt:
        gt_boxes = all_gt[fname]
        pred_boxes = all_preds.get(fname, [])
        matched_gt = set()
        pred_sorted = sorted(pred_boxes, key=lambda x: -x[1])

        for p_cls, p_conf, p_x1, p_y1, p_x2, p_y2 in pred_sorted:
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
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp + fp, tp, fp, fn, prec, rec, f1


def point_near_any(px, py, gates, radius):
    """(px, py)가 gates 중 하나의 중심에서 radius 이내인지"""
    for gx1, gy1, gx2, gy2 in gates:
        gcx = (gx1 + gx2) / 2
        gcy = (gy1 + gy2) / 2
        if abs(px - gcx) <= radius and abs(py - gcy) <= radius:
            return True
    return False


# 모델 로드
print(f"모델: {args.model}")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=SAHI_CONF,
    device="0",
)

full_model = YOLO(args.model)
full_model.to("cuda:0")
print("모델 로드 완료\n")

# GT 로드 + 추론
images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
all_gt = {}
all_sahi_preds = {}  # SAHI만
all_full_dets = {}   # 풀이미지 탐지 (gate용)

print(f"이미지: {len(images)}장\n")
print("1) SAHI + 풀이미지 추론...")
t0 = time.time()

for i, fname in enumerate(images):
    if i % 100 == 0 and i > 0:
        print(f"  {i}/{len(images)}...")

    img_path = os.path.join(IMG_DIR, fname)
    img = Image.open(img_path)
    img_w, img_h = img.size

    # GT
    lbl_path = os.path.join(LBL_DIR, fname.replace(".jpg", ".txt"))
    all_gt[fname] = load_gt(lbl_path, img_w, img_h)

    # SAHI 추론
    result = get_sliced_prediction(
        img_path, sahi_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS",
    )

    sahi_preds = []
    for p in result.object_prediction_list:
        bbox = p.bbox
        sahi_preds.append((
            p.category.id, p.score.value,
            bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        ))
    all_sahi_preds[fname] = sahi_preds

    # 풀이미지 추론 (low conf)
    full_results = full_model.predict(
        img_path, imgsz=640, conf=GATE_CONF,
        device="0", verbose=False
    )
    full_dets = []
    for r in full_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            full_dets.append((float(x1), float(y1), float(x2), float(y2)))
    all_full_dets[fname] = full_dets

elapsed = time.time() - t0
print(f"  완료 ({elapsed:.1f}s)\n")

total_gt = sum(len(v) for v in all_gt.values())
total_sahi = sum(len(v) for v in all_sahi_preds.values())
total_full = sum(len(v) for v in all_full_dets.values())
print(f"GT: {total_gt}개, SAHI 예측: {total_sahi}개, 풀이미지 후보: {total_full}개\n")

# 평가: SAHI only (baseline)
n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, all_sahi_preds)
print(f"{'=' * 80}")
print(f"{'설정':<35} {'예측':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print(f"{'-' * 80}")
print(f"{'SAHI only (baseline)':<35} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

# 평가: 풀이미지 게이트 (반경별)
for radius in GATE_RADII:
    gated_preds = {}
    for fname in all_sahi_preds:
        gates = all_full_dets.get(fname, [])
        filtered = []
        for cls_id, conf, x1, y1, x2, y2 in all_sahi_preds[fname]:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if not gates or point_near_any(cx, cy, gates, radius):
                filtered.append((cls_id, conf, x1, y1, x2, y2))
        gated_preds[fname] = filtered

    n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, gated_preds)
    removed = sum(len(all_sahi_preds[f]) - len(gated_preds[f]) for f in all_sahi_preds)
    label = f"gate r={radius}px (conf>={GATE_CONF})"
    print(f"{label:<35} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}  (-{removed})")

# 게이트 conf 변화 테스트 (반경 100px 고정)
print(f"\n{'=' * 80}")
print(f"게이트 conf 변화 (반경=100px)")
print(f"{'설정':<35} {'예측':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print(f"{'-' * 80}")

for gate_conf in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]:
    gated_preds = {}
    for fname in all_sahi_preds:
        # 풀이미지 재필터 (conf 기준)
        full_results_filtered = full_model.predict(
            os.path.join(IMG_DIR, fname), imgsz=640, conf=gate_conf,
            device="0", verbose=False
        )
        gates = []
        for r in full_results_filtered:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                gates.append((float(x1), float(y1), float(x2), float(y2)))

        filtered = []
        for cls_id, conf, x1, y1, x2, y2 in all_sahi_preds[fname]:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if not gates or point_near_any(cx, cy, gates, 100):
                filtered.append((cls_id, conf, x1, y1, x2, y2))
        gated_preds[fname] = filtered

    n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, gated_preds)
    removed = sum(len(all_sahi_preds[f]) - len(gated_preds[f]) for f in all_sahi_preds)
    label = f"gate conf={gate_conf}, r=100px"
    print(f"{label:<35} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}  (-{removed})")
