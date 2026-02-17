#!/usr/bin/env python3
"""v17 최적 후처리 적용 시각화 (Gate + Cross-class NMS + Per-class conf)

최적 파이프라인:
  cross_class_nms(IoU=0.3) → min_area(5e-05) → gate(conf=0.20, r=30px)
  → per_class_conf(helmet_on=0.40, helmet_off=0.15)
"""
import os, time, logging
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

IMG_DIR = "/home/lay/hoban/datasets/cvat_helmet_off/results/images"
MODEL = "/home/lay/hoban/hoban_go3k_v17/weights/best.pt"
OUT_DIR = "/home/lay/hoban/datasets/cvat_helmet_off/v17_vis_opt"
os.makedirs(OUT_DIR, exist_ok=True)

# 최적 파라미터
CROSS_NMS_IOU = 0.3
MIN_AREA = 5e-05
GATE_CONF = 0.20
GATE_RADIUS = 30
CONF_ON = 0.40   # helmet_on
CONF_OFF = 0.15  # helmet_off

COLORS = {0: (0, 200, 0), 1: (255, 40, 40)}
LABELS = {0: "ON", 1: "OFF"}
BG_COLORS = {0: (0, 140, 0), 1: (180, 0, 0)}

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def cross_class_nms(preds, iou_thresh):
    if len(preds) <= 1:
        return preds
    sorted_preds = sorted(preds, key=lambda x: -x[1])
    keep, suppressed = [], set()
    for i, (c1, s1, x1a, y1a, x2a, y2a) in enumerate(sorted_preds):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            c2 = sorted_preds[j][0]
            if c1 != c2:
                iou = compute_iou((x1a, y1a, x2a, y2a),
                                  (sorted_preds[j][2], sorted_preds[j][3],
                                   sorted_preds[j][4], sorted_preds[j][5]))
                if iou >= iou_thresh:
                    suppressed.add(j)
    return keep


def apply_pipeline(preds, full_raw, img_w, img_h):
    # 1. Cross-class NMS
    filtered = cross_class_nms(preds, CROSS_NMS_IOU)
    # 2. Min area
    img_area = img_w * img_h
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if ((x2-x1)*(y2-y1)) / img_area >= MIN_AREA]
    # 3. Gate
    gates = [(x1,y1,x2,y2) for conf,x1,y1,x2,y2 in full_raw if conf >= GATE_CONF]
    if gates:
        gated = []
        for c,s,x1,y1,x2,y2 in filtered:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            for gx1,gy1,gx2,gy2 in gates:
                gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
                if abs(cx-gcx) <= GATE_RADIUS and abs(cy-gcy) <= GATE_RADIUS:
                    gated.append((c,s,x1,y1,x2,y2))
                    break
        filtered = gated
    # 4. Per-class conf
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if s >= (CONF_ON if c == 0 else CONF_OFF)]
    return filtered


all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
print(f"시각화 대상: {len(all_imgs)}장")

# SAHI 모델
print("v17 SAHI 모델 로드...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=MODEL,
    confidence_threshold=0.05, device="0", image_size=1280)

# Full-image 모델 (Gate용)
print("v17 Full-image 모델 로드...")
yolo_model = YOLO(MODEL)

t0 = time.time()
has_off_count = 0

for i, fname in enumerate(all_imgs):
    if i % 50 == 0:
        print(f"  {i}/{len(all_imgs)}...", end="\r")

    img_path = os.path.join(IMG_DIR, fname)

    # SAHI 추론
    r = get_sliced_prediction(
        img_path, sahi_model,
        slice_height=720, slice_width=1280,
        overlap_height_ratio=0.15, overlap_width_ratio=0.15,
        perform_standard_pred=True,
        postprocess_type="NMS", postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS", verbose=0)

    raw_preds = [(p.category.id, p.score.value,
                  p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                 for p in r.object_prediction_list]

    # Full-image 추론 (Gate)
    results = yolo_model.predict(img_path, conf=0.01, imgsz=1280,
                                 device="0", verbose=False)
    boxes = results[0].boxes
    full_raw = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                for j in range(len(boxes))]

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    # 후처리 적용
    preds = apply_pipeline(raw_preds, full_raw, img_w, img_h)

    has_off = any(c == 1 for c, *_ in preds)
    if not has_off:
        continue

    has_off_count += 1
    draw = ImageDraw.Draw(img)

    # helmet_on 먼저, helmet_off 나중
    preds.sort(key=lambda x: x[0])

    for cls_id, conf, x1, y1, x2, y2 in preds:
        color = COLORS[cls_id]
        bg = BG_COLORS[cls_id]
        label = f"{LABELS[cls_id]} {conf:.2f}"

        lw = 3 if cls_id == 1 else 2
        for offset in range(lw):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]

        if cls_id == 1:
            tx, ty = x1, y1 - th - 6
            if ty < 0:
                ty = y1 + 2
        else:
            tx, ty = x1, y2 + 2
            if ty + th > img.height:
                ty = y1 - th - 6

        draw.rectangle([tx-1, ty-1, tx+tw+4, ty+th+3], fill=bg)
        draw.text((tx+2, ty), label, fill="white", font=font)

    img.save(os.path.join(OUT_DIR, fname), quality=90)

elapsed = time.time() - t0
print(f"\n완료: {has_off_count}/{len(all_imgs)}장 시각화 ({elapsed:.0f}s)")
print(f"출력: {OUT_DIR}")
print(f"\n후처리: cross_nms(IoU={CROSS_NMS_IOU}) + area(>{MIN_AREA}) + gate(conf={GATE_CONF},r={GATE_RADIUS}) + per_class(ON≥{CONF_ON},OFF≥{CONF_OFF})")
