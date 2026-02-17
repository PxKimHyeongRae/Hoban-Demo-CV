#!/usr/bin/env python3
"""v17 helmet_off 탐지 결과 시각화 (bbox + confidence)"""
import os, time, logging
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageDraw, ImageFont

IMG_DIR = "/home/lay/hoban/datasets/cvat_helmet_off/results/images"
MODEL = "/home/lay/hoban/hoban_go3k_v17/weights/best.pt"
OUT_DIR = "/home/lay/hoban/datasets/cvat_helmet_off/v17_vis"
os.makedirs(OUT_DIR, exist_ok=True)

# 색상: helmet_on=초록, helmet_off=빨강
COLORS = {0: (0, 200, 0), 1: (255, 40, 40)}
LABELS = {0: "ON", 1: "OFF"}
BG_COLORS = {0: (0, 140, 0), 1: (180, 0, 0)}

# 폰트
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except:
    font = ImageFont.load_default()
    font_sm = font

all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
print(f"시각화 대상: {len(all_imgs)}장")

print("v17 모델 로드...")
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=MODEL,
    confidence_threshold=0.15, device="0",
    image_size=1280)

t0 = time.time()
has_off_count = 0

for i, fname in enumerate(all_imgs):
    if i % 50 == 0:
        print(f"  {i}/{len(all_imgs)}...", end="\r")

    img_path = os.path.join(IMG_DIR, fname)
    r = get_sliced_prediction(
        img_path, model,
        slice_height=720, slice_width=1280,
        overlap_height_ratio=0.15, overlap_width_ratio=0.15,
        perform_standard_pred=True,
        postprocess_type="NMS", postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS", verbose=0)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    has_off = False
    preds = []
    for p in r.object_prediction_list:
        cls_id = p.category.id
        conf = p.score.value
        x1, y1 = int(p.bbox.minx), int(p.bbox.miny)
        x2, y2 = int(p.bbox.maxx), int(p.bbox.maxy)
        preds.append((cls_id, conf, x1, y1, x2, y2))
        if cls_id == 1:
            has_off = True

    if not has_off:
        continue

    has_off_count += 1

    # helmet_on 먼저, helmet_off 나중에 (위에 그려지도록)
    preds.sort(key=lambda x: x[0])

    for cls_id, conf, x1, y1, x2, y2 in preds:
        color = COLORS[cls_id]
        bg = BG_COLORS[cls_id]
        label = f"{LABELS[cls_id]} {conf:.2f}"

        # bbox
        lw = 3 if cls_id == 1 else 2
        for offset in range(lw):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)

        # 텍스트 위치: helmet_off는 위, helmet_on은 아래
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if cls_id == 1:  # OFF: 상단
            tx, ty = x1, y1 - th - 6
            if ty < 0:
                ty = y1 + 2
        else:  # ON: 하단
            tx, ty = x1, y2 + 2
            if ty + th > img.height:
                ty = y1 - th - 6

        # 배경 박스 + 텍스트
        draw.rectangle([tx-1, ty-1, tx+tw+4, ty+th+3], fill=bg)
        draw.text((tx+2, ty), label, fill="white", font=font)

    img.save(os.path.join(OUT_DIR, fname), quality=90)

elapsed = time.time() - t0
print(f"\n완료: {has_off_count}/{len(all_imgs)}장 시각화 ({elapsed:.0f}s)")
print(f"출력: {OUT_DIR}")
