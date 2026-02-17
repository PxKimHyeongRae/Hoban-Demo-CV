#!/usr/bin/env python3
"""기존 helmet_off 이미지를 v17 모델로 재탐지 → CVAT 패키징"""
import os, zipfile, time, logging
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

IMG_DIR = "/home/lay/hoban/datasets/cvat_helmet_off/results/images"
MODEL = "/home/lay/hoban/hoban_go3k_v17/weights/best.pt"
OUT_DIR = "/home/lay/hoban/datasets/cvat_helmet_off/v17_cvat"
CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]

os.makedirs(OUT_DIR, exist_ok=True)

# 1. 이미지 목록
all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
print(f"재탐지 대상: {len(all_imgs)}장")

# 2. v17 SAHI 모델
print(f"v17 모델 로드 (image_size=1280)...")
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=MODEL,
    confidence_threshold=0.15, device="0",
    image_size=1280)

# 3. 추론
results = []  # (fname, label_text, n_off)
t0 = time.time()
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

    img = Image.open(img_path)
    img_w, img_h = img.size

    lines = []
    n_off = 0
    for p in r.object_prediction_list:
        cls_id = p.category.id
        x1, y1 = p.bbox.minx, p.bbox.miny
        x2, y2 = p.bbox.maxx, p.bbox.maxy
        cx = ((x1+x2)/2) / img_w
        cy = ((y1+y2)/2) / img_h
        w = (x2-x1) / img_w
        h = (y2-y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if cls_id == 1:
            n_off += 1

    if n_off > 0:
        results.append((fname, "\n".join(lines), n_off))

elapsed = time.time() - t0
print(f"\n추론 완료: {len(results)}/{len(all_imgs)}장에서 helmet_off 탐지 ({elapsed:.0f}s)")

# 4. CVAT 패키징
ann_path = os.path.join(OUT_DIR, "annotations.zip")
with zipfile.ZipFile(ann_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("obj.data",
        f"classes = {len(CLASS_NAMES)}\n"
        f"train = data/train.txt\n"
        f"names = data/obj.names\n"
        f"backup = backup/\n")
    zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")
    train_lines = []
    for fname, labels, _ in results:
        train_lines.append(f"data/obj_train_data/{fname}")
        zf.writestr(f"obj_train_data/{fname.replace('.jpg', '.txt')}", labels + "\n")
    zf.writestr("train.txt", "\n".join(train_lines) + "\n")

img_path = os.path.join(OUT_DIR, "images.zip")
with zipfile.ZipFile(img_path, "w", zipfile.ZIP_STORED) as zf:
    for i, (fname, _, _) in enumerate(results):
        zf.write(os.path.join(IMG_DIR, fname), fname)

print(f"\n{'='*60}")
print(f"완료! {len(results)}장 (helmet_off 포함)")
print(f"  annotations.zip: {os.path.getsize(ann_path)/1024/1024:.1f}MB")
print(f"  images.zip: {os.path.getsize(img_path)/1024/1024:.1f}MB")
print(f"  출력: {OUT_DIR}")
