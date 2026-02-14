#!/usr/bin/env python3
"""
captures cam1+cam2 주간(8~18시) → SAHI 탐지 → 탐지된 이미지만 라벨 저장
30초 간격 샘플링, go2k 타임스탬프 제외
3000장 탐지 도달 시 조기 종료

실행: python detect_captures_batch.py
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import os
import time
from collections import defaultdict

CAP_DIR = "/home/lay/video_indoor/static/captures"
GO2K_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
OUT_DIR = "/home/lay/hoban/datasets/captures_labels_3k"
MODEL_PATH = "/home/lay/hoban/hoban_go2k_v2/weights/best.pt"
TARGET_COUNT = 3000
MIN_INTERVAL = 30
CONF = 0.50

os.makedirs(OUT_DIR, exist_ok=True)


def parse_ts(fname):
    parts = fname.split("_")
    if len(parts) < 3:
        return None, None
    try:
        date = int(parts[1])
        t = parts[2]
        h, m, s = int(t[:2]), int(t[2:4]), int(t[4:6])
        return date * 86400 + h * 3600 + m * 60 + s, h
    except (ValueError, IndexError):
        return None, None


# go2k 타임스탬프 제외
go2k_keys = set()
for f in os.listdir(GO2K_DIR):
    parts = f.split("_")
    if len(parts) >= 3:
        go2k_keys.add("_".join(parts[:3]))

# cam1+cam2 주간 이미지 수집 (30초 간격)
targets = []
for cam in ["cam1", "cam2"]:
    cam_dir = os.path.join(CAP_DIR, cam)
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))

    last_ts = None
    count = 0
    for f in files:
        ts, h = parse_ts(f)
        if ts is None or not (8 <= h < 18):
            continue
        key = "_".join(f.split("_")[:3])
        if key in go2k_keys:
            continue
        if last_ts and (ts - last_ts) < MIN_INTERVAL:
            continue
        targets.append((cam, f))
        last_ts = ts
        count += 1

    print(f"{cam}: {count}장 (주간, 30초 간격, go2k 제외)")

print(f"\n대상: {len(targets)}장, 목표: {TARGET_COUNT}장 탐지")
print(f"모델: {MODEL_PATH}, conf={CONF}")
print(f"SAHI: 640x640, overlap=0.2, NMS/0.4/IOS\n")

# 모델 로드
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=MODEL_PATH,
    confidence_threshold=CONF,
    device="0",
)

# 추론
t0 = time.time()
total_boxes = 0
img_with_det = 0
processed_files = []  # (cam, fname) 기록 (나중에 중복 검사용)

for i, (cam, fname) in enumerate(targets):
    if img_with_det >= TARGET_COUNT:
        print(f"\n  목표 {TARGET_COUNT}장 도달! 조기 종료.")
        break

    if i % 500 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(targets) - i)
        print(f"  [{i}/{len(targets)}] elapsed={elapsed/60:.1f}m, ETA={eta/60:.1f}m, "
              f"탐지={img_with_det}, bbox={total_boxes}")

    img_path = os.path.join(CAP_DIR, cam, fname)

    result = get_sliced_prediction(
        img_path, model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS",
    )

    preds = result.object_prediction_list
    if not preds:
        continue

    img_with_det += 1
    img = Image.open(img_path)
    img_w, img_h = img.size

    label_name = fname.replace(".jpg", ".txt")
    with open(os.path.join(OUT_DIR, label_name), "w") as f:
        for p in preds:
            bbox = p.bbox
            cx = (bbox.minx + bbox.maxx) / 2 / img_w
            cy = (bbox.miny + bbox.maxy) / 2 / img_h
            w = (bbox.maxx - bbox.minx) / img_w
            h = (bbox.maxy - bbox.miny) / img_h
            f.write(f"{p.category.id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            total_boxes += 1

    processed_files.append((cam, fname))

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"완료! ({elapsed/60:.1f}분)")
print(f"처리: {i+1}/{len(targets)}장")
print(f"탐지 이미지: {img_with_det}장 ({img_with_det/(i+1)*100:.1f}%)")
print(f"총 bbox: {total_boxes}개")
print(f"라벨 저장: {OUT_DIR}")

# 카메라별 통계
cam_counts = defaultdict(int)
for cam, fname in processed_files:
    cam_counts[cam] += 1
for cam in sorted(cam_counts):
    print(f"  {cam}: {cam_counts[cam]}장")
