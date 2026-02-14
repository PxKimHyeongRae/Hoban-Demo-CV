#!/usr/bin/env python3
"""
snapshots_labels → CVAT 업로드용 zip 패키지 생성
- YOLO 1.1 format (images.zip + annotations.zip per batch)
- 30초 간격 필터링 (연속 프레임 중복 제거)
- 3개 task로 분할

실행: python prepare_cvat_snapshots.py
"""
import os
import zipfile
from collections import defaultdict

LABEL_DIR = "/home/lay/hoban/datasets/snapshots_labels"
IMG_DIR = "/home/lay/video_indoor/static/snapshots_raw"
OUT_BASE = "/home/lay/hoban/datasets/cvat_snapshots"
MIN_INTERVAL = 30  # 최소 간격 (초)
N_TASKS = 3
CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]


def parse_timestamp(fname):
    parts = fname.split("_")
    if len(parts) < 4:
        return None
    try:
        date = int(parts[1])
        time_str = parts[2]
        h, m, s = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
        ms = int(parts[3]) if len(parts) >= 4 else 0
        return date * 86400 + h * 3600 + m * 60 + s + ms / 1000
    except (ValueError, IndexError):
        return None


# 라벨 파일 목록 (이미지 존재 확인)
label_files = sorted(os.listdir(LABEL_DIR))
pairs = []
for lf in label_files:
    img_name = lf.replace(".txt", ".jpg")
    if os.path.exists(os.path.join(IMG_DIR, img_name)):
        pairs.append((img_name, lf))

print(f"전체 라벨: {len(pairs)}개")

# 카메라별 30초 간격 필터링
cam_pairs = defaultdict(list)
for img_name, lf in pairs:
    cam = img_name.split("_")[0]
    cam_pairs[cam].append((img_name, lf))

filtered = []
for cam in sorted(cam_pairs):
    cam_list = cam_pairs[cam]
    sampled = [cam_list[0]]
    last_ts = parse_timestamp(cam_list[0][0])

    for img_name, lf in cam_list[1:]:
        ts = parse_timestamp(img_name)
        if ts and last_ts and (ts - last_ts) >= MIN_INTERVAL:
            sampled.append((img_name, lf))
            last_ts = ts
        elif ts is None:
            sampled.append((img_name, lf))

    print(f"  {cam}: {len(cam_list)} → {len(sampled)}장 (30초 간격)")
    filtered.extend(sampled)

filtered.sort(key=lambda x: x[0])
print(f"\n필터 후: {len(filtered)}장")

# N개 task로 분할
batch_size = (len(filtered) + N_TASKS - 1) // N_TASKS
print(f"배치: {N_TASKS}개 (각 ~{batch_size}장)\n")

os.makedirs(OUT_BASE, exist_ok=True)

for batch_idx in range(N_TASKS):
    start = batch_idx * batch_size
    end = min(start + batch_size, len(filtered))
    batch = filtered[start:end]
    part = batch_idx + 1

    print(f"[Part {part}/{N_TASKS}] {len(batch)}장")

    # annotations.zip
    ann_zip_path = os.path.join(OUT_BASE, f"snapshots_part{part}_annotations.zip")
    with zipfile.ZipFile(ann_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.data", f"classes = {len(CLASS_NAMES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
        zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")

        train_lines = []
        for img_name, lbl_name in batch:
            train_lines.append(f"data/obj_train_data/{img_name}")
            zf.write(os.path.join(LABEL_DIR, lbl_name), f"obj_train_data/{lbl_name}")

        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    # images.zip
    img_zip_path = os.path.join(OUT_BASE, f"snapshots_part{part}_images.zip")
    with zipfile.ZipFile(img_zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for i, (img_name, _) in enumerate(batch):
            if i % 100 == 0:
                print(f"  이미지 압축: {i}/{len(batch)}...")
            zf.write(os.path.join(IMG_DIR, img_name), img_name)

    ann_size = os.path.getsize(ann_zip_path) / 1024 / 1024
    img_size = os.path.getsize(img_zip_path) / 1024 / 1024
    print(f"  → annotations: {ann_size:.1f}MB, images: {img_size:.1f}MB\n")

# 요약
print(f"{'=' * 60}")
print(f"완료! 출력: {OUT_BASE}")
print(f"총 {len(filtered)}장 → {N_TASKS}개 task")
print(f"\nCVAT 업로드 방법:")
print(f"  1. Create Task → snapshots_part1 (등)")
print(f"  2. 이미지: snapshots_partN_images.zip 업로드")
print(f"  3. Actions → Upload annotations → YOLO 1.1")
print(f"  4. snapshots_partN_annotations.zip 업로드")

total_size = sum(
    os.path.getsize(os.path.join(OUT_BASE, f))
    for f in os.listdir(OUT_BASE)
) / 1024 / 1024 / 1024
print(f"\n총 용량: {total_size:.2f}GB")
