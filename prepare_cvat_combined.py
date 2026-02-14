#!/usr/bin/env python3
"""
1차(30초 필터) + 2차 라벨 합쳐서 CVAT 3개 task로 패키징

실행: python prepare_cvat_combined.py
"""
import os
import zipfile
from collections import defaultdict

SNAP_DIR = "/home/lay/video_indoor/static/snapshots_raw"
LABEL1_DIR = "/home/lay/hoban/datasets/snapshots_labels"        # 1차 (3072장, 30초 필터 필요)
LABEL2_DIR = "/home/lay/hoban/datasets/snapshots_labels_batch2"  # 2차 (1504장, 이미 필터됨)
OUT_BASE = "/home/lay/hoban/datasets/cvat_snapshots"
N_TASKS = 3
CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
MIN_INTERVAL = 20  # 1차도 20초로 통일


def parse_ts(fname):
    parts = fname.split("_")
    if len(parts) < 4:
        return None
    try:
        date = int(parts[1])
        t = parts[2]
        h, m, s = int(t[:2]), int(t[2:4]), int(t[4:6])
        ms = int(parts[3])
        return date * 86400 + h * 3600 + m * 60 + s + ms / 1000
    except (ValueError, IndexError):
        return None


# 2차 라벨 파일명 수집
batch2_files = set(os.listdir(LABEL2_DIR))

# 1차: 30초→20초 필터 재적용
batch1_all = sorted(os.listdir(LABEL1_DIR))
cam_files1 = defaultdict(list)
for lf in batch1_all:
    cam = lf.split("_")[0]
    cam_files1[cam].append(lf)

batch1_filtered = []
for cam in sorted(cam_files1):
    last_ts = None
    for lf in cam_files1[cam]:
        ts = parse_ts(lf)
        if ts is None:
            batch1_filtered.append(lf)
            continue
        if last_ts and (ts - last_ts) < MIN_INTERVAL:
            continue
        batch1_filtered.append(lf)
        last_ts = ts

print(f"1차: {len(batch1_all)} → {len(batch1_filtered)}장 (20초 필터)")
print(f"2차: {len(batch2_files)}장")

# 합치기 (label_file, label_dir, img_name)
combined = []
for lf in batch1_filtered:
    img_name = lf.replace(".txt", ".jpg")
    if os.path.exists(os.path.join(SNAP_DIR, img_name)):
        combined.append((lf, LABEL1_DIR, img_name))

for lf in sorted(batch2_files):
    img_name = lf.replace(".txt", ".jpg")
    if os.path.exists(os.path.join(SNAP_DIR, img_name)):
        combined.append((lf, LABEL2_DIR, img_name))

# 정렬 (파일명 기준 = 카메라 + 시간순)
combined.sort(key=lambda x: x[2])

# 중복 제거
seen = set()
deduped = []
for lf, ldir, img in combined:
    if img not in seen:
        seen.add(img)
        deduped.append((lf, ldir, img))
combined = deduped

print(f"합계 (중복 제거): {len(combined)}장\n")

# 3개 task로 분할
batch_size = (len(combined) + N_TASKS - 1) // N_TASKS
os.makedirs(OUT_BASE, exist_ok=True)

for task_idx in range(N_TASKS):
    start = task_idx * batch_size
    end = min(start + batch_size, len(combined))
    batch = combined[start:end]
    part = task_idx + 1

    # 통계
    has_det = sum(1 for _, ldir, _ in batch
                  if os.path.getsize(os.path.join(ldir, _.replace(".jpg", ".txt"))) > 0)

    print(f"[Part {part}/{N_TASKS}] {len(batch)}장 (탐지: {has_det}, 빈: {len(batch)-has_det})")

    # annotations.zip
    ann_path = os.path.join(OUT_BASE, f"snapshots_part{part}_annotations.zip")
    with zipfile.ZipFile(ann_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.data", f"classes = {len(CLASS_NAMES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
        zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")

        train_lines = []
        for lf, ldir, img_name in batch:
            train_lines.append(f"data/obj_train_data/{img_name}")
            lbl_path = os.path.join(ldir, lf)
            # 빈 라벨도 포함 (CVAT에서 빈 이미지로 표시)
            if os.path.getsize(lbl_path) > 0:
                zf.write(lbl_path, f"obj_train_data/{lf}")
            else:
                zf.writestr(f"obj_train_data/{lf}", "")

        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    # images.zip
    img_path = os.path.join(OUT_BASE, f"snapshots_part{part}_images.zip")
    with zipfile.ZipFile(img_path, 'w', zipfile.ZIP_STORED) as zf:
        for i, (_, _, img_name) in enumerate(batch):
            if i % 200 == 0:
                print(f"  이미지: {i}/{len(batch)}...")
            zf.write(os.path.join(SNAP_DIR, img_name), img_name)

    ann_size = os.path.getsize(ann_path) / 1024 / 1024
    img_size = os.path.getsize(img_path) / 1024 / 1024
    print(f"  → annotations: {ann_size:.1f}MB, images: {img_size:.1f}MB\n")

total_size = sum(
    os.path.getsize(os.path.join(OUT_BASE, f))
    for f in os.listdir(OUT_BASE)
) / 1024 / 1024 / 1024

print(f"{'=' * 60}")
print(f"완료! {len(combined)}장 → {N_TASKS}개 task")
print(f"총 용량: {total_size:.2f}GB")
print(f"출력: {OUT_BASE}")
