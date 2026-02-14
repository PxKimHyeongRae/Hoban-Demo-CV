#!/usr/bin/env python3
"""
snapshots_labels → CVAT 업로드용 zip 패키지 생성
- YOLO 1.1 format (images.zip + annotations.zip per batch)
- 500장씩 배치 분할

실행: python prepare_cvat_snapshots.py
"""
import os
import shutil
import zipfile

LABEL_DIR = "/home/lay/hoban/datasets/snapshots_labels"
IMG_DIR = "/home/lay/video_indoor/static/snapshots_raw"
OUT_BASE = "/home/lay/hoban/datasets/cvat_snapshots"
BATCH_SIZE = 500
CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]

# 라벨 파일 목록 (이미지 존재 확인)
label_files = sorted(os.listdir(LABEL_DIR))
pairs = []
missing = 0
for lf in label_files:
    img_name = lf.replace(".txt", ".jpg")
    img_path = os.path.join(IMG_DIR, img_name)
    if os.path.exists(img_path):
        pairs.append((img_name, lf))
    else:
        missing += 1

print(f"라벨: {len(label_files)}개, 이미지 매칭: {len(pairs)}개, 누락: {missing}개")

# 배치 분할
n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"배치: {n_batches}개 (각 {BATCH_SIZE}장)\n")

os.makedirs(OUT_BASE, exist_ok=True)

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(pairs))
    batch = pairs[start:end]
    part = batch_idx + 1

    print(f"[Part {part}/{n_batches}] {len(batch)}장 ({batch[0][0][:20]}... ~ {batch[-1][0][:20]}...)")

    # annotations.zip (labels + obj.names + train.txt)
    ann_zip_path = os.path.join(OUT_BASE, f"snapshots_part{part}_annotations.zip")
    with zipfile.ZipFile(ann_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # obj.names
        obj_names = "\n".join(CLASS_NAMES) + "\n"
        zf.writestr("obj.names", obj_names)

        # train.txt + label files
        train_lines = []
        for img_name, lbl_name in batch:
            train_lines.append(f"data/obj_train_data/{img_name}")
            lbl_path = os.path.join(LABEL_DIR, lbl_name)
            zf.write(lbl_path, f"obj_train_data/{lbl_name}")

        train_txt = "\n".join(train_lines) + "\n"
        zf.writestr("train.txt", train_txt)

    # images.zip
    img_zip_path = os.path.join(OUT_BASE, f"snapshots_part{part}_images.zip")
    with zipfile.ZipFile(img_zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for i, (img_name, _) in enumerate(batch):
            if i % 100 == 0:
                print(f"  이미지 압축: {i}/{len(batch)}...")
            img_path = os.path.join(IMG_DIR, img_name)
            zf.write(img_path, img_name)

    ann_size = os.path.getsize(ann_zip_path) / 1024 / 1024
    img_size = os.path.getsize(img_zip_path) / 1024 / 1024
    print(f"  → annotations: {ann_size:.1f}MB, images: {img_size:.1f}MB\n")

# 요약
print(f"{'=' * 60}")
print(f"완료! 출력: {OUT_BASE}")
print(f"총 {len(pairs)}장 → {n_batches}개 배치")
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
