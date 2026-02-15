#!/usr/bin/env python3
"""
go2k_manual (604) + captures_gated (3000) → CVAT 3개 task 패키징
YOLO 1.1 format (obj.data + obj.names + train.txt + labels)

실행: python prepare_cvat_all.py
"""
import os
import zipfile

GO2K_IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
GO2K_LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
CAP_LBL_DIR = "/home/lay/hoban/datasets/captures_labels_3k"
CAP_IMG_DIR = "/home/lay/video_indoor/static/captures"
OUT_BASE = "/home/lay/hoban/datasets/cvat_all"
N_TASKS = 3
CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]

os.makedirs(OUT_BASE, exist_ok=True)

# 1) go2k_manual 수집
pairs = []
for lf in sorted(os.listdir(GO2K_LBL_DIR)):
    if not lf.endswith(".txt"):
        continue
    img_name = lf.replace(".txt", ".jpg")
    img_path = os.path.join(GO2K_IMG_DIR, img_name)
    lbl_path = os.path.join(GO2K_LBL_DIR, lf)
    if os.path.exists(img_path):
        pairs.append((lf, img_name, img_path, lbl_path, "go2k"))

print(f"go2k_manual: {len(pairs)}장")

# 2) captures_gated 수집
cap_count = 0
for lf in sorted(os.listdir(CAP_LBL_DIR)):
    if not lf.endswith(".txt"):
        continue
    img_name = lf.replace(".txt", ".jpg")
    cam = img_name.split("_")[0]
    img_path = os.path.join(CAP_IMG_DIR, cam, img_name)
    lbl_path = os.path.join(CAP_LBL_DIR, lf)
    if os.path.exists(img_path):
        pairs.append((lf, img_name, img_path, lbl_path, "captures"))
        cap_count += 1

print(f"captures_gated: {cap_count}장")
print(f"합계: {len(pairs)}장 → {N_TASKS}개 task\n")

# 3개 task 분할
batch_size = (len(pairs) + N_TASKS - 1) // N_TASKS

for task_idx in range(N_TASKS):
    start = task_idx * batch_size
    end = min(start + batch_size, len(pairs))
    batch = pairs[start:end]
    part = task_idx + 1

    go2k_in_batch = sum(1 for _, _, _, _, src in batch if src == "go2k")
    cap_in_batch = sum(1 for _, _, _, _, src in batch if src == "captures")
    print(f"[Part {part}/{N_TASKS}] {len(batch)}장 (go2k: {go2k_in_batch}, captures: {cap_in_batch})")

    # annotations.zip
    ann_path = os.path.join(OUT_BASE, f"all_part{part}_annotations.zip")
    with zipfile.ZipFile(ann_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.data",
                     f"classes = {len(CLASS_NAMES)}\n"
                     f"train = data/train.txt\n"
                     f"names = data/obj.names\n"
                     f"backup = backup/\n")
        zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")

        train_lines = []
        for lf, img_name, img_path, lbl_path, src in batch:
            train_lines.append(f"data/obj_train_data/{img_name}")
            zf.write(lbl_path, f"obj_train_data/{lf}")
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    # images.zip
    img_zip_path = os.path.join(OUT_BASE, f"all_part{part}_images.zip")
    with zipfile.ZipFile(img_zip_path, 'w', zipfile.ZIP_STORED) as zf:
        for i, (_, img_name, img_path, _, _) in enumerate(batch):
            if i % 200 == 0:
                print(f"  이미지: {i}/{len(batch)}...")
            zf.write(img_path, img_name)

    ann_size = os.path.getsize(ann_path) / 1024 / 1024
    img_size = os.path.getsize(img_zip_path) / 1024 / 1024
    print(f"  → annotations: {ann_size:.1f}MB, images: {img_size:.1f}MB")

total_size = sum(
    os.path.getsize(os.path.join(OUT_BASE, f))
    for f in os.listdir(OUT_BASE)
) / 1024 / 1024 / 1024

print(f"\n{'=' * 60}")
print(f"완료! {len(pairs)}장 → {N_TASKS}개 task")
print(f"총 용량: {total_size:.2f}GB")
print(f"출력: {OUT_BASE}")
