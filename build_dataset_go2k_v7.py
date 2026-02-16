#!/usr/bin/env python3
"""
go2k_v7 데이터셋 빌드: 근본 문제 해결
1. Train/Eval 완전 분리 (data leakage 제거)
2. x1~x7 복제 대신 실제 augmentation 적용
3. v13 데이터 bbox 크기 필터링 (eval 도메인 범위만)
"""
import os
import sys
import shutil
import random
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2

random.seed(42)
np.random.seed(42)

# === Paths ===
GO2K_MANUAL_IMG = "/home/lay/hoban/datasets/go2k_manual/images"
GO2K_MANUAL_LBL = "/home/lay/hoban/datasets/go2k_manual/labels"
V13_TRAIN_IMG = "/home/lay/hoban/datasets_go2k_v2/train/images"
V13_TRAIN_LBL = "/home/lay/hoban/datasets_go2k_v2/train/labels"
OUTPUT = "/home/lay/hoban/datasets_go2k_v7"


def get_bbox_areas(label_path):
    """Get normalized bbox areas from label file."""
    areas = []
    if not os.path.exists(label_path):
        return areas
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                w, h = float(parts[3]), float(parts[4])
                areas.append(w * h)
    return areas


def augment_image(img_path, out_path, aug_type):
    """Apply real augmentation to image."""
    img = Image.open(img_path)

    if aug_type == 0:
        # Brightness variation
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)
    elif aug_type == 1:
        # Contrast variation
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)
    elif aug_type == 2:
        # Saturation variation
        factor = random.uniform(0.6, 1.4)
        img = ImageEnhance.Color(img).enhance(factor)
    elif aug_type == 3:
        # Gaussian blur
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    elif aug_type == 4:
        # Combined: brightness + contrast
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    elif aug_type == 5:
        # Sharpness
        factor = random.uniform(0.5, 2.0)
        img = ImageEnhance.Sharpness(img).enhance(factor)
    elif aug_type == 6:
        # Gaussian noise via numpy
        arr = np.array(img)
        noise = np.random.normal(0, 8, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    img.save(out_path, quality=95)


def split_go2k_manual():
    """Split go2k_manual into train/eval with NO overlap."""
    all_files = sorted([f for f in os.listdir(GO2K_MANUAL_IMG) if f.endswith(".jpg")])

    # Group by camera + date for temporal split
    # Format: cam{N}_{date}_{time}_{id}_event_{hash}.jpg
    by_cam_date = {}
    for f in all_files:
        parts = f.split("_")
        cam = parts[0]  # cam1, cam2, etc.
        date = parts[1]  # 20260211, 20260212
        key = f"{cam}_{date}"
        if key not in by_cam_date:
            by_cam_date[key] = []
        by_cam_date[key].append(f)

    print("카메라-날짜별 분포:")
    for key in sorted(by_cam_date.keys()):
        print(f"  {key}: {len(by_cam_date[key])}장")

    # Strategy: Use all cam2_20260211 data for train (majority),
    # keep cam1 + cam2_20260212 for eval (different time/camera)
    train_files = []
    eval_files = []

    for key, files in by_cam_date.items():
        if key == "cam2_20260211":
            # Main training source - use 80% for train
            random.shuffle(files)
            split = int(len(files) * 0.8)
            train_files.extend(files[:split])
            eval_files.extend(files[split:])
        elif key == "cam1_20260211":
            # Small set - split 50/50 for diversity
            random.shuffle(files)
            split = int(len(files) * 0.5)
            train_files.extend(files[:split])
            eval_files.extend(files[split:])
        else:
            # cam1_20260212, cam2_20260212 etc. - all to eval (temporal separation)
            eval_files.extend(files)

    return sorted(train_files), sorted(eval_files)


def filter_v13_by_bbox_size(min_area=0.00005, max_area=0.005):
    """Filter v13 (S2-*) images by bbox area range matching eval domain."""
    v13_files = []
    v13_rejected = 0

    for f in os.listdir(V13_TRAIN_IMG):
        if not f.startswith("S2-") or not f.endswith(".jpg"):
            continue
        label_path = os.path.join(V13_TRAIN_LBL, f.replace(".jpg", ".txt"))
        areas = get_bbox_areas(label_path)

        if not areas:
            # Empty label (negative) - keep some
            v13_files.append(f)
            continue

        median_area = np.median(areas)
        if min_area <= median_area <= max_area:
            v13_files.append(f)
        else:
            v13_rejected += 1

    return sorted(v13_files), v13_rejected


def main():
    print("=" * 60)
    print("go2k_v7 데이터셋 빌드")
    print("=" * 60)

    # === Step 1: Split go2k_manual (no leakage) ===
    print("\n[1/4] go2k_manual Train/Eval 분리 (leakage 제거)")
    go2k_train, go2k_eval = split_go2k_manual()

    # Count GT per split
    for name, files in [("Train", go2k_train), ("Eval", go2k_eval)]:
        gt_count = 0
        cls0 = cls1 = 0
        for f in files:
            lbl = os.path.join(GO2K_MANUAL_LBL, f.replace(".jpg", ".txt"))
            if os.path.exists(lbl):
                with open(lbl) as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            gt_count += 1
                            if int(parts[0]) == 0:
                                cls0 += 1
                            else:
                                cls1 += 1
        print(f"  go2k {name}: {len(files)}장, {gt_count} bbox (cls0={cls0}, cls1={cls1})")

    # === Step 2: Filter v13 by bbox size ===
    print("\n[2/4] v13 bbox 크기 필터링 (eval 도메인 범위)")
    v13_selected, v13_rejected = filter_v13_by_bbox_size(
        min_area=0.00005, max_area=0.005
    )
    print(f"  v13 선택: {len(v13_selected)}장 (제외: {v13_rejected}장)")

    # === Step 3: Create augmented copies ===
    print(f"\n[3/4] 데이터셋 구성")

    # Create output dirs
    for split in ["train", "valid"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(OUTPUT, split, sub), exist_ok=True)

    # -- Train: go2k_train (original + 3 augmented copies) + filtered v13 --
    n_aug = 3  # 원본 + 3 증강 = 4x (8x에서 감소, 하지만 실제 augmentation)
    train_count = 0

    # go2k train originals + augmented
    print(f"  go2k train: {len(go2k_train)}장 x {n_aug+1} = {len(go2k_train)*(n_aug+1)}장")
    for f in go2k_train:
        src_img = os.path.join(GO2K_MANUAL_IMG, f)
        src_lbl = os.path.join(GO2K_MANUAL_LBL, f.replace(".jpg", ".txt"))
        dst_img = os.path.join(OUTPUT, "train/images", f)
        dst_lbl = os.path.join(OUTPUT, "train/labels", f.replace(".jpg", ".txt"))

        # Original (symlink)
        os.symlink(os.path.abspath(src_img), dst_img)
        if os.path.exists(src_lbl):
            os.symlink(os.path.abspath(src_lbl), dst_lbl)
        else:
            open(dst_lbl, 'w').close()
        train_count += 1

        # Augmented copies
        for aug_i in range(n_aug):
            aug_name = f.replace(".jpg", f"_aug{aug_i}.jpg")
            aug_lbl_name = f.replace(".jpg", f"_aug{aug_i}.txt")
            aug_img_path = os.path.join(OUTPUT, "train/images", aug_name)
            aug_lbl_path = os.path.join(OUTPUT, "train/labels", aug_lbl_name)

            augment_image(src_img, aug_img_path, aug_type=aug_i + random.randint(0, 3))
            # Label is same (bbox unchanged for color augmentations)
            if os.path.exists(src_lbl):
                os.symlink(os.path.abspath(src_lbl), aug_lbl_path)
            else:
                open(aug_lbl_path, 'w').close()
            train_count += 1

    # v13 filtered (symlink)
    print(f"  v13 filtered: {len(v13_selected)}장")
    for f in v13_selected:
        src_img = os.path.join(V13_TRAIN_IMG, f)
        src_lbl = os.path.join(V13_TRAIN_LBL, f.replace(".jpg", ".txt"))
        dst_img = os.path.join(OUTPUT, "train/images", f)
        dst_lbl = os.path.join(OUTPUT, "train/labels", f.replace(".jpg", ".txt"))

        if os.path.exists(dst_img):
            continue
        os.symlink(os.path.abspath(src_img), dst_img)
        if os.path.exists(src_lbl):
            os.symlink(os.path.abspath(src_lbl), dst_lbl)
        else:
            open(dst_lbl, 'w').close()
        train_count += 1

    # -- Valid: go2k_eval (NO overlap with train) --
    valid_count = 0
    for f in go2k_eval:
        src_img = os.path.join(GO2K_MANUAL_IMG, f)
        src_lbl = os.path.join(GO2K_MANUAL_LBL, f.replace(".jpg", ".txt"))
        dst_img = os.path.join(OUTPUT, "valid/images", f)
        dst_lbl = os.path.join(OUTPUT, "valid/labels", f.replace(".jpg", ".txt"))

        os.symlink(os.path.abspath(src_img), dst_img)
        if os.path.exists(src_lbl):
            os.symlink(os.path.abspath(src_lbl), dst_lbl)
        else:
            open(dst_lbl, 'w').close()
        valid_count += 1

    print(f"\n  총 Train: {train_count}장")
    print(f"  총 Valid: {valid_count}장")

    # === Step 4: Write data.yaml ===
    yaml_path = os.path.join(OUTPUT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(OUTPUT)}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: 2\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")

    print(f"\n[4/4] data.yaml 생성: {yaml_path}")
    print("\n" + "=" * 60)
    print("완료!")
    print(f"  Train: {train_count}장 (go2k {len(go2k_train)}x{n_aug+1} + v13 {len(v13_selected)})")
    print(f"  Valid: {valid_count}장 (go2k eval, train과 겹침 0%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
