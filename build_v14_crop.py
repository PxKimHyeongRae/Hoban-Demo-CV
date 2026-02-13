#!/usr/bin/env python3
"""
v14 Crop 데이터셋 빌더

- datasets_v13 전체(train+valid)에서 bbox별 640x640 crop
- 클래스 밸런스: person_with_helmet 30k + person_without_helmet 30k = 60k
- negative 없음
- 새로운 80/20 split

실행 (서버):
  python build_v14_crop.py

출력: /home/lay/hoban/datasets_v14/
"""

import os
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ===== Config =====
SEED = 42
random.seed(SEED)

SRC_DIR = "/home/lay/hoban/datasets_v13"
OUT_DIR = "/home/lay/hoban/datasets_v14"
CROP_SIZE = 640
TARGET_PER_CLASS = 30000
VALID_RATIO = 0.2
MIN_BBOX_IN_CROP = 0.5  # crop 안에 bbox가 50% 이상 포함되어야 유효


def parse_labels(label_path):
    """YOLO 라벨 파싱 → [(cls, cx, cy, w, h), ...]"""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append((cls, cx, cy, w, h))
    return bboxes


def compute_crop_region(target_bbox, img_w, img_h, crop_size, jitter_scale=0.15):
    """target bbox 중심으로 crop 영역 계산 (jitter 포함)"""
    _, cx, cy, _, _ = target_bbox
    px_cx = cx * img_w
    px_cy = cy * img_h

    jitter = crop_size * jitter_scale
    px_cx += random.uniform(-jitter, jitter)
    px_cy += random.uniform(-jitter, jitter)

    x1 = int(px_cx - crop_size // 2)
    y1 = int(px_cy - crop_size // 2)

    # 경계 clamp
    x1 = max(0, min(x1, img_w - crop_size))
    y1 = max(0, min(y1, img_h - crop_size))

    return x1, y1, x1 + crop_size, y1 + crop_size


def transform_bboxes(bboxes, crop_x1, crop_y1, crop_size, img_w, img_h):
    """원본 bbox → crop 기준 bbox로 변환"""
    new_bboxes = []
    for cls, cx, cy, w, h in bboxes:
        px_cx = cx * img_w
        px_cy = cy * img_h
        px_w = w * img_w
        px_h = h * img_h

        bx1 = px_cx - px_w / 2
        by1 = px_cy - px_h / 2
        bx2 = px_cx + px_w / 2
        by2 = px_cy + px_h / 2

        ix1 = max(bx1, crop_x1)
        iy1 = max(by1, crop_y1)
        ix2 = min(bx2, crop_x1 + crop_size)
        iy2 = min(by2, crop_y1 + crop_size)

        if ix1 >= ix2 or iy1 >= iy2:
            continue

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        orig_area = px_w * px_h
        if orig_area > 0 and inter_area / orig_area < MIN_BBOX_IN_CROP:
            continue

        new_cx = ((ix1 + ix2) / 2 - crop_x1) / crop_size
        new_cy = ((iy1 + iy2) / 2 - crop_y1) / crop_size
        new_w = (ix2 - ix1) / crop_size
        new_h = (iy2 - iy1) / crop_size

        new_cx = max(0, min(1, new_cx))
        new_cy = max(0, min(1, new_cy))
        new_w = min(new_w, 1.0)
        new_h = min(new_h, 1.0)

        if new_w > 0.01 and new_h > 0.01:
            new_bboxes.append((cls, new_cx, new_cy, new_w, new_h))

    return new_bboxes


def collect_all_bboxes():
    """v13 train+valid에서 모든 (이미지경로, bbox) 쌍 수집"""
    # class별 bbox 목록: {cls: [(img_path, label_path, bbox_index, all_bboxes), ...]}
    class_bboxes = defaultdict(list)

    for split in ["train", "valid"]:
        img_dir = os.path.join(SRC_DIR, split, "images")
        lbl_dir = os.path.join(SRC_DIR, split, "labels")

        if not os.path.exists(lbl_dir):
            continue

        label_files = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
        for lf in label_files:
            bboxes = parse_labels(os.path.join(lbl_dir, lf))
            if not bboxes:
                continue

            stem = lf.replace('.txt', '')
            img_path = os.path.join(img_dir, stem + '.jpg')
            if not os.path.exists(img_path):
                continue

            for i, bbox in enumerate(bboxes):
                cls = bbox[0]
                class_bboxes[cls].append((img_path, bboxes, i))

    return class_bboxes


def main():
    print("=" * 60)
    print("v14 Crop Dataset Builder")
    print(f"  Source: {SRC_DIR} (train + valid)")
    print(f"  Output: {OUT_DIR}")
    print(f"  Crop: {CROP_SIZE}x{CROP_SIZE}")
    print(f"  Target: {TARGET_PER_CLASS} per class (total {TARGET_PER_CLASS * 2})")
    print(f"  No negatives")
    print("=" * 60)

    # 기존 출력 삭제
    import shutil
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    for d in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

    # 1. 모든 bbox 수집
    print("\n[1/3] Collecting bboxes from v13...")
    class_bboxes = collect_all_bboxes()
    for cls in sorted(class_bboxes.keys()):
        print(f"  Class {cls}: {len(class_bboxes[cls])} bboxes")

    # 2. 클래스별 30k 샘플링
    print(f"\n[2/3] Sampling {TARGET_PER_CLASS} per class...")
    all_crops = []  # [(img_path, all_bboxes, target_bbox_idx, crop_id)]

    for cls in [0, 1]:
        items = class_bboxes[cls]
        random.shuffle(items)

        if len(items) >= TARGET_PER_CLASS:
            selected = items[:TARGET_PER_CLASS]
            print(f"  Class {cls}: {len(items)} available → sampled {TARGET_PER_CLASS}")
        else:
            # 부족하면 다른 jitter로 반복 추출
            selected = list(items)
            need = TARGET_PER_CLASS - len(selected)
            extra = random.choices(items, k=need)
            selected.extend(extra)
            print(f"  Class {cls}: {len(items)} available → {len(items)} + {need} augmented = {TARGET_PER_CLASS}")

        for i, (img_path, bboxes, bbox_idx) in enumerate(selected):
            crop_id = f"c{cls}_{i:06d}"
            all_crops.append((img_path, bboxes, bbox_idx, crop_id))

    random.shuffle(all_crops)
    total = len(all_crops)
    n_val = int(total * VALID_RATIO)
    val_crops = all_crops[:n_val]
    train_crops = all_crops[n_val:]

    print(f"\n  Total: {total} crops → train {len(train_crops)}, valid {len(val_crops)}")

    # 3. Crop 실행
    print(f"\n[3/3] Cropping images...")
    stats = {"train": defaultdict(int), "valid": defaultdict(int)}
    area_before = []
    area_after = []

    # 이미지 캐시 (동일 이미지 반복 열기 방지)
    img_cache = {}
    CACHE_MAX = 200

    for split_name, split_data in [("train", train_crops), ("valid", val_crops)]:
        out_img = os.path.join(OUT_DIR, split_name, "images")
        out_lbl = os.path.join(OUT_DIR, split_name, "labels")

        for idx, (img_path, bboxes, target_idx, crop_id) in enumerate(split_data):
            # 이미지 로드 (캐시)
            if img_path not in img_cache:
                if len(img_cache) >= CACHE_MAX:
                    img_cache.clear()
                try:
                    im = Image.open(img_path)
                    im.load()
                    img_cache[img_path] = im
                except Exception:
                    img_cache[img_path] = None
            img = img_cache[img_path]
            if img is None:
                continue
            img_w, img_h = img.size

            if img_w < CROP_SIZE or img_h < CROP_SIZE:
                continue

            target_bbox = bboxes[target_idx]

            # area before
            for _, _, _, w, h in bboxes:
                area_before.append(w * h * 100)

            # crop 영역 (augmented items는 jitter가 다름 → 자동으로 다른 crop)
            cx1, cy1, cx2, cy2 = compute_crop_region(target_bbox, img_w, img_h, CROP_SIZE)

            # crop
            cropped = img.crop((cx1, cy1, cx2, cy2))

            # bbox 변환
            new_bboxes = transform_bboxes(bboxes, cx1, cy1, CROP_SIZE, img_w, img_h)

            if not new_bboxes:
                continue

            # area after
            for _, _, _, w, h in new_bboxes:
                area_after.append(w * h * 100)

            # 저장
            out_name = f"{crop_id}.jpg"
            cropped.save(os.path.join(out_img, out_name), quality=95)

            lbl_name = f"{crop_id}.txt"
            with open(os.path.join(out_lbl, lbl_name), 'w') as f:
                for cls, cx, cy, w, h in new_bboxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    stats[split_name][cls] += 1

            if (idx + 1) % 5000 == 0:
                print(f"  [{split_name}] {idx+1}/{len(split_data)} processed...")

        print(f"  [{split_name}] done: {len(os.listdir(out_img))} images")

    # data.yaml (Windows)
    data_yaml = (
        f"path: Z:\\home\\lay\\hoban\\datasets_v14\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: person_with_helmet\n"
        f"  1: person_without_helmet\n"
    )
    with open(os.path.join(OUT_DIR, "data.yaml"), 'w') as f:
        f.write(data_yaml)

    # 통계
    print(f"\n{'='*60}")
    print("Dataset Summary")
    for split_name in ["train", "valid"]:
        s = stats[split_name]
        imgs = len(os.listdir(os.path.join(OUT_DIR, split_name, "images")))
        print(f"  [{split_name}] {imgs} images | helmet_o={s[0]}, helmet_x={s[1]}")

    if area_before and area_after:
        area_before.sort()
        area_after.sort()
        nb, na = len(area_before), len(area_after)
        print(f"\n  Bbox Area (P25): {area_before[nb//4]:.3f}% → {area_after[na//4]:.3f}%")
        print(f"  Bbox Area (P50): {area_before[nb//2]:.3f}% → {area_after[na//2]:.3f}%")
        print(f"  Bbox Area (P75): {area_before[3*nb//4]:.3f}% → {area_after[3*na//4]:.3f}%")

    print(f"\n  Output: {OUT_DIR}")
    print(f"  data.yaml: {os.path.join(OUT_DIR, 'data.yaml')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
