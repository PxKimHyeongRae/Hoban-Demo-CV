#!/usr/bin/env python3
"""
v15 Tiled Training 데이터셋 빌더

- datasets_v13 이미지(1920x1080)를 640x640 타일로 균일 분할
- overlap 20%로 경계 객체 보존
- 빈 타일(negative)도 포함 → SAHI 추론과 동일한 분포
- crop과의 차이: 그리드 기반 균일 분할 (bbox 중심 아님)

실행 (서버):
  python build_v15_tiled.py
  python build_v15_tiled.py --negative-ratio 0.3

출력: /home/lay/hoban/datasets_v15/
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ===== Config =====
SEED = 42
random.seed(SEED)

SRC_DIR = "/home/lay/hoban/datasets_v13"
OUT_DIR = "/home/lay/hoban/datasets_v15"
TILE_SIZE = 640
OVERLAP = 0.2  # 20% overlap
MIN_BBOX_IN_TILE = 0.4  # 타일 안에 bbox가 40% 이상 포함되어야 유효
MIN_BBOX_SIZE = 0.01  # 타일 기준 1% 미만 bbox 제거


def parse_labels(label_path):
    """YOLO 라벨 파싱"""
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


def compute_tile_grid(img_w, img_h, tile_size, overlap):
    """이미지에 대한 타일 그리드 좌표 계산"""
    stride = int(tile_size * (1 - overlap))
    tiles = []

    # x, y 시작점 계산
    x_starts = []
    x = 0
    while x + tile_size <= img_w:
        x_starts.append(x)
        x += stride
    # 마지막 타일이 이미지 끝에 닿도록
    if not x_starts or x_starts[-1] + tile_size < img_w:
        x_starts.append(max(0, img_w - tile_size))

    y_starts = []
    y = 0
    while y + tile_size <= img_h:
        y_starts.append(y)
        y += stride
    if not y_starts or y_starts[-1] + tile_size < img_h:
        y_starts.append(max(0, img_h - tile_size))

    # 중복 제거
    x_starts = sorted(set(x_starts))
    y_starts = sorted(set(y_starts))

    for y0 in y_starts:
        for x0 in x_starts:
            tiles.append((x0, y0, x0 + tile_size, y0 + tile_size))

    return tiles


def transform_bboxes_to_tile(bboxes, tx1, ty1, tile_size, img_w, img_h):
    """원본 bbox → 타일 기준 bbox 변환"""
    new_bboxes = []
    for cls, cx, cy, w, h in bboxes:
        # pixel 좌표
        px_cx = cx * img_w
        px_cy = cy * img_h
        px_w = w * img_w
        px_h = h * img_h

        bx1 = px_cx - px_w / 2
        by1 = px_cy - px_h / 2
        bx2 = px_cx + px_w / 2
        by2 = px_cy + px_h / 2

        # 타일과 교집합
        ix1 = max(bx1, tx1)
        iy1 = max(by1, ty1)
        ix2 = min(bx2, tx1 + tile_size)
        iy2 = min(by2, ty1 + tile_size)

        if ix1 >= ix2 or iy1 >= iy2:
            continue

        # 교집합 비율
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        orig_area = px_w * px_h
        if orig_area > 0 and inter_area / orig_area < MIN_BBOX_IN_TILE:
            continue

        # 타일 기준 normalized
        new_cx = ((ix1 + ix2) / 2 - tx1) / tile_size
        new_cy = ((iy1 + iy2) / 2 - ty1) / tile_size
        new_w = (ix2 - ix1) / tile_size
        new_h = (iy2 - iy1) / tile_size

        new_cx = max(0, min(1, new_cx))
        new_cy = max(0, min(1, new_cy))
        new_w = min(new_w, 1.0)
        new_h = min(new_h, 1.0)

        if new_w > MIN_BBOX_SIZE and new_h > MIN_BBOX_SIZE:
            new_bboxes.append((cls, new_cx, new_cy, new_w, new_h))

    return new_bboxes


def process_split(split, negative_ratio=0.3):
    """한 split 처리"""
    img_dir = os.path.join(SRC_DIR, split, "images")
    lbl_dir = os.path.join(SRC_DIR, split, "labels")
    out_img = os.path.join(OUT_DIR, split, "images")
    out_lbl = os.path.join(OUT_DIR, split, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    label_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    print(f"\n[{split}] Processing {len(label_files)} images...")

    positive_tiles = 0
    negative_tiles = 0
    skipped_negatives = 0
    area_stats = []
    cls_stats = defaultdict(int)

    for idx, lf in enumerate(label_files):
        stem = lf.replace('.txt', '')
        img_path = os.path.join(img_dir, stem + '.jpg')

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            img.load()
        except Exception:
            continue

        img_w, img_h = img.size
        bboxes = parse_labels(os.path.join(lbl_dir, lf))

        # 타일 그리드 계산
        tiles = compute_tile_grid(img_w, img_h, TILE_SIZE, OVERLAP)

        for ti, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile_bboxes = transform_bboxes_to_tile(bboxes, tx1, ty1, TILE_SIZE, img_w, img_h)

            is_positive = len(tile_bboxes) > 0

            # negative 타일은 비율에 따라 샘플링
            if not is_positive:
                if random.random() > negative_ratio:
                    skipped_negatives += 1
                    continue
                negative_tiles += 1
            else:
                positive_tiles += 1

            # 타일 crop & 저장
            tile_name = f"{stem}_t{ti:02d}.jpg"
            tile_img = img.crop((tx1, ty1, tx2, ty2))
            tile_img.save(os.path.join(out_img, tile_name), quality=95)

            # 라벨 저장
            lbl_name = f"{stem}_t{ti:02d}.txt"
            with open(os.path.join(out_lbl, lbl_name), 'w') as f:
                for cls, cx, cy, w, h in tile_bboxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    cls_stats[cls] += 1

            for _, _, _, w, h in tile_bboxes:
                area_stats.append(w * h * 100)

        if (idx + 1) % 2000 == 0:
            print(f"  {idx+1}/{len(label_files)} images → {positive_tiles} pos + {negative_tiles} neg tiles")

    total = positive_tiles + negative_tiles
    print(f"  [{split}] Done: {total} tiles (positive={positive_tiles}, negative={negative_tiles}, skipped_neg={skipped_negatives})")
    print(f"  Class 0 (helmet_o): {cls_stats[0]}, Class 1 (helmet_x): {cls_stats[1]}")

    if area_stats:
        area_stats.sort()
        n = len(area_stats)
        print(f"  Bbox Area P25={area_stats[n//4]:.3f}%, P50={area_stats[n//2]:.3f}%, P75={area_stats[3*n//4]:.3f}%")

    return total, positive_tiles, negative_tiles


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-ratio", type=float, default=0.3,
                        help="negative 타일 포함 비율 (0.3 = 30%)")
    args = parser.parse_args()

    print("=" * 60)
    print("v15 Tiled Training Dataset Builder")
    print(f"  Source: {SRC_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Tile: {TILE_SIZE}x{TILE_SIZE}, Overlap: {OVERLAP}")
    print(f"  Negative ratio: {args.negative_ratio}")
    print("=" * 60)

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    train_total, train_pos, train_neg = process_split("train", args.negative_ratio)
    valid_total, valid_pos, valid_neg = process_split("valid", args.negative_ratio)

    # data.yaml (서버 경로)
    data_yaml = (
        f"path: {OUT_DIR}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: person_with_helmet\n"
        f"  1: person_without_helmet\n"
    )
    with open(os.path.join(OUT_DIR, "data.yaml"), 'w') as f:
        f.write(data_yaml)

    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"  [train] {train_total} tiles (pos={train_pos}, neg={train_neg})")
    print(f"  [valid] {valid_total} tiles (pos={valid_pos}, neg={valid_neg})")
    print(f"  data.yaml: {os.path.join(OUT_DIR, 'data.yaml')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
