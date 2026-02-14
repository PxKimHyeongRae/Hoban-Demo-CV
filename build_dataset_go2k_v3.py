#!/usr/bin/env python3
"""
go2k_v3 데이터셋 구축:
  - go2k_manual 604장 → 640x640 타일 분할 (SAHI 추론과 동일 형태)
  - v13 서브샘플 8,000장 (go2k_v2에서 복사)
  - 타일 train/valid 분할: go2k_manual의 기존 train(479)/valid(120) + 나머지 5장
  - 오버샘플링 제거 (타일화로 자연 증강)

실행: python build_dataset_go2k_v3.py
"""
import os
import shutil
import cv2
import numpy as np
from collections import defaultdict

# 소스
GO2K_MANUAL_IMG = "/home/lay/hoban/datasets/go2k_manual/images"
GO2K_MANUAL_LBL = "/home/lay/hoban/datasets/go2k_manual/labels"
GO2K_V2_TRAIN = "/home/lay/hoban/datasets_go2k_v2/train"
GO2K_V2_VALID = "/home/lay/hoban/datasets_go2k_v2/valid"

# 출력
OUT_DIR = "/home/lay/hoban/datasets_go2k_v3"

# 타일 설정 (SAHI 추론과 동일)
TILE_SIZE = 640
OVERLAP = 0.2
MIN_BBOX_AREA = 100  # 타일 내 최소 bbox 면적 (px²)


def calc_slices(img_h, img_w, tile_size, overlap):
    step = int(tile_size * (1 - overlap))
    slices = []
    y = 0
    while y < img_h:
        y_end = min(y + tile_size, img_h)
        x = 0
        while x < img_w:
            x_end = min(x + tile_size, img_w)
            slices.append((x, y, x_end, y_end))
            if x_end >= img_w:
                break
            x += step
        if y_end >= img_h:
            break
        y += step
    return slices


def tile_image_and_labels(img_path, lbl_path, tile_size, overlap):
    """이미지와 라벨을 타일로 분할. 각 타일의 (crop, labels) 리스트 반환"""
    img = cv2.imread(img_path)
    if img is None:
        return []

    img_h, img_w = img.shape[:2]
    slices = calc_slices(img_h, img_w, tile_size, overlap)

    # YOLO 라벨 로드 (절대 좌표로 변환)
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                gt_boxes.append((cls, x1, y1, x2, y2))

    tiles = []
    for si, (sx, sy, ex, ey) in enumerate(slices):
        crop = img[sy:ey, sx:ex]
        crop_h, crop_w = crop.shape[:2]

        # 이 타일에 포함되는 bbox 찾기
        tile_labels = []
        for cls, bx1, by1, bx2, by2 in gt_boxes:
            # bbox와 타일의 교집합
            ix1 = max(bx1, sx)
            iy1 = max(by1, sy)
            ix2 = min(bx2, ex)
            iy2 = min(by2, ey)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # 원본 bbox 대비 교집합 비율 (너무 작으면 무시)
            orig_area = (bx2 - bx1) * (by2 - by1)
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            if inter_area < MIN_BBOX_AREA:
                continue
            if inter_area / orig_area < 0.3:  # 원본의 30% 미만이면 무시
                continue

            # 타일 내 상대 좌표 (YOLO format)
            tcx = ((ix1 + ix2) / 2 - sx) / crop_w
            tcy = ((iy1 + iy2) / 2 - sy) / crop_h
            tw = (ix2 - ix1) / crop_w
            th = (iy2 - iy1) / crop_h

            # 범위 클리핑
            tcx = max(0.001, min(0.999, tcx))
            tcy = max(0.001, min(0.999, tcy))
            tw = min(tw, 1.0)
            th = min(th, 1.0)

            tile_labels.append(f"{cls} {tcx:.6f} {tcy:.6f} {tw:.6f} {th:.6f}")

        tiles.append((si, crop, tile_labels))

    return tiles


def get_go2k_split():
    """go2k_v2의 train/valid 분할 가져오기 (오버샘플 제외)"""
    train_files = set()
    for f in os.listdir(os.path.join(GO2K_V2_TRAIN, "images")):
        if f.startswith("cam") and "_x" not in f:
            train_files.add(f)

    valid_files = set()
    for f in os.listdir(os.path.join(GO2K_V2_VALID, "images")):
        if f.startswith("cam"):
            valid_files.add(f)

    return train_files, valid_files


def main():
    # 디렉토리 생성
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

    # go2k train/valid 분할 가져오기
    go2k_train, go2k_valid = get_go2k_split()
    print(f"go2k train: {len(go2k_train)}장, valid: {len(go2k_valid)}장")

    # 전체 go2k_manual 파일
    all_go2k = sorted([f for f in os.listdir(GO2K_MANUAL_IMG) if f.endswith(".jpg")])
    print(f"go2k_manual 전체: {len(all_go2k)}장")

    # train/valid에 없는 파일은 train으로
    unassigned = [f for f in all_go2k if f not in go2k_train and f not in go2k_valid]
    if unassigned:
        print(f"미분류 {len(unassigned)}장 → train에 추가")
        go2k_train.update(unassigned)

    # 1) go2k_manual 타일 생성
    stats = defaultdict(int)
    for split_name, split_files in [("train", go2k_train), ("valid", go2k_valid)]:
        print(f"\n[{split_name}] go2k 타일 생성 ({len(split_files)}장)...")
        tile_count = 0
        label_count = 0
        empty_tiles = 0

        for fi, fname in enumerate(sorted(split_files)):
            img_path = os.path.join(GO2K_MANUAL_IMG, fname)
            lbl_path = os.path.join(GO2K_MANUAL_LBL, fname.replace(".jpg", ".txt"))

            if not os.path.exists(img_path):
                continue

            tiles = tile_image_and_labels(img_path, lbl_path, TILE_SIZE, OVERLAP)
            base = fname.replace(".jpg", "")

            for si, crop, labels in tiles:
                tile_name = f"{base}_tile{si:02d}"

                # 이미지 저장
                out_img = os.path.join(OUT_DIR, split_name, "images", f"{tile_name}.jpg")
                cv2.imwrite(out_img, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 라벨 저장 (빈 타일도 negative sample로 저장)
                out_lbl = os.path.join(OUT_DIR, split_name, "labels", f"{tile_name}.txt")
                with open(out_lbl, "w") as f:
                    if labels:
                        f.write("\n".join(labels) + "\n")

                tile_count += 1
                label_count += len(labels)
                if not labels:
                    empty_tiles += 1

            if (fi + 1) % 100 == 0:
                print(f"  {fi + 1}/{len(split_files)}...")

        stats[f"{split_name}_tiles"] = tile_count
        stats[f"{split_name}_labels"] = label_count
        stats[f"{split_name}_empty"] = empty_tiles
        print(f"  → {tile_count}타일 ({label_count} bbox, 빈타일 {empty_tiles}개)")

    # 2) v13 서브샘플 복사 (심볼릭 링크로 용량 절약)
    print(f"\n[train] v13 서브샘플 복사...")
    v13_count = 0
    for f in sorted(os.listdir(os.path.join(GO2K_V2_TRAIN, "images"))):
        if not f.startswith("cam") and not "_x" in f:  # v13 파일 (S2-* 등)
            src_img = os.path.join(GO2K_V2_TRAIN, "images", f)
            src_lbl = os.path.join(GO2K_V2_TRAIN, "labels", f.replace(".jpg", ".txt"))
            dst_img = os.path.join(OUT_DIR, "train", "images", f)
            dst_lbl = os.path.join(OUT_DIR, "train", "labels", f.replace(".jpg", ".txt"))

            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(src_img), dst_img)
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                os.symlink(os.path.abspath(src_lbl), dst_lbl)
            v13_count += 1

    # v13 valid도 복사
    v13_valid_count = 0
    for f in sorted(os.listdir(os.path.join(GO2K_V2_VALID, "images"))):
        if not f.startswith("cam"):
            src_img = os.path.join(GO2K_V2_VALID, "images", f)
            src_lbl = os.path.join(GO2K_V2_VALID, "labels", f.replace(".jpg", ".txt"))
            dst_img = os.path.join(OUT_DIR, "valid", "images", f)
            dst_lbl = os.path.join(OUT_DIR, "valid", "labels", f.replace(".jpg", ".txt"))

            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(src_img), dst_img)
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                os.symlink(os.path.abspath(src_lbl), dst_lbl)
            v13_valid_count += 1

    print(f"  → train v13: {v13_count}장, valid v13: {v13_valid_count}장")

    # 3) data.yaml 생성
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: 2\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")

    # 최종 통계
    train_total = len(os.listdir(os.path.join(OUT_DIR, "train", "images")))
    valid_total = len(os.listdir(os.path.join(OUT_DIR, "valid", "images")))

    print(f"\n{'=' * 60}")
    print(f"go2k_v3 데이터셋 생성 완료!")
    print(f"{'=' * 60}")
    print(f"  위치: {OUT_DIR}")
    print(f"  Train: {train_total}장")
    print(f"    - go2k 타일: {stats['train_tiles']}개 ({stats['train_labels']} bbox, 빈타일 {stats['train_empty']})")
    print(f"    - v13 서브샘플: {v13_count}장")
    print(f"  Valid: {valid_total}장")
    print(f"    - go2k 타일: {stats['valid_tiles']}개 ({stats['valid_labels']} bbox, 빈타일 {stats['valid_empty']})")
    print(f"    - v13 서브샘플: {v13_valid_count}장")
    print(f"\n  타일 설정: {TILE_SIZE}x{TILE_SIZE}, overlap={OVERLAP}")


if __name__ == "__main__":
    main()
