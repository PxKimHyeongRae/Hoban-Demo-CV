#!/usr/bin/env python3
"""
v13 Crop 실험용 데이터셋 빌더

전략:
- 원본 1920x1080 이미지에서 bbox 주변을 640x640으로 crop
- 작은 bbox(0.3~1% area)가 crop 후 3~10%로 확대됨
- 샘플링: train 3000장, valid 750장 (빠른 실험용)

실행: python build_crop_experiment.py
출력: datasets_v13_crop/
"""

import os
import random
import shutil
from pathlib import Path
from PIL import Image

# ===== Config =====
SRC_DIR = "/home/lay/hoban/datasets_v13"
OUT_DIR = "/home/lay/hoban/datasets_v13_crop"
CROP_SIZE = 640       # crop 크기 (정사각형)
TRAIN_SAMPLE = 3000   # train 샘플 수
VALID_SAMPLE = 750    # valid 샘플 수
SEED = 42
MIN_BBOX_IN_CROP = 0.5  # crop 안에 bbox가 50% 이상 들어와야 유효

random.seed(SEED)


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


def compute_crop_region(bboxes, img_w, img_h, crop_size):
    """bbox 중심 기준으로 crop 영역 계산 (pixel 좌표)"""
    if not bboxes:
        # negative: 랜덤 crop
        cx = random.randint(crop_size // 2, max(crop_size // 2, img_w - crop_size // 2))
        cy = random.randint(crop_size // 2, max(crop_size // 2, img_h - crop_size // 2))
    else:
        # bbox 중심들의 평균 위치
        cx_avg = sum(b[1] * img_w for b in bboxes) / len(bboxes)
        cy_avg = sum(b[2] * img_h for b in bboxes) / len(bboxes)

        # 약간의 jitter 추가 (augmentation 효과)
        jitter = crop_size * 0.15
        cx = cx_avg + random.uniform(-jitter, jitter)
        cy = cy_avg + random.uniform(-jitter, jitter)

    # crop 좌상단
    x1 = int(cx - crop_size // 2)
    y1 = int(cy - crop_size // 2)

    # 경계 clamp
    x1 = max(0, min(x1, img_w - crop_size))
    y1 = max(0, min(y1, img_h - crop_size))

    return x1, y1, x1 + crop_size, y1 + crop_size


def transform_bboxes(bboxes, crop_x1, crop_y1, crop_size, img_w, img_h):
    """원본 bbox → crop 기준 bbox로 변환. crop 밖 bbox 제거."""
    new_bboxes = []
    for cls, cx, cy, w, h in bboxes:
        # pixel 좌표로 변환
        px_cx = cx * img_w
        px_cy = cy * img_h
        px_w = w * img_w
        px_h = h * img_h

        # bbox xyxy
        bx1 = px_cx - px_w / 2
        by1 = px_cy - px_h / 2
        bx2 = px_cx + px_w / 2
        by2 = px_cy + px_h / 2

        # crop 영역과의 교집합
        ix1 = max(bx1, crop_x1)
        iy1 = max(by1, crop_y1)
        ix2 = min(bx2, crop_x1 + crop_size)
        iy2 = min(by2, crop_y1 + crop_size)

        if ix1 >= ix2 or iy1 >= iy2:
            continue  # crop 밖

        # 교집합 비율 체크
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        orig_area = px_w * px_h
        if orig_area > 0 and inter_area / orig_area < MIN_BBOX_IN_CROP:
            continue  # 너무 잘림

        # crop 기준 normalized 좌표
        new_cx = ((ix1 + ix2) / 2 - crop_x1) / crop_size
        new_cy = ((iy1 + iy2) / 2 - crop_y1) / crop_size
        new_w = (ix2 - ix1) / crop_size
        new_h = (iy2 - iy1) / crop_size

        # clamp
        new_cx = max(0, min(1, new_cx))
        new_cy = max(0, min(1, new_cy))
        new_w = min(new_w, 1.0)
        new_h = min(new_h, 1.0)

        if new_w > 0.01 and new_h > 0.01:  # 너무 작으면 제거
            new_bboxes.append((cls, new_cx, new_cy, new_w, new_h))

    return new_bboxes


def process_split(split, n_samples):
    """한 split (train/valid) 처리"""
    img_dir = os.path.join(SRC_DIR, split, "images")
    lbl_dir = os.path.join(SRC_DIR, split, "labels")
    out_img = os.path.join(OUT_DIR, split, "images")
    out_lbl = os.path.join(OUT_DIR, split, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    # 파일 목록
    all_labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    random.shuffle(all_labels)

    # negative vs positive 분리
    positives = []
    negatives = []
    for lf in all_labels:
        bboxes = parse_labels(os.path.join(lbl_dir, lf))
        if bboxes:
            positives.append(lf)
        else:
            negatives.append(lf)

    # 샘플링: positive 80%, negative 20%
    n_pos = min(int(n_samples * 0.8), len(positives))
    n_neg = min(n_samples - n_pos, len(negatives))
    selected = positives[:n_pos] + negatives[:n_neg]
    random.shuffle(selected)

    print(f"\n[{split}] {len(selected)} images (pos={n_pos}, neg={n_neg})")

    success = 0
    area_before = []
    area_after = []

    for i, lf in enumerate(selected):
        stem = lf.replace('.txt', '')
        img_name = stem + '.jpg'
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size
        bboxes = parse_labels(os.path.join(lbl_dir, lf))

        # area 기록 (before)
        for _, _, _, w, h in bboxes:
            area_before.append(w * h * 100)

        # crop 영역 계산
        cx1, cy1, cx2, cy2 = compute_crop_region(bboxes, img_w, img_h, CROP_SIZE)

        # crop 실행
        cropped = img.crop((cx1, cy1, cx2, cy2))

        # bbox 변환
        new_bboxes = transform_bboxes(bboxes, cx1, cy1, CROP_SIZE, img_w, img_h)

        # positive 이미지인데 crop 후 bbox가 0개면 스킵
        if bboxes and not new_bboxes:
            continue

        # area 기록 (after)
        for _, _, _, w, h in new_bboxes:
            area_after.append(w * h * 100)

        # 저장
        cropped.save(os.path.join(out_img, img_name), quality=95)
        with open(os.path.join(out_lbl, lf), 'w') as f:
            for cls, cx, cy, w, h in new_bboxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        success += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(selected)} processed...")

    print(f"  완료: {success} images saved")

    if area_before and area_after:
        area_before.sort()
        area_after.sort()
        n_b, n_a = len(area_before), len(area_after)
        print(f"  Area 변화 (P50): {area_before[n_b//2]:.3f}% → {area_after[n_a//2]:.3f}%")
        print(f"  Area 변화 (P25): {area_before[n_b//4]:.3f}% → {area_after[n_a//4]:.3f}%")
        print(f"  Area 변화 (P75): {area_before[3*n_b//4]:.3f}% → {area_after[3*n_a//4]:.3f}%")

    return success


def main():
    print("=" * 60)
    print("v13 Crop Experiment Dataset Builder")
    print(f"  Source: {SRC_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"  Samples: train={TRAIN_SAMPLE}, valid={VALID_SAMPLE}")
    print("=" * 60)

    # 기존 출력 삭제
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # 처리
    train_count = process_split("train", TRAIN_SAMPLE)
    valid_count = process_split("valid", VALID_SAMPLE)

    # data.yaml 생성 (Windows 경로)
    data_yaml = (
        f"path: Z:\\home\\lay\\hoban\\datasets_v13_crop\n"
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
    print(f"Dataset ready!")
    print(f"  Train: {train_count} images")
    print(f"  Valid: {valid_count} images")
    print(f"  data.yaml: {os.path.join(OUT_DIR, 'data.yaml')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
