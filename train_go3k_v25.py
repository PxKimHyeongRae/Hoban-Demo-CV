#!/usr/bin/env python3
"""v25 3-class: L티어 헬멧 2,000장 + v24 fallen (area<5% 필터링)

클래스:
  0: person_with_helmet
  1: person_without_helmet
  2: fallen

데이터 구성:
  - L 티어 train: 2,000장 (helmet_on + helmet_off, 현장 데이터)
  - Fallen: v24 데이터셋에서 area < 5% 필터링 (CCTV 현실적 크기)
  - Val: 기존 helmet 605+88장 + v24 fallen val 100장

사용법:
  python train_3class_exp.py --prepare   # 데이터셋 빌드
  python train_3class_exp.py --train     # 학습
  python train_3class_exp.py --resume    # 이어서 학습
"""
import os
import sys
import shutil
import random
from pathlib import Path

random.seed(42)

HOBAN = "/home/lay/hoban"
OUT_DIR = f"{HOBAN}/datasets_go3k_v25"
MODEL = f"{HOBAN}/yolo26m.pt"

# 소스
L_TIER_TRAIN_IMG = f"{HOBAN}/datasets_minimal_l/train/images"
L_TIER_TRAIN_LBL = f"{HOBAN}/datasets_minimal_l/train/labels"
L_TIER_VAL_IMG = f"{HOBAN}/datasets_minimal_l/valid/images"
L_TIER_VAL_LBL = f"{HOBAN}/datasets_minimal_l/valid/labels"

V24_TRAIN_IMG = f"{HOBAN}/datasets_go3k_v24/train/images"
V24_TRAIN_LBL = f"{HOBAN}/datasets_go3k_v24/train/labels"
V24_VAL_IMG = f"{HOBAN}/datasets_go3k_v24/valid/images"
V24_VAL_LBL = f"{HOBAN}/datasets_go3k_v24/valid/labels"

# 최대 fallen 수 (area 필터 후 전부 사용, 상한만 설정)
MAX_FALLEN = 800


def parse_labels(label_path):
    """라벨 파일 파싱 → [(class_id, cx, cy, w, h), ...]"""
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


def filter_fallen_by_area(max_area=0.05):
    """v24 fallen 이미지 중 area < max_area인 bbox만 가진 이미지 선별"""
    fallen_images = []

    for lbl_name in os.listdir(V24_TRAIN_LBL):
        if not lbl_name.startswith("fallen_"):
            continue
        lbl_path = os.path.join(V24_TRAIN_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)

        # fallen bbox (class 2)만 확인
        fallen_bboxes = [(c, cx, cy, w, h) for c, cx, cy, w, h in bboxes if c == 2]
        if not fallen_bboxes:
            continue

        # 모든 fallen bbox가 area < max_area인 이미지만 선택
        all_small = all(w * h < max_area for _, _, _, w, h in fallen_bboxes)
        if all_small:
            img_name = lbl_name.replace(".txt", ".jpg")
            if os.path.exists(os.path.join(V24_TRAIN_IMG, img_name)):
                avg_area = sum(w * h for _, _, _, w, h in fallen_bboxes) / len(fallen_bboxes)
                fallen_images.append((img_name, lbl_name, avg_area, len(fallen_bboxes)))

    # area 작은 순으로 정렬 (CCTV에 더 유사한 것 우선)
    fallen_images.sort(key=lambda x: x[2])
    return fallen_images


def filter_fallen_val():
    """v24 val에서 fallen 이미지 추출"""
    fallen_val = []
    for lbl_name in os.listdir(V24_VAL_LBL):
        if not lbl_name.startswith("fallen_"):
            continue
        lbl_path = os.path.join(V24_VAL_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)
        fallen_bboxes = [b for b in bboxes if b[0] == 2]
        if fallen_bboxes:
            img_name = lbl_name.replace(".txt", ".jpg")
            if os.path.exists(os.path.join(V24_VAL_IMG, img_name)):
                fallen_val.append((img_name, lbl_name))
    return fallen_val


def prepare():
    """3-class 데이터셋 빌드"""
    print("=" * 60)
    print("  3-class 데이터셋 빌드")
    print("  L 티어 helmet 2,000장 + v24 fallen (area<5%)")
    print("=" * 60)

    # 디렉터리 생성
    for subdir in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)

    train_img_dir = os.path.join(OUT_DIR, "train", "images")
    train_lbl_dir = os.path.join(OUT_DIR, "train", "labels")
    val_img_dir = os.path.join(OUT_DIR, "valid", "images")
    val_lbl_dir = os.path.join(OUT_DIR, "valid", "labels")

    # ── 1. L 티어 helmet train (class 0, 1) ──
    print("\n[1/4] L 티어 helmet train 복사...")
    helmet_count = 0
    for img_name in os.listdir(L_TIER_TRAIN_IMG):
        if not img_name.endswith((".jpg", ".png")):
            continue
        lbl_name = Path(img_name).stem + ".txt"
        src_img = os.path.join(L_TIER_TRAIN_IMG, img_name)
        src_lbl = os.path.join(L_TIER_TRAIN_LBL, lbl_name)

        dst_img = os.path.join(train_img_dir, img_name)
        dst_lbl = os.path.join(train_lbl_dir, lbl_name)

        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)
        helmet_count += 1
    print(f"  Helmet train: {helmet_count}장")

    # ── 2. Fallen train (area < 5% 필터링) ──
    print("\n[2/4] Fallen train 필터링 (area < 5%)...")
    fallen_candidates = filter_fallen_by_area(max_area=0.05)
    fallen_selected = fallen_candidates[:MAX_FALLEN]

    fallen_count = 0
    fallen_bbox_count = 0
    for img_name, lbl_name, avg_area, n_bbox in fallen_selected:
        src_img = os.path.join(V24_TRAIN_IMG, img_name)
        src_lbl = os.path.join(V24_TRAIN_LBL, lbl_name)
        dst_img = os.path.join(train_img_dir, img_name)
        dst_lbl = os.path.join(train_lbl_dir, lbl_name)

        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)
        fallen_count += 1
        fallen_bbox_count += n_bbox

    print(f"  Fallen candidates (area<5%): {len(fallen_candidates)}장")
    print(f"  Fallen selected: {fallen_count}장 ({fallen_bbox_count} bbox)")
    if fallen_selected:
        print(f"  Area range: {fallen_selected[0][2]:.4f} ~ {fallen_selected[-1][2]:.4f}")

    # ── 3. Helmet val (L 티어 val = 기존 605+88장) ──
    print("\n[3/4] Helmet val 복사...")
    val_helmet = 0
    for img_name in os.listdir(L_TIER_VAL_IMG):
        if not img_name.endswith((".jpg", ".png")):
            continue
        lbl_name = Path(img_name).stem + ".txt"
        src_img = os.path.join(L_TIER_VAL_IMG, img_name)
        src_lbl = os.path.join(L_TIER_VAL_LBL, lbl_name)
        dst_img = os.path.join(val_img_dir, img_name)
        dst_lbl = os.path.join(val_lbl_dir, lbl_name)

        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)
        val_helmet += 1
    print(f"  Helmet val: {val_helmet}장")

    # ── 4. Fallen val ──
    print("\n[4/4] Fallen val 복사...")
    fallen_val = filter_fallen_val()
    val_fallen = 0
    for img_name, lbl_name in fallen_val:
        src_img = os.path.join(V24_VAL_IMG, img_name)
        src_lbl = os.path.join(V24_VAL_LBL, lbl_name)
        dst_img = os.path.join(val_img_dir, img_name)
        dst_lbl = os.path.join(val_lbl_dir, lbl_name)

        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)
        val_fallen += 1
    print(f"  Fallen val: {val_fallen}장")

    # ── data.yaml ──
    data_yaml = os.path.join(OUT_DIR, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: 3\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")
        f.write("  2: fallen\n")

    total_train = helmet_count + fallen_count
    total_val = val_helmet + val_fallen
    fallen_ratio = fallen_count / total_train * 100 if total_train > 0 else 0

    print(f"\n{'='*60}")
    print(f"  데이터셋 완성: {OUT_DIR}")
    print(f"  Train: {total_train}장 (helmet {helmet_count} + fallen {fallen_count}, {fallen_ratio:.1f}%)")
    print(f"  Val: {total_val}장 (helmet {val_helmet} + fallen {val_fallen})")
    print(f"  data.yaml: {data_yaml}")
    print(f"{'='*60}")


def train(batch=4, epochs=100, resume=False):
    """3-class 학습"""
    from ultralytics import YOLO

    project = HOBAN
    name = "hoban_go3k_v25"
    data_yaml = os.path.join(OUT_DIR, "data.yaml")

    if not os.path.exists(data_yaml):
        print("데이터셋 없음. 먼저 --prepare 실행")
        return

    if resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        if not os.path.exists(ckpt):
            print(f"체크포인트 없음: {ckpt}")
            return
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print("=" * 60)
    print("  3-class 실험: L 티어 + fallen (area<5%)")
    print(f"  Model: yolo26m.pt (COCO pretrained)")
    print(f"  Optimizer: SGD, lr0=0.005, 1280px, batch={batch}")
    print("=" * 60)

    model = YOLO(MODEL)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=1280,
        batch=batch,
        device="0",
        project=project,
        name=name,
        exist_ok=True,

        # SGD (검증된 설정)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.15,
        close_mosaic=10,

        # Early stopping
        patience=20,
        amp=True,
        workers=4,
        seed=42,
        plots=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/{name}/")
    print(f"Best weights: {project}/{name}/weights/best.pt")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="3-class 실험 (helmet + fallen)")
    parser.add_argument("--prepare", action="store_true", help="데이터셋 빌드")
    parser.add_argument("--train", action="store_true", help="학습 시작")
    parser.add_argument("--resume", action="store_true", help="학습 재개")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.train:
        train(batch=args.batch, epochs=args.epochs)
    elif args.resume:
        train(batch=args.batch, resume=True)
    else:
        parser.print_help()
        print("\n예시:")
        print("  python train_3class_exp.py --prepare   # 데이터셋 빌드")
        print("  python train_3class_exp.py --train     # 학습")
        print("  python train_3class_exp.py --resume    # 재개")


if __name__ == "__main__":
    main()
