#!/usr/bin/env python3
"""
go3k v19: v17 base + CVAT 검수 helmet_off 946장 + CVAT 검수 negative 960장

데이터: v16 base (10,564) + helmet_1k_manual (946) + neg_1k_manual (960) = ~12,470 train
시작 가중치: v17 best.pt
학습: 100ep, SGD lr0=0.005, patience=40

사용법:
  1. 데이터셋 준비: python train_go3k_v19.py --prepare
  2. 학습 시작:     python train_go3k_v19.py
  3. 이어서 학습:    python train_go3k_v19.py --resume
"""
import argparse
import os
import shutil
import glob
import random


HOBAN = "/home/lay/hoban"
V16_DATASET = f"{HOBAN}/datasets_go3k_v16"
V19_DATASET = f"{HOBAN}/datasets_go3k_v19"
V17_WEIGHTS = f"{HOBAN}/hoban_go3k_v17/weights/best.pt"

# 추가 데이터 소스 (CVAT 검수 완료)
EXTRA_SOURCES = [
    # (이름, 이미지 디렉터리, 라벨 디렉터리, 설명, max_count)
    ("helmet_1k_manual",
     f"{HOBAN}/datasets/cvat/helmet_1k_manual/images",
     f"{HOBAN}/datasets/cvat/helmet_1k_manual/labels",
     "CVAT 검수 helmet_off 946장",
     None),
    ("neg_1k_manual",
     f"{HOBAN}/datasets/cvat/neg_1k_manual/images",
     f"{HOBAN}/datasets/cvat/neg_1k_manual/labels",
     "CVAT 검수 negative 960장 (중복 제거)",
     None),
]


def prepare_dataset():
    """v19 데이터셋 준비: v16 base + helmet_off + negative"""
    print("=" * 60)
    print("  v19 데이터셋 준비")
    print("=" * 60)

    train_img = f"{V19_DATASET}/train/images"
    train_lbl = f"{V19_DATASET}/train/labels"
    val_img = f"{V19_DATASET}/valid/images"
    val_lbl = f"{V19_DATASET}/valid/labels"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # 1. v16 base 데이터 심볼릭 링크
    print("\n[1] v16 base 데이터 링크...")
    v16_train_imgs = glob.glob(f"{V16_DATASET}/train/images/*.jpg")
    v16_train_lbls = glob.glob(f"{V16_DATASET}/train/labels/*.txt")
    v16_val_imgs = glob.glob(f"{V16_DATASET}/valid/images/*.jpg")
    v16_val_lbls = glob.glob(f"{V16_DATASET}/valid/labels/*.txt")

    linked = 0
    for src in v16_train_imgs + v16_train_lbls:
        dst = src.replace(V16_DATASET, V19_DATASET)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            linked += 1
    for src in v16_val_imgs + v16_val_lbls:
        dst = src.replace(V16_DATASET, V19_DATASET)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            linked += 1
    print(f"  v16 링크: {linked}개 (train: {len(v16_train_imgs)}, val: {len(v16_val_imgs)})")

    # 2. 추가 데이터 복사 (train에 추가)
    total_added = 0
    existing_train = set(os.listdir(train_img))

    for name, img_dir, lbl_dir, desc, max_count in EXTRA_SOURCES:
        if not os.path.isdir(img_dir):
            print(f"\n[{name}] 디렉터리 없음 (스킵): {img_dir}")
            continue

        imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))

        # max_count 제한 시 랜덤 샘플링
        if max_count and len(imgs) > max_count:
            random.seed(42)
            imgs = random.sample(imgs, max_count)

        added = 0
        for fname in imgs:
            if fname in existing_train:
                continue

            src_img = os.path.join(img_dir, fname)
            lbl_name = fname.replace(".jpg", ".txt")
            src_lbl = os.path.join(lbl_dir, lbl_name)
            dst_img = os.path.join(train_img, fname)
            dst_lbl = os.path.join(train_lbl, lbl_name)

            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # negative: 빈 라벨 생성
                open(dst_lbl, "w").close()
            added += 1
            existing_train.add(fname)

        total_added += added
        print(f"\n[{name}] {desc}")
        print(f"  가용: {len(imgs)}장, 추가: {added}장")

    # 3. data.yaml
    yaml_path = f"{V19_DATASET}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {V19_DATASET}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n\n")
        f.write("nc: 2\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")

    final_train = len([f for f in os.listdir(train_img) if f.endswith(".jpg")])
    final_val = len([f for f in os.listdir(val_img) if f.endswith(".jpg")])
    print(f"\n{'='*60}")
    print(f"  v19 데이터셋 완성")
    print(f"  Train: {final_train}장 (v16 {len(v16_train_imgs)} + 추가 {total_added})")
    print(f"  Val:   {final_val}장")
    print(f"  YAML:  {yaml_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train go3k v19")
    parser.add_argument("--prepare", action="store_true", help="데이터셋 준비만")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare_dataset()
        return

    project = HOBAN
    name = "hoban_go3k_v19"
    data_yaml = f"{V19_DATASET}/data.yaml"

    if not os.path.exists(data_yaml):
        print("데이터셋 없음. 먼저 --prepare 실행:")
        print("  python train_go3k_v19.py --prepare")
        return

    from ultralytics import YOLO

    if args.resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print(f"=== go3k v19: v17 + helmet_off + negative ===")
    print(f"  Base weights: {V17_WEIGHTS}")
    print(f"  Data: {data_yaml}")
    print(f"  imgsz: 1280, batch: {args.batch}")
    print(f"  epochs: {args.epochs}, lr0: 0.005")
    print()

    model = YOLO(V17_WEIGHTS)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=1280,
        batch=args.batch,
        device="0",
        project=project,
        name=name,
        exist_ok=True,

        # SGD
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=2.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (v17과 동일)
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
        patience=40,
        amp=True,
        workers=4,
        seed=42,
        plots=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/{name}/")


if __name__ == "__main__":
    main()
