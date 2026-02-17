#!/usr/bin/env python3
"""
go3k v18: v17 fine-tune + helmet_off 강화 + hard negatives

시작 가중치: v17 best.pt (transfer learning)
데이터: v16 base + verified helmet_off + ④ 추가 데이터
학습: 짧은 fine-tune (50ep), 낮은 lr (0.001)

사용법:
  1. 먼저 데이터셋 준비: python train_go3k_v18.py --prepare
  2. 학습 시작:          python train_go3k_v18.py
  3. 이어서 학습:         python train_go3k_v18.py --resume
"""
import argparse
import os
import shutil
import glob


HOBAN = "/home/lay/hoban"
V16_DATASET = f"{HOBAN}/datasets_go3k_v16"
V18_DATASET = f"{HOBAN}/datasets_go3k_v18"
V17_WEIGHTS = f"{HOBAN}/hoban_go3k_v17/weights/best.pt"

# 추가 데이터 소스
EXTRA_SOURCES = [
    # (이름, 이미지 디렉터리, 라벨 디렉터리, 설명)
    ("cvat_helmet_off_137",
     f"{HOBAN}/datasets/cvat_helmet_off/valid_helmet_off_137/images",
     f"{HOBAN}/datasets/cvat_helmet_off/valid_helmet_off_137/labels",
     "CVAT 검증 helmet_off 137장"),
    ("helmet_off_v17",
     f"{HOBAN}/datasets/helmet_off_v17/results/images",
     f"{HOBAN}/datasets/helmet_off_v17/results/labels",
     "④ v17 helmet_off 추출"),
    ("hard_neg_v17",
     f"{HOBAN}/datasets/hard_neg_v17/results/images",
     f"{HOBAN}/datasets/hard_neg_v17/results/labels",
     "④ v17 hard negatives (빈 라벨)"),
]


def prepare_dataset():
    """v18 데이터셋 준비: v16 base + 추가 데이터"""
    print("=" * 60)
    print("  v18 데이터셋 준비")
    print("=" * 60)

    train_img = f"{V18_DATASET}/train/images"
    train_lbl = f"{V18_DATASET}/train/labels"
    val_img = f"{V18_DATASET}/valid/images"
    val_lbl = f"{V18_DATASET}/valid/labels"

    # 디렉터리 생성
    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # 1. v16 base 데이터 심볼릭 링크 (용량 절약)
    print("\n[1] v16 base 데이터 링크...")
    v16_train_imgs = glob.glob(f"{V16_DATASET}/train/images/*.jpg")
    v16_train_lbls = glob.glob(f"{V16_DATASET}/train/labels/*.txt")
    v16_val_imgs = glob.glob(f"{V16_DATASET}/valid/images/*.jpg")
    v16_val_lbls = glob.glob(f"{V16_DATASET}/valid/labels/*.txt")

    linked = 0
    for src in v16_train_imgs + v16_train_lbls:
        dst = src.replace(V16_DATASET, V18_DATASET)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            linked += 1
    for src in v16_val_imgs + v16_val_lbls:
        dst = src.replace(V16_DATASET, V18_DATASET)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            linked += 1
    print(f"  v16 링크: {linked}개 (train: {len(v16_train_imgs)} img, val: {len(v16_val_imgs)} img)")

    # 2. 추가 데이터 복사 (train에 추가)
    total_added = 0
    existing_train = set(os.listdir(train_img))

    for name, img_dir, lbl_dir, desc in EXTRA_SOURCES:
        if not os.path.isdir(img_dir):
            print(f"\n[{name}] 디렉터리 없음 (스킵): {img_dir}")
            continue

        imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        added = 0
        for fname in imgs:
            if fname in existing_train:
                continue  # 중복 방지
            src_img = os.path.join(img_dir, fname)
            src_lbl = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
            dst_img = os.path.join(train_img, fname)
            dst_lbl = os.path.join(train_lbl, fname.replace(".jpg", ".txt"))

            shutil.copy2(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                # hard negative: 빈 라벨 생성
                open(dst_lbl, "w").close()
            added += 1
            existing_train.add(fname)

        total_added += added
        print(f"\n[{name}] {desc}")
        print(f"  가용: {len(imgs)}장, 추가: {added}장")

    # 3. data.yaml 생성
    yaml_path = f"{V18_DATASET}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {V18_DATASET}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("\n")
        f.write("nc: 2\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")

    # 최종 통계
    final_train = len([f for f in os.listdir(train_img) if f.endswith(".jpg")])
    final_val = len([f for f in os.listdir(val_img) if f.endswith(".jpg")])
    print(f"\n{'='*60}")
    print(f"  v18 데이터셋 완성")
    print(f"  Train: {final_train}장 (v16 {len(v16_train_imgs)} + 추가 {total_added})")
    print(f"  Val:   {final_val}장")
    print(f"  YAML:  {yaml_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train go3k v18")
    parser.add_argument("--prepare", action="store_true", help="데이터셋 준비만")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare_dataset()
        return

    project = HOBAN
    name = "hoban_go3k_v18"
    data_yaml = f"{V18_DATASET}/data.yaml"

    if not os.path.exists(data_yaml):
        print(f"데이터셋 없음. 먼저 --prepare 실행:")
        print(f"  python train_go3k_v18.py --prepare")
        return

    from ultralytics import YOLO

    if args.resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print(f"=== go3k v18: v17 fine-tune ===")
    print(f"  Base: {V17_WEIGHTS}")
    print(f"  Data: {data_yaml}")
    print(f"  imgsz: 1280, batch: {args.batch}")
    print(f"  epochs: {args.epochs}, lr0: 0.001 (v17의 1/5)")
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

        # SGD (v17과 동일, lr만 낮게)
        optimizer="SGD",
        lr0=0.001,       # v17의 1/5 (fine-tune)
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=1.0,  # 짧은 warmup (이미 수렴된 모델)
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
        patience=15,  # 더 빠른 조기 종료 (fine-tune이므로)
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
