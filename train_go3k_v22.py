#!/usr/bin/env python3
"""
go3k v22: yolo26x COCO pretrained + v19 데이터셋

모델: yolo26x.pt (59.0M params, COCO pretrained) — yolo26m 대비 2.7x
데이터: v19 dataset (v16 base 10,564 + helmet_off 946 + neg 960 = 12,470 train)
학습: 100ep, SGD lr0=0.005, patience=20, batch=2

사용법:
  python train_go3k_v22.py              # 학습 시작
  python train_go3k_v22.py --resume     # 이어서 학습
"""
import argparse
from ultralytics import YOLO

HOBAN = "/home/lay/hoban"
MODEL = f"{HOBAN}/yolo26x.pt"
DATA = f"{HOBAN}/datasets_go3k_v19/data.yaml"
PROJECT = HOBAN
NAME = "hoban_go3k_v22"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.resume:
        ckpt = f"{PROJECT}/{NAME}/weights/last.pt"
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print(f"=== go3k v21-x: yolo26x COCO pt ===")
    print(f"  Model: {MODEL} (59.0M params)")
    print(f"  Data:  {DATA}")
    print(f"  imgsz: 1280, batch: {args.batch}")
    print(f"  epochs: {args.epochs}, lr0: 0.005")
    print()

    model = YOLO(MODEL)

    model.train(
        data=DATA,
        epochs=args.epochs,
        imgsz=1280,
        batch=args.batch,
        device="0",
        project=PROJECT,
        name=NAME,
        exist_ok=True,

        # SGD (v17 동일)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (v17 동일)
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

    print(f"\nDone! Results: {PROJECT}/{NAME}/")


if __name__ == "__main__":
    main()
