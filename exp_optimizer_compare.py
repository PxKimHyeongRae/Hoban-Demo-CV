#!/usr/bin/env python3
"""SGD vs AdamW 옵티마이저 비교 실험

M 티어(1,000장) 데이터셋으로 SGD/AdamW 비교.
SGD는 이미 학습 완료 (hoban_minimal_m), AdamW만 실행.

사용법:
  python exp_optimizer_compare.py              # AdamW 학습
  python exp_optimizer_compare.py --eval-only  # 평가만
"""
import argparse
import os

HOBAN = "/home/lay/hoban"
DATA_YAML = f"{HOBAN}/datasets_minimal_m/data.yaml"
MODEL = f"{HOBAN}/yolo26m.pt"
SEED = 42


def train_adamw(epochs=100, batch=6):
    """AdamW로 M 티어 학습"""
    from ultralytics import YOLO

    project = HOBAN
    name = "hoban_minimal_m_adamw"

    print("=" * 60)
    print("  옵티마이저 비교 실험: AdamW (M 티어 1,000장)")
    print(f"  Model: yolo26m.pt (COCO pretrained)")
    print(f"  Optimizer: AdamW, lr0=0.001, 1280px, batch={batch}")
    print("=" * 60)

    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=1280,
        batch=batch,
        device="0",
        project=project,
        name=name,
        exist_ok=True,

        # AdamW
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (동일)
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
        seed=SEED,
        plots=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/{name}/")
    print(f"Best weights: {project}/{name}/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="SGD vs AdamW 비교 실험")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--eval-only", action="store_true", help="평가만 실행")
    args = parser.parse_args()

    if args.eval_only:
        sgd_pt = f"{HOBAN}/hoban_minimal_m/weights/best.pt"
        adamw_pt = f"{HOBAN}/hoban_minimal_m_adamw/weights/best.pt"
        print("평가 명령어:")
        print(f"  python eval_ignore_aware.py --model {sgd_pt}")
        print(f"  python eval_ignore_aware.py --model {adamw_pt}")
    else:
        train_adamw(args.epochs, args.batch)


if __name__ == "__main__":
    main()
