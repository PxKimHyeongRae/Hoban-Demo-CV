#!/usr/bin/env python3
"""
go2k v2: v13 stage2 base + v13 8K subsample + go2k ×8 oversample

실행: python train_go2k_v2.py
로컬: python train_go2k_v2.py --local
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Windows 로컬 환경")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.local:
        base_model = r"Z:\home\lay\hoban\hoban_v13_stage2\weights\best.pt"
        data_yaml = r"Z:\home\lay\hoban\datasets_go2k_v2\data.yaml"
        project = r"Z:\home\lay\hoban"
    else:
        base_model = "/home/lay/hoban/hoban_v13_stage2/weights/best.pt"
        data_yaml = "/home/lay/hoban/datasets_go2k_v2/data.yaml"
        project = "/home/lay/hoban"

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device="0",
        project=project,
        name="hoban_go2k_v2",
        exist_ok=True,

        # SGD (안정성 - AdamW 붕괴 방지)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # 증강 (오버샘플 과적합 방지)
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.1,

        # 기타
        patience=20,
        amp=True,
        workers=4,
        seed=42,
        plots=True,
        save=True,
        val=True,
    )

    print(f"Done! Results: {project}/hoban_go2k_v2/")


if __name__ == "__main__":
    main()
