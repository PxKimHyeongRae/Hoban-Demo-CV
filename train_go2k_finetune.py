#!/usr/bin/env python3
"""
go2k fine-tune: go500 fine-tune 모델 → go2k 수동 라벨링 데이터로 추가 학습

실행: python train_go2k_finetune.py
로컬: python train_go2k_finetune.py --local
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
        base_model = r"Z:\home\lay\hoban\hoban_go500_finetune\weights\best.pt"
        data_yaml = r"Z:\home\lay\hoban\datasets_go2k\data.yaml"
        project = r"Z:\home\lay\hoban"
    else:
        base_model = "/home/lay/hoban/hoban_go500_finetune/weights/best.pt"
        data_yaml = "/home/lay/hoban/datasets_go2k/data.yaml"
        project = "/home/lay/hoban"

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device="0",
        project=project,
        name="hoban_go2k_finetune_adamw",
        exist_ok=True,

        # AdamW
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # 증강 (소규모 데이터 → 적당한 증강)
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

    print(f"Done! Results: {project}/hoban_go2k_finetune/")


if __name__ == "__main__":
    main()
