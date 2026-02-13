#!/usr/bin/env python3
"""
v15 Tiled Training 학습 스크립트

- 640x640 타일 데이터 (datasets_v15)
- yolo26m.pt COCO pretrained 기반
- negative 타일 포함 (SAHI 추론과 동일한 분포)

실행 (Server):
  python train_v15.py
  python train_v15.py --epochs 100

실행 (Windows):
  python train_v15.py --local
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Windows 로컬 환경")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭")
    parser.add_argument("--batch", type=int, default=24, help="배치 크기 (0=auto)")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.local:
        data_yaml = r"Z:\home\lay\hoban\datasets_v15\data.yaml"
        project = r"Z:\home\lay\hoban"
        base_model = r"Z:\home\lay\hoban\yolo26m.pt"
    else:
        data_yaml = "/home/lay/hoban/datasets_v15/data.yaml"
        project = "/home/lay/hoban"
        base_model = "/home/lay/hoban/yolo26m.pt"

    batch = args.batch if args.batch > 0 else (-1 if not args.local else 16)

    print("=" * 60)
    print("v15 Tiled Training")
    print(f"  Model: {base_model}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {batch}")
    print(f"  imgsz: 640 (tiled images)")
    print("=" * 60)

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=640,
        batch=batch,
        device="0",
        project=project,
        name="hoban_v15",
        exist_ok=True,

        # 학습 설정
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        cos_lr=True,

        # 증강 (타일이므로 가벼운 증강)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.3,
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.1,

        # 기타
        patience=15,
        amp=True,
        workers=4,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/hoban_v15/")


if __name__ == "__main__":
    main()
