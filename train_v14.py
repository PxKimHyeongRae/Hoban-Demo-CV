#!/usr/bin/env python3
"""
v14 Crop 학습 스크립트

- 60k crop 640x640 이미지 (helmet 30k + no_helmet 30k)
- yolo26m.pt COCO pretrained 기반
- negative 없음, 클래스 밸런스 1:1

실행 (Windows):
  python train_v14.py
  python train_v14.py --epochs 50

실행 (Server):
  python train_v14.py --server
  python train_v14.py --server --epochs 50
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="서버 환경 (Linux)")
    parser.add_argument("--epochs", type=int, default=150, help="학습 에폭 (기본: 30)")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기 (0=auto)")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.server:
        data_yaml = "/home/lay/hoban/datasets_v14/data.yaml"
        project = "/home/lay/hoban"
        base_model = "/home/lay/hoban/yolo26m.pt"
        # 서버용 data.yaml 덮어쓰기
        yaml_content = (
            "path: /home/lay/hoban/datasets_v14\n"
            "train: train/images\n"
            "val: valid/images\n"
            "nc: 2\n"
            "names:\n"
            "  0: person_with_helmet\n"
            "  1: person_without_helmet\n"
        )
        with open(data_yaml, 'w') as f:
            f.write(yaml_content)
    else:
        data_yaml = r"Z:\home\lay\hoban\datasets_v14\data.yaml"
        project = r"Z:\home\lay\hoban"
        base_model = r"Z:\home\lay\hoban\yolo26m.pt"

    batch = args.batch if args.batch > 0 else (-1 if args.server else 16)

    print("=" * 60)
    print("v14 Crop Training")
    print(f"  Model: {base_model}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {batch}")
    print(f"  imgsz: 640 (crop images)")
    print("=" * 60)

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=640,
        batch=batch,
        device="0",
        project=project,
        name="hoban_v14",
        exist_ok=True,

        # 학습 설정
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        cos_lr=True,

        # 증강
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
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
        deterministic=True,
        plots=True,
        verbose=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/hoban_v14/")


if __name__ == "__main__":
    main()
