#!/usr/bin/env python3
"""
v13 Crop 실험 학습 스크립트

- crop된 640x640 이미지로 학습
- v13 stage2 best.pt에서 파인튜닝
- 실험용: 30 epochs, 빠른 검증

실행 (Windows):
  python train_v13_crop.py

실행 (Server):
  python train_v13_crop.py --server
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="서버 환경 (Linux)")
    parser.add_argument("--epochs", type=int, default=30, help="학습 에폭 (기본: 30)")
    parser.add_argument("--batch", type=int, default=0, help="배치 크기 (0=auto)")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.server:
        data_yaml = "/home/lay/hoban/datasets_v13_crop/data.yaml"
        project = "/home/lay/hoban"
        # 서버용 data.yaml 덮어쓰기
        yaml_content = (
            "path: /home/lay/hoban/datasets_v13_crop\n"
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
        data_yaml = r"Z:\home\lay\hoban\datasets_v13_crop\data.yaml"
        project = r"Z:\home\lay\hoban"

    # COCO pretrained base (v13과 동일 출발점)
    if args.server:
        base_model = "/home/lay/hoban/yolo26m.pt"
    else:
        base_model = r"Z:\home\lay\hoban\yolo26m.pt"

    batch = args.batch if args.batch > 0 else (-1 if args.server else 16)

    print("=" * 60)
    print("v13 Crop Experiment Training")
    print(f"  Base model: {base_model}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {batch}")
    print(f"  imgsz: 640 (crop된 이미지)")
    print("=" * 60)

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=640,
        batch=batch,
        device="0",
        project=project,
        name="hoban_v13_crop",
        exist_ok=True,

        # COCO pretrained 기반 학습
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        cos_lr=True,

        # 증강: 가벼운 수준 (이미 crop으로 변환됨)
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
        erasing=0.0,

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

    print("\nTraining complete!")
    print(f"Results: {project}/hoban_v13_crop/")


if __name__ == "__main__":
    main()
