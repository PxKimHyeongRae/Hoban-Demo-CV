#!/usr/bin/env python3
"""
3k CVAT 검수 데이터로 go2k_v3 파인튜닝

- Base: go2k_v3 best.pt (1280px 학습, mAP50=0.954)
- Data: 3k_finetune (2,564 train / 641 val, CVAT 검수 완료)
- Strategy: 낮은 lr + freeze 없이 전체 파인튜닝
"""
from ultralytics import YOLO

model = YOLO("/home/lay/hoban/hoban_go2k_v3/weights/best.pt")

model.train(
    data="/home/lay/hoban/datasets/3k_finetune/dataset.yaml",
    epochs=100,
    patience=20,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.1,
    degrees=5,
    erasing=0.1,
    project="/home/lay/hoban",
    name="hoban_3k_finetune",
    exist_ok=True,
    device=0,
    workers=8,
)
