#!/usr/bin/env python3
"""
go2k v4: v13 8K subsample + 3k CVAT 검수 데이터 x3 oversample
- Base: v13 stage2 best.pt
- Data: 15,692 train / 1,641 val
- v2 전략 (v13 혼합) + copy_paste 추가
"""
from ultralytics import YOLO

model = YOLO("/home/lay/hoban/hoban_v13_stage2/weights/best.pt")

model.train(
    data="/home/lay/hoban/datasets_go2k_v4/data.yaml",
    epochs=100,
    patience=20,
    imgsz=640,
    batch=24,
    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    cos_lr=True,
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
    erasing=0.1,
    project="/home/lay/hoban",
    name="hoban_go2k_v4",
    exist_ok=True,
    device=0,
    workers=8,
    seed=42,
    amp=True,
)
