#!/usr/bin/env python3
"""
Quick baseline: v2와 동일한 하이퍼파라미터, 축소 데이터셋
- 3,916 train (v13 2K + go2k ×4)
- 20 epochs, ~18분 예상
- 이후 실험들의 비교 기준점
"""
from ultralytics import YOLO

model = YOLO("/home/lay/hoban/hoban_v13_stage2/weights/best.pt")

model.train(
    data="/home/lay/hoban/datasets_go2k_quick/data.yaml",
    epochs=20,
    patience=20,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    cos_lr=True,
    # 증강 (v2 원본 설정)
    mosaic=1.0,
    close_mosaic=0,
    mixup=0.1,
    copy_paste=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    translate=0.1,
    degrees=5.0,
    fliplr=0.5,
    erasing=0.0,
    # Loss (v2 원본)
    box=7.5,
    dfl=1.5,
    # 기타
    project="/home/lay/hoban",
    name="hoban_quick_baseline",
    exist_ok=True,
    device=0,
    workers=8,
    seed=42,
    amp=True,
)
