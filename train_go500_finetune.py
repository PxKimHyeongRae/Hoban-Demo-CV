#!/usr/bin/env python3
"""
go500 Fine-tuning: v13 stage2 best.pt 기반 CCTV 도메인 적응

실행: python train_go500_finetune.py
"""

from ultralytics import YOLO

DATA = "/home/lay/hoban/datasets/go500_yolo/data.yaml"
BASE = "/home/lay/hoban/hoban_v13_stage2/weights/best.pt"
PROJECT = "/home/lay/hoban"

print("=" * 60)
print("go500 Fine-tune (v13 stage2 → CCTV)")
print(f"  Base: {BASE}")
print(f"  Data: {DATA}")
print("=" * 60)

model = YOLO(BASE)

model.train(
    data=DATA,
    epochs=100,
    imgsz=1280,
    batch=-1,
    device="0",
    project=PROJECT,
    name="hoban_go500_finetune",
    exist_ok=True,

    optimizer="AdamW",
    lr0=0.0002,
    lrf=0.01,
    warmup_epochs=5.0,
    weight_decay=0.0005,
    cos_lr=True,

    # 소량 데이터 → 강한 증강
    mosaic=1.0,
    mixup=0.3,
    copy_paste=0.2,
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    translate=0.15,
    degrees=10.0,
    fliplr=0.5,
    erasing=0.3,

    patience=30,
    amp=True,
    workers=4,
    seed=42,
    plots=True,
    val=True,
)
