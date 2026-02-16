#!/usr/bin/env python3
"""A4: strong augmentation - scale=0.9, mixup=0.2, degrees=10"""
from ultralytics import YOLO
model = YOLO("/home/lay/hoban/hoban_v13_stage2/weights/best.pt")
model.train(
    data="/home/lay/hoban/datasets_go2k_quick/data.yaml",
    epochs=20, patience=20, imgsz=640, batch=16,
    optimizer="SGD", lr0=0.005, lrf=0.01, momentum=0.937,
    warmup_epochs=3.0, weight_decay=0.0005, cos_lr=True,
    mosaic=1.0, close_mosaic=10, mixup=0.2,  # KEY CHANGE
    copy_paste=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    scale=0.9, translate=0.2, degrees=10.0,  # KEY CHANGE
    fliplr=0.5, erasing=0.0,
    box=7.5, dfl=1.5,
    project="/home/lay/hoban", name="hoban_quick_a4", exist_ok=True,
    device=0, workers=8, seed=42, amp=True,
)
