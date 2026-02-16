#!/usr/bin/env python3
"""
go2k v6: v2 데이터 + v5 학습 최적화

핵심 가설: v2 데이터(v13 8K + go2k 479×8)가 최적인데,
학습 하이퍼파라미터를 개선하면 더 나아질 수 있는가?

v2 대비 변경점:
  - close_mosaic=15: 마지막 15에폭 mosaic off
  - erasing=0.3: 가림 증강 강화
  - cls=2.0: 분류 loss 가중치 증가
  - copy_paste=0.15: 추가
  - albumentations: 자동 적용
"""
from ultralytics import YOLO

model = YOLO("/home/lay/hoban/hoban_v13_stage2/weights/best.pt")

model.train(
    data="/home/lay/hoban/datasets_go2k_v2/data.yaml",
    epochs=40,
    patience=40,
    imgsz=640,
    batch=16,
    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    cos_lr=True,
    # 증강 (v5 최적화 적용)
    mosaic=1.0,
    close_mosaic=15,
    mixup=0.1,
    copy_paste=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    translate=0.1,
    degrees=5.0,
    fliplr=0.5,
    erasing=0.3,
    # Loss
    cls=2.0,
    box=7.5,
    dfl=1.5,
    # 기타
    project="/home/lay/hoban",
    name="hoban_go2k_v6",
    exist_ok=True,
    device=0,
    workers=8,
    seed=42,
    amp=True,
)
