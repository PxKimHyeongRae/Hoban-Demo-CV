#!/usr/bin/env python3
"""
go2k v5: v13 8K + 3k CVAT 검수 x3 oversample + 학습 최적화

개선점 (v2 대비):
  1. close_mosaic=15: 마지막 15에폭 mosaic 비활성화 → 실제 분포 수렴
  2. erasing=0.3: 가림 증강 강화 (0.1→0.3) → 건설현장 가림 대응
  3. cls=2.0: 분류 loss 가중치 증가 → helmet_on/off 분류 정밀도 향상
  4. albumentations: CCTV 환경 시뮬레이션 (자동 감지)

- Base: v13 stage2 best.pt
- Data: 15,692 train / 1,641 val (3k 비중 49%)
"""
from ultralytics import YOLO

model = YOLO("/home/lay/hoban/hoban_v13_stage2/weights/best.pt")

model.train(
    data="/home/lay/hoban/datasets_go2k_v5/data.yaml",
    epochs=100,
    patience=25,
    imgsz=640,
    batch=24,
    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    cos_lr=True,
    # 증강
    mosaic=1.0,
    close_mosaic=15,       # [NEW] 마지막 15에폭 mosaic off
    mixup=0.1,
    copy_paste=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    translate=0.1,
    degrees=5.0,
    fliplr=0.5,
    erasing=0.3,           # [NEW] 0.1→0.3 가림 증강 강화
    # Loss 가중치
    cls=2.0,               # [NEW] 분류 loss 가중치 증가 (기본 0.5)
    box=7.5,
    dfl=1.5,
    # 기타
    project="/home/lay/hoban",
    name="hoban_go2k_v5",
    exist_ok=True,
    device=0,
    workers=8,
    seed=42,
    amp=True,
)
