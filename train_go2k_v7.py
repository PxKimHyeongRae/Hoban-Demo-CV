#!/usr/bin/env python3
"""
go2k_v7 학습: 근본 문제 해결 후 재학습
- Train/Eval 완전 분리 (data leakage 0%)
- 실제 augmentation 적용 (byte-identical copy 제거)
- v13 bbox 크기 필터링 (eval 도메인 매칭)
- v2 최적 하이퍼파라미터 유지
"""
from ultralytics import YOLO

model = YOLO("yolo26m.pt")

results = model.train(
    data="/home/lay/hoban/datasets_go2k_v7/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    device=0,
    project="/home/lay/hoban",
    name="hoban_go2k_v7",
    exist_ok=True,

    # v2 최적 하이퍼파라미터 (Phase A에서 검증됨)
    optimizer="SGD",
    lr0=0.005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,

    # Augmentation
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    erasing=0.0,

    # Other
    workers=8,
    patience=20,
    save_period=5,
    val=True,
    plots=True,
)

print("\n학습 완료!")
print(f"Best weights: /home/lay/hoban/hoban_go2k_v7/weights/best.pt")
