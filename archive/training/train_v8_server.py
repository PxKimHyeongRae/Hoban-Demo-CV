"""
v8 서버 학습 스크립트
- datasets_v8 + yolo26m.pt
- batch=-1 (auto), workers=8, 상대경로
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolo26m.pt")
    results = model.train(
        data="datasets_v8/data.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        device=0,
        workers=8,
        patience=15,
        project=".",
        name="hoban_v8",
        exist_ok=True,
        # augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # loss
        cls=1.0,
        box=7.5,
        dfl=1.5,
    )


if __name__ == "__main__":
    main()
