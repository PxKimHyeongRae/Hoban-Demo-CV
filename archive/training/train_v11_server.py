"""
v11 서버 학습 - 3클래스 (person 제거)
- cls 0: helmet_o, cls 1: helmet_x, cls 2: fallen
"""
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo26m.pt")
    results = model.train(
        data="dataset_v11/data.yaml",
        epochs=300,
        imgsz=640,
        batch=24,
        device=0,
        workers=8,
        patience=30,
        project=".",
        name="hoban_v11",
        exist_ok=True,
        verbose=False,
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
    freeze_support()
    main()
