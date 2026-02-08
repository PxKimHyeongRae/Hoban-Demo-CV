"""
v8 서브셋 로컬 학습 (30에폭)
- datasets_v8_sub (3K bbox/cls)
- RTX 5070 Ti 16GB
"""
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo26m.pt")
    results = model.train(
        data="datasets_v8_sub/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        project=".",
        name="hoban_v8_sub",
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
    freeze_support()
    main()
