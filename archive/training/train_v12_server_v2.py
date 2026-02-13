"""
v12 서버 학습 v2 - SGD 과적합 방지
- 3클래스: helmet_o / helmet_x / fallen, negative 없음
"""
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo26m.pt")
    results = model.train(
        data="datasets_v12/data.yaml",
        epochs=300,
        imgsz=640,
        batch=24,
        device=0,
        workers=8,
        patience=50,
        project=".",
        name="hoban_v12_v2",
        exist_ok=True,
        verbose=False,
        # optimizer: SGD
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.001,
        warmup_epochs=10,
        warmup_momentum=0.8,
        cos_lr=True,
        # augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.15,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        close_mosaic=100,
        # regularization
        label_smoothing=0.1,
        dropout=0.1,
        # loss
        cls=1.0,
        box=7.5,
        dfl=1.5,
    )


if __name__ == "__main__":
    freeze_support()
    main()

