"""
v9 서버 학습 (200에폭)
- datasets_v9 (30K bbox/cls, 필터링된 고품질 데이터)
- 서버: lay@pluxity:~/hoban, conda llm env
- batch=-1 (auto), workers=8
"""
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo26m.pt")
    results = model.train(
        data="datasets_v9/data.yaml",
        epochs=300,
        imgsz=640,
        batch=24,
        device=0,
        workers=8,
        patience=30,
        project=".",
        name="hoban_v9",
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
