"""
v10 서버 학습 v2 - 과적합 방지 버전
변경점 (v1 대비):
1. SGD → AdamW보다 안정적, 과적합 느림
2. lr0=0.01 (SGD 기본) → 충분한 학습력
3. close_mosaic=100 → epoch 200부터 mosaic off, 실제 이미지로 fine-tune
4. warmup_epochs=10 → 느린 워밍업
5. label_smoothing=0.1 → 과확신 방지
6. weight_decay=0.001 → 강화된 정규화
7. dropout=0.1 → 모델 정규화
"""
from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo26m.pt")
    results = model.train(
        data="datasets_v10/data.yaml",
        epochs=300,
        imgsz=640,
        batch=24,
        device=0,
        workers=8,
        patience=50,
        project=".",
        name="hoban_v10_v2",
        exist_ok=True,
        verbose=False,
        # optimizer: SGD (YOLO 기본, AdamW보다 과적합 저항)
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
