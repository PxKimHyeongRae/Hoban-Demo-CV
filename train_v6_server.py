"""
v6 서버 학습: aihub(헬멧) + WiderPerson(사람) + robo(fallen)
- 4클래스: helmet_o, helmet_x, person, fallen
- 서버 GPU용 (메모리 제한 없음)
"""
from ultralytics import YOLO

def main():
    model = YOLO("yolo26m.pt")

    results = model.train(
        data="datasets_v6/data.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        device=0,
        workers=8,
        patience=15,
        project=".",
        name="hoban_v6",
        exist_ok=True,
        # 증강
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # 최적화
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        cls=1.0,
        box=7.5,
        dfl=1.5,
    )

    print("Training complete!")

if __name__ == '__main__':
    main()
