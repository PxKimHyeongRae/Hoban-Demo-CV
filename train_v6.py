"""
v6 학습: aihub(헬멧) + WiderPerson(사람) + robo(fallen)
- 4클래스: helmet_o, helmet_x, person, fallen
- GPU 메모리 70% 제한
"""
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO

def main():
    torch.cuda.set_per_process_memory_fraction(0.7, device=0)

    model = YOLO(r"D:\task\hoban\yolo26m.pt")

    results = model.train(
        data=r"D:\task\hoban\datasets_v6\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=15,
        project=r"D:\task\hoban",
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
    freeze_support()
    main()
