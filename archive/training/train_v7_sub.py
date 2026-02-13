"""
v7 서브셋 학습: aihub + COCO person + robo fallen (4000장, 30에폭)
"""
import torch
from multiprocessing import freeze_support
from ultralytics import YOLO

def main():
    torch.cuda.set_per_process_memory_fraction(0.7, device=0)
    model = YOLO(r"D:\task\hoban\yolo26m.pt")
    results = model.train(
        data=r"D:\task\hoban\datasets_v7_sub\data.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        project=r"D:\task\hoban",
        name="hoban_v7_sub",
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
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
