#!/usr/bin/env python3
"""
go3k v17: COCO pretrained (yolo26m) + 1280px 학습

A5(COCO pt, F1=0.898) + A4(1280px, F1=0.892) 결합
시작 가중치: yolo26m.pt (COCO pretrained)

실행: python train_go3k_v17.py
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train go3k v17")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    base_model = "yolo26m.pt"
    data_yaml = "/home/lay/hoban/datasets_go3k_v16/data.yaml"
    project = "/home/lay/hoban"
    name = "hoban_go3k_v17"

    if args.resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print(f"=== go3k v17: COCO pt + 1280px ===")
    print(f"  Base: {base_model}")
    print(f"  Data: {data_yaml}")
    print(f"  imgsz: 1280, batch: {args.batch}")
    print(f"  epochs: {args.epochs}")
    print()

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=1280,
        batch=args.batch,
        device="0",
        project=project,
        name=name,
        exist_ok=True,    

        # SGD
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.15,
        close_mosaic=10,

        # Early stopping
        patience=20,
        amp=True,
        workers=4,
        seed=42,
        plots=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/{name}/")


if __name__ == "__main__":
    main()
