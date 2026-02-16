#!/usr/bin/env python3
"""
go3k v16: Clean 데이터셋 재학습

핵심 변경 (vs v2~v7):
  - Train: 3k_finetune train 2,564장 + v13 8K (byte copy 없음)
  - Val: 3k_finetune val 641장 (leakage 없음)
  - 라벨: 3k 보수적 라벨로 통일 (확실한 것만)
  - augmentation: YOLO 내장 (mosaic, mixup, copy_paste)

실행 예시:
  python train_go3k_v16.py                    # 기본 (640px)
  python train_go3k_v16.py --imgsz 1280       # 1280px (소형객체 강화)
  python train_go3k_v16.py --no-v13           # v13 없이 CCTV만
  python train_go3k_v16.py --from-scratch     # COCO pretrained에서 시작
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Train go3k v16")
    parser.add_argument("--imgsz", type=int, default=640,
                        choices=[640, 1280], help="학습 해상도")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=None,
                        help="배치 크기 (default: 640→16, 1280→4)")
    parser.add_argument("--no-v13", action="store_true",
                        help="v13 제외 데이터셋 사용")
    parser.add_argument("--from-scratch", action="store_true",
                        help="v13_stage2 대신 COCO pretrained 사용")
    parser.add_argument("--resume", action="store_true",
                        help="이전 학습 이어서")
    args = parser.parse_args()

    from ultralytics import YOLO

    # Batch size
    if args.batch is None:
        args.batch = 4 if args.imgsz == 1280 else 24

    # Model
    if args.from_scratch:
        base_model = "yolov8m.pt"
        suffix = "_coco"
    else:
        base_model = "/home/lay/hoban/hoban_v13_stage2/weights/best.pt"
        suffix = ""

    # Dataset
    data_yaml = "/home/lay/hoban/datasets_go3k_v16/data.yaml"

    # Name
    name = f"hoban_go3k_v16_{args.imgsz}{suffix}"
    project = "/home/lay/hoban"

    if args.resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print(f"=== go3k v16 Training ===")
    print(f"  Base: {base_model}")
    print(f"  Data: {data_yaml}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  batch: {args.batch}")
    print(f"  epochs: {args.epochs}")
    print(f"  name: {name}")
    print()

    model = YOLO(base_model)

    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="0",
        project=project,
        name=name,
        exist_ok=True,

        # SGD (안정적, 과적합 방지)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (실제 augmentation, byte copy X)
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

        # Early stopping & 저장
        patience=20,
        amp=True,
        workers=4,
        seed=42,
        plots=True,
        save=True,
        val=True,
    )

    model.train(**train_kwargs)
    print(f"\nDone! Results: {project}/{name}/")


if __name__ == "__main__":
    main()
