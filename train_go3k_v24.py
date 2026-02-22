#!/usr/bin/env python3
"""v24 완성형 모델 학습

3-class: person_with_helmet(0), person_without_helmet(1), fallen(2)
yolo26m COCO pt, AdamW, 1280px

사용법:
  python train_go3k_v24.py              # 학습
  python train_go3k_v24.py --resume     # 이어서 학습
  python train_go3k_v24.py --prepare    # 데이터셋 준비만
"""
import argparse, os, sys, subprocess

DATA_YAML = "/home/lay/hoban/datasets_go3k_v24/data.yaml"
MODEL = "yolo26m.pt"  # COCO pretrained
PROJECT = "/home/lay/hoban"
NAME = "hoban_go3k_v24"
RESUME_PATH = f"/home/lay/hoban/{NAME}/weights/last.pt"


def prepare_dataset():
    """데이터셋 준비 스크립트 실행"""
    script = "/home/lay/hoban/prepare_v24_dataset.py"
    if not os.path.exists(script):
        print(f"오류: {script} 없음")
        sys.exit(1)
    subprocess.run([sys.executable, script], check=True)


def train(resume=False):
    from ultralytics import YOLO

    if resume:
        if not os.path.exists(RESUME_PATH):
            print(f"오류: {RESUME_PATH} 없음")
            sys.exit(1)
        print(f"\n학습 재개: {RESUME_PATH}")
        model = YOLO(RESUME_PATH)
        model.train(resume=True)
    else:
        if not os.path.exists(DATA_YAML):
            print(f"오류: {DATA_YAML} 없음")
            print("먼저 python train_go3k_v24.py --prepare 실행")
            sys.exit(1)

        print(f"\nv24 학습 시작")
        print(f"  모델: {MODEL}")
        print(f"  데이터: {DATA_YAML}")
        print(f"  Optimizer: AdamW")
        print(f"  imgsz: 1280, batch: 6")

        model = YOLO(MODEL)
        model.train(
            data=DATA_YAML,
            project=PROJECT,
            name=NAME,
            exist_ok=True,

            # Training
            epochs=100,
            patience=20,
            batch=4,
            imgsz=1280,
            device="0",
            seed=42,
            deterministic=True,

            # Optimizer: AdamW
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.01,
            cos_lr=True,
            warmup_epochs=3.0,

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

            # Output
            save=True,
            save_period=-1,
            plots=True,
            verbose=True,
        )

    print(f"\n학습 완료!")
    print(f"  Best: {PROJECT}/{NAME}/weights/best.pt")
    print(f"  Last: {PROJECT}/{NAME}/weights/last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="이어서 학습")
    parser.add_argument("--prepare", action="store_true", help="데이터셋 준비만")
    args = parser.parse_args()

    if args.prepare:
        prepare_dataset()
    elif args.resume:
        train(resume=True)
    else:
        train()
