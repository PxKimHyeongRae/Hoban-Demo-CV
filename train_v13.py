#!/usr/bin/env python3
"""
v13 3-Stage 커리큘럼 학습 스크립트

합의안:
- Stage 1 (0~30):  Medium + Large only, 피처 안정화
- Stage 2 (30~100): Small + Medium + Large 전체, small 점진 도입
- Stage 3 (100~150): Small 2× oversample + M + L, 원거리 강화

GPU: RTX 4080 16GB
Model: yolo26m.pt (COCO pretrained)
imgsz: 1280, batch: 4
"""

import os
import sys
import shutil
import random
from pathlib import Path

# ===== Config =====
BASE_DIR = "/home/lay/hoban"
V13_DIR = os.path.join(BASE_DIR, "datasets_v13")
STAGE_DIR = os.path.join(BASE_DIR, "datasets_v13_stages")
MODEL = "yolo26m.pt"
PROJECT = BASE_DIR
NAME = "hoban_v13"
SEED = 42

# 버킷 경계 (area %)
BUCKETS = {
    "small":  (0.3, 1.0),
    "medium": (1.0, 3.0),
    "large":  (3.0, 20.0),
}

# Stage 설정
STAGES = {
    1: {
        "epochs": 30,
        "buckets": ["medium", "large"],
        "oversample": {},
        "multi_scale": 0.0,       # Stage 1은 안정화 목적, multi-scale OFF
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "scale": 0.5,
        "description": "피처 안정화 (Medium + Large)",
    },
    2: {
        "epochs": 70,   # 30~100 (총 70 epochs)
        "buckets": ["small", "medium", "large"],
        "oversample": {},
        "multi_scale": 0.3,       # 960~1440 범위
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "scale": 0.5,
        "description": "Small 점진 도입 (전체 데이터)",
    },
    3: {
        "epochs": 50,   # 100~150 (총 50 epochs)
        "buckets": ["small", "medium", "large"],
        "oversample": {"small": 2},   # small 2배
        "multi_scale": 0.3,
        "mosaic": 1.0,
        "mixup": 0.3,
        "copy_paste": 0.3,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "scale": 0.5,
        "description": "Small 강화 마무리 (Small 2× oversample)",
    },
}


def classify_image(label_path):
    """라벨 파일에서 버킷을 결정 (max bbox area 기준)."""
    with open(label_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return "negative"

    max_area = 0
    for line in lines:
        parts = line.split()
        w, h = float(parts[3]), float(parts[4])
        area = w * h * 100
        max_area = max(max_area, area)

    for bname, (lo, hi) in BUCKETS.items():
        if lo <= max_area < hi:
            return bname

    return "other"


def prepare_stage_data():
    """각 stage별 데이터셋 디렉토리를 심볼릭 링크로 생성."""
    print("=" * 60)
    print("Preparing stage-specific datasets...")
    print("=" * 60)

    # 원본 파일을 버킷별로 분류
    bucket_map = {"train": {}, "valid": {}}

    for split in ["train", "valid"]:
        lbl_dir = os.path.join(V13_DIR, split, "labels")
        img_dir = os.path.join(V13_DIR, split, "images")

        counts = {"small": 0, "medium": 0, "large": 0, "negative": 0, "other": 0}

        for lbl_name in os.listdir(lbl_dir):
            if not lbl_name.endswith(".txt"):
                continue

            lbl_path = os.path.join(lbl_dir, lbl_name)
            img_name = lbl_name.replace(".txt", ".jpg")
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            bucket = classify_image(lbl_path)
            counts[bucket] += 1

            if img_name not in bucket_map[split]:
                bucket_map[split][img_name] = bucket

        print(f"\n  [{split}] Bucket distribution:")
        for b, c in sorted(counts.items()):
            print(f"    {b}: {c:,}")

    # Stage별 디렉토리 생성
    for stage_num, stage_cfg in STAGES.items():
        stage_name = f"stage{stage_num}"
        print(f"\n  Creating {stage_name}...")

        for split in ["train", "valid"]:
            img_out = os.path.join(STAGE_DIR, stage_name, split, "images")
            lbl_out = os.path.join(STAGE_DIR, stage_name, split, "labels")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)

            included_buckets = set(stage_cfg["buckets"])
            # negative는 항상 포함
            included_buckets.add("negative")

            count = 0
            for img_name, bucket in bucket_map[split].items():
                if bucket not in included_buckets:
                    continue

                lbl_name = img_name.replace(".jpg", ".txt")
                src_img = os.path.join(V13_DIR, split, "images", img_name)
                src_lbl = os.path.join(V13_DIR, split, "labels", lbl_name)
                dst_img = os.path.join(img_out, img_name)
                dst_lbl = os.path.join(lbl_out, lbl_name)

                # 심볼릭 링크
                if not os.path.exists(dst_img):
                    os.symlink(os.path.abspath(src_img), dst_img)
                if not os.path.exists(dst_lbl):
                    os.symlink(os.path.abspath(src_lbl), dst_lbl)
                count += 1

                # Oversample: train에서만 적용 (val은 절대 중복하지 않음)
                if split == "train" and bucket in stage_cfg.get("oversample", {}):
                    n_copies = stage_cfg["oversample"][bucket] - 1  # 이미 1개 있으니
                    for c in range(n_copies):
                        dup_img_name = img_name.replace(".jpg", f"_dup{c}.jpg")
                        dup_lbl_name = lbl_name.replace(".txt", f"_dup{c}.txt")
                        dup_img = os.path.join(img_out, dup_img_name)
                        dup_lbl = os.path.join(lbl_out, dup_lbl_name)
                        if not os.path.exists(dup_img):
                            os.symlink(os.path.abspath(src_img), dup_img)
                        if not os.path.exists(dup_lbl):
                            os.symlink(os.path.abspath(src_lbl), dup_lbl)
                        count += 1

            print(f"    {stage_name}/{split}: {count:,} files")

        # data.yaml
        data_yaml_content = (
            f"path: {os.path.join(STAGE_DIR, stage_name)}\n"
            f"train: train/images\n"
            f"val: valid/images\n"
            f"nc: 2\n"
            f"names:\n"
            f"  0: person_with_helmet\n"
            f"  1: person_without_helmet\n"
        )
        yaml_path = os.path.join(STAGE_DIR, stage_name, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(data_yaml_content)

    print("\n  Stage datasets ready!")
    return bucket_map


def train_stage(stage_num, stage_cfg, resume_from=None):
    """한 stage 학습 실행."""
    from ultralytics import YOLO

    stage_name = f"stage{stage_num}"
    data_yaml = os.path.join(STAGE_DIR, stage_name, "data.yaml")

    print(f"\n{'='*60}")
    print(f"Stage {stage_num}: {stage_cfg['description']}")
    print(f"  Epochs: {stage_cfg['epochs']}")
    print(f"  Data: {data_yaml}")
    print(f"  Resume from: {resume_from or MODEL}")
    print(f"{'='*60}")

    # 모델 로딩
    if resume_from and os.path.exists(resume_from):
        model = YOLO(resume_from)
        print(f"  Loaded weights from {resume_from}")
    else:
        model = YOLO(MODEL)
        print(f"  Loaded pretrained {MODEL}")

    # 학습
    results = model.train(
        data=data_yaml,
        epochs=stage_cfg["epochs"],
        imgsz=1280,
        batch=4,
        device="0",
        project=PROJECT,
        name=f"{NAME}_stage{stage_num}",
        exist_ok=True,
        amp=True,
        workers=4,
        patience=50,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001 if stage_num == 1 else 0.0005,
        lrf=0.01,
        warmup_epochs=3.0 if stage_num == 1 else 1.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        weight_decay=0.0005,
        # 증강 (stage별 차등)
        mosaic=stage_cfg["mosaic"],
        mixup=stage_cfg["mixup"],
        copy_paste=stage_cfg["copy_paste"],
        hsv_h=0.015,
        hsv_s=stage_cfg["hsv_s"],
        hsv_v=stage_cfg["hsv_v"],
        scale=stage_cfg["scale"],
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.4,
        multi_scale=stage_cfg["multi_scale"],
        close_mosaic=max(5, stage_cfg["epochs"] // 10),
        # 기타
        seed=SEED,
        deterministic=True,
        plots=True,
        verbose=True,
        save=True,
        save_period=10,
        val=True,
    )

    # best.pt 경로 반환
    best_pt = os.path.join(PROJECT, f"{NAME}_stage{stage_num}", "weights", "best.pt")
    print(f"\n  Stage {stage_num} done! Best: {best_pt}")
    return best_pt, results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="v13 3-Stage Curriculum Training")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run specific stage (1/2/3). 0=all stages sequentially.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from specific weights file.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare stage datasets, don't train.")
    args = parser.parse_args()

    # Stage 데이터 준비
    if os.path.exists(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    prepare_stage_data()

    if args.prepare_only:
        print("\nDatasets prepared. Exiting (--prepare-only).")
        return

    # 학습 실행
    if args.stage > 0:
        # 단일 stage
        stage_cfg = STAGES[args.stage]
        resume = args.resume
        train_stage(args.stage, stage_cfg, resume_from=resume)
    else:
        # 전체 3-stage 순차 실행
        best_pt = None

        for stage_num in [1, 2, 3]:
            stage_cfg = STAGES[stage_num]
            resume = args.resume if stage_num == 1 else best_pt
            best_pt, results = train_stage(stage_num, stage_cfg, resume_from=resume)

        print(f"\n{'='*60}")
        print(f"ALL 3 STAGES COMPLETE!")
        print(f"  Final best model: {best_pt}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
