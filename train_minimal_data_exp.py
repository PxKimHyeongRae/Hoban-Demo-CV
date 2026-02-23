#!/usr/bin/env python3
"""
최소 데이터 실험: 현장(cam) 데이터만으로 helmet_on/off 최적 판별

가설: 고품질 현장 데이터만 사용하면 소량으로도 높은 F1 달성 가능.
      helmet_off(소수 클래스) 이미지를 우선 배치하여 판별력 극대화.

데이터 선택 우선순위:
  1. helmet_off 포함 이미지 (소수 클래스, 판별 핵심)
  2. helmet_on 전용 이미지 (양성 대조)
  3. 빈 이미지/negative (FP 억제)

실험 티어:
  S:  500장  (helmet_off 중심, 최소 실험)
  M:  1000장 (균형 확보)
  L:  2000장 (helmet_off 전량 + 보충)
  XL: 4038장 (v23 전체, 현장 100%)

사용법:
  python train_minimal_data_exp.py --prepare          # 전체 티어 데이터셋 준비
  python train_minimal_data_exp.py --tier S            # S 티어 학습
  python train_minimal_data_exp.py --tier M            # M 티어 학습
  python train_minimal_data_exp.py --tier L            # L 티어 학습
  python train_minimal_data_exp.py --tier XL           # XL 티어 학습 (= v23 전체)
  python train_minimal_data_exp.py --tier S --resume   # 이어서 학습
"""
import argparse
import os
import random
import shutil
import yaml

HOBAN = "/home/lay/hoban"
V23_DATASET = f"{HOBAN}/datasets_go3k_v23"
MODEL = f"{HOBAN}/yolo26m.pt"  # COCO pretrained
SEED = 42

# 티어별 구성 (총 이미지 수, helmet_off 비율 목표)
TIERS = {
    "S":  {"total": 500,  "desc": "500장 (최소, helmet_off 중심)"},
    "M":  {"total": 1000, "desc": "1,000장 (균형 확보)"},
    "L":  {"total": 2000, "desc": "2,000장 (helmet_off 전량 + 보충)"},
    "XL": {"total": 4038, "desc": "4,038장 (v23 전체, 현장 100%)"},
}


def analyze_v23():
    """v23 데이터를 이미지별 클래스로 분류"""
    labels_dir = f"{V23_DATASET}/train/labels"
    images_dir = f"{V23_DATASET}/train/images"

    has_cls0_only = []  # helmet_on만 있는 이미지
    has_cls1_only = []  # helmet_off만 있는 이미지
    has_both = []       # 두 클래스 모두 있는 이미지
    negatives = []      # 빈 라벨 (배경)

    for f in sorted(os.listdir(labels_dir)):
        if not f.endswith(".txt"):
            continue
        img = f.replace(".txt", ".jpg")
        if not os.path.exists(os.path.join(images_dir, img)):
            continue

        with open(os.path.join(labels_dir, f)) as fh:
            lines = [l.strip() for l in fh if l.strip()]

        if not lines:
            negatives.append(img)
            continue

        classes = set(l.split()[0] for l in lines)
        if "1" in classes and "0" in classes:
            has_both.append(img)
        elif "1" in classes:
            has_cls1_only.append(img)
        elif "0" in classes:
            has_cls0_only.append(img)

    return has_cls0_only, has_cls1_only, has_both, negatives


def build_tier_selection(tier_name, has_cls0, has_cls1, has_both, negatives):
    """티어별 이미지 선택 (helmet_off 우선)"""
    rng = random.Random(SEED)
    target = TIERS[tier_name]["total"]

    # 각 그룹을 섞기 (재현성 위해 seed 고정)
    cls0 = list(has_cls0)
    cls1 = list(has_cls1)
    both = list(has_both)
    negs = list(negatives)
    rng.shuffle(cls0)
    rng.shuffle(cls1)
    rng.shuffle(both)
    rng.shuffle(negs)

    selected = []

    if tier_name == "XL":
        # 전체 사용
        return cls0 + cls1 + both + negs

    # 1단계: helmet_off 포함 이미지 우선 (cls1 + both)
    off_images = cls1 + both  # 총 944장
    if tier_name == "S":
        # S: 500장 중 helmet_off 300장 (60%), on 150장, neg 50장
        n_off = min(300, len(off_images))
        n_on = min(150, len(cls0))
        n_neg = min(50, len(negs))
    elif tier_name == "M":
        # M: 1000장 중 helmet_off 500장 (50%), on 350장, neg 150장
        n_off = min(500, len(off_images))
        n_on = min(350, len(cls0))
        n_neg = min(150, len(negs))
    elif tier_name == "L":
        # L: 2000장 중 helmet_off 전량 944장 (47%), on 750장, neg 306장
        n_off = len(off_images)  # 전량
        n_on = min(750, len(cls0))
        n_neg = target - n_off - n_on

    selected.extend(off_images[:n_off])
    selected.extend(cls0[:n_on])
    selected.extend(negs[:n_neg])

    # 목표 수에 맞추기 (부족하면 나머지에서 채움)
    remaining_pool = []
    if n_off < len(off_images):
        remaining_pool.extend(off_images[n_off:])
    if n_on < len(cls0):
        remaining_pool.extend(cls0[n_on:])
    if n_neg < len(negs):
        remaining_pool.extend(negs[n_neg:])

    rng.shuffle(remaining_pool)
    while len(selected) < target and remaining_pool:
        selected.append(remaining_pool.pop())

    return selected[:target]


def prepare_tier(tier_name, selected_images):
    """선택된 이미지로 데이터셋 디렉터리 구성"""
    ds_path = f"{HOBAN}/datasets_minimal_{tier_name.lower()}"
    train_img = f"{ds_path}/train/images"
    train_lbl = f"{ds_path}/train/labels"
    val_img = f"{ds_path}/valid/images"
    val_lbl = f"{ds_path}/valid/labels"

    # 기존 디렉터리 정리
    if os.path.exists(ds_path):
        shutil.rmtree(ds_path)

    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # train: symlink
    v23_img = f"{V23_DATASET}/train/images"
    v23_lbl = f"{V23_DATASET}/train/labels"

    for img_name in selected_images:
        lbl_name = img_name.replace(".jpg", ".txt")

        # 이미지 (실제 경로 추적)
        src_img = os.path.join(v23_img, img_name)
        if os.path.islink(src_img):
            src_img = os.path.realpath(src_img)
        os.symlink(src_img, os.path.join(train_img, img_name))

        # 라벨
        src_lbl = os.path.join(v23_lbl, lbl_name)
        if os.path.exists(src_lbl):
            if os.path.islink(src_lbl):
                src_lbl = os.path.realpath(src_lbl)
            os.symlink(src_lbl, os.path.join(train_lbl, lbl_name))
        else:
            open(os.path.join(train_lbl, lbl_name), "w").close()

    # val: v23 val 그대로 (641장, 100% 현장)
    for src in sorted(os.listdir(f"{V23_DATASET}/valid/images")):
        if not src.endswith(".jpg"):
            continue
        real = os.path.realpath(os.path.join(f"{V23_DATASET}/valid/images", src))
        os.symlink(real, os.path.join(val_img, src))
    for src in sorted(os.listdir(f"{V23_DATASET}/valid/labels")):
        if not src.endswith(".txt"):
            continue
        real = os.path.realpath(os.path.join(f"{V23_DATASET}/valid/labels", src))
        os.symlink(real, os.path.join(val_lbl, src))

    # data.yaml
    yaml_path = f"{ds_path}/data.yaml"
    data = {
        "path": ds_path,
        "train": "train/images",
        "val": "valid/images",
        "nc": 2,
        "names": {0: "person_with_helmet", 1: "person_without_helmet"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    return ds_path


def count_stats(ds_path):
    """데이터셋 통계"""
    train_img = f"{ds_path}/train/images"
    train_lbl = f"{ds_path}/train/labels"
    val_img = f"{ds_path}/valid/images"

    n_train = len([f for f in os.listdir(train_img) if f.endswith(".jpg")])
    n_val = len([f for f in os.listdir(val_img) if f.endswith(".jpg")])

    cls0_bbox, cls1_bbox = 0, 0
    n_with_off, n_with_on, n_empty = 0, 0, 0

    for f in os.listdir(train_lbl):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(train_lbl, f)) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if not lines:
            n_empty += 1
            continue
        classes = set()
        for l in lines:
            c = l.split()[0]
            classes.add(c)
            if c == "0":
                cls0_bbox += 1
            elif c == "1":
                cls1_bbox += 1
        if "1" in classes:
            n_with_off += 1
        if "0" in classes:
            n_with_on += 1

    return {
        "train": n_train,
        "val": n_val,
        "cls0_bbox": cls0_bbox,
        "cls1_bbox": cls1_bbox,
        "img_with_on": n_with_on,
        "img_with_off": n_with_off,
        "img_empty": n_empty,
        "off_ratio": n_with_off / n_train * 100 if n_train > 0 else 0,
    }


def prepare_all():
    """전체 티어 데이터셋 준비"""
    print("=" * 60)
    print("  최소 데이터 실험: 데이터셋 준비")
    print("=" * 60)

    # v23 분석
    print("\n[분석] v23 현장 데이터 분류 중...")
    cls0, cls1, both, negs = analyze_v23()
    print(f"  helmet_on 전용: {len(cls0)}장")
    print(f"  helmet_off 전용: {len(cls1)}장")
    print(f"  양쪽 클래스: {len(both)}장")
    print(f"  빈 이미지(neg): {len(negs)}장")
    print(f"  helmet_off 포함 합계: {len(cls1)+len(both)}장")

    for tier_name in ["S", "M", "L", "XL"]:
        print(f"\n{'─'*60}")
        print(f"[{tier_name}] {TIERS[tier_name]['desc']}")

        selected = build_tier_selection(tier_name, cls0, cls1, both, negs)
        ds_path = prepare_tier(tier_name, selected)
        stats = count_stats(ds_path)

        print(f"  경로: {ds_path}")
        print(f"  Train: {stats['train']}장")
        print(f"    - helmet_on bbox: {stats['cls0_bbox']}")
        print(f"    - helmet_off bbox: {stats['cls1_bbox']}")
        print(f"    - helmet_off 이미지: {stats['img_with_off']}장 ({stats['off_ratio']:.1f}%)")
        print(f"    - 빈 이미지: {stats['img_empty']}장")
        print(f"  Val: {stats['val']}장")

    print(f"\n{'='*60}")
    print("  준비 완료! 학습 명령:")
    print("    python train_minimal_data_exp.py --tier S")
    print("    python train_minimal_data_exp.py --tier M")
    print("    python train_minimal_data_exp.py --tier L")
    print("    python train_minimal_data_exp.py --tier XL")
    print(f"{'='*60}")


def train(tier_name, epochs=100, batch=6, resume=False):
    """지정 티어 학습"""
    ds_path = f"{HOBAN}/datasets_minimal_{tier_name.lower()}"
    data_yaml = f"{ds_path}/data.yaml"
    project = HOBAN
    name = f"hoban_minimal_{tier_name.lower()}"

    if not os.path.exists(data_yaml):
        print(f"데이터셋 없음. 먼저 --prepare 실행:")
        print(f"  python train_minimal_data_exp.py --prepare")
        return

    from ultralytics import YOLO

    if resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        if not os.path.exists(ckpt):
            print(f"체크포인트 없음: {ckpt}")
            return
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    stats = count_stats(ds_path)
    print(f"{'='*60}")
    print(f"  최소 데이터 실험 [{tier_name}]: {TIERS[tier_name]['desc']}")
    print(f"  Train: {stats['train']}장 (OFF 이미지: {stats['img_with_off']})")
    print(f"  Val: {stats['val']}장")
    print(f"  Model: yolo26m.pt (COCO pretrained)")
    print(f"  Optimizer: SGD, lr0=0.005, 1280px, batch={batch}")
    print(f"{'='*60}\n")

    model = YOLO(MODEL)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=1280,
        batch=batch,
        device="0",
        project=project,
        name=name,
        exist_ok=True,

        # SGD (검증된 설정)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (v17 표준)
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
        seed=SEED,
        plots=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/{name}/")
    print(f"Best weights: {project}/{name}/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="최소 데이터 실험 (현장 데이터)")
    parser.add_argument("--prepare", action="store_true", help="전체 티어 데이터셋 준비")
    parser.add_argument("--tier", type=str, choices=["S", "M", "L", "XL"],
                        help="학습할 티어 (S=500, M=1000, L=2000, XL=4038)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare_all()
    elif args.tier:
        train(args.tier, args.epochs, args.batch, args.resume)
    else:
        parser.print_help()
        print("\n예시:")
        print("  python train_minimal_data_exp.py --prepare        # 데이터셋 준비")
        print("  python train_minimal_data_exp.py --tier S         # 500장 학습")
        print("  python train_minimal_data_exp.py --tier M         # 1000장 학습")


if __name__ == "__main__":
    main()
