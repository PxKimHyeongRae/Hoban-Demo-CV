#!/usr/bin/env python3
"""v29 3-class: 랜덤 샘플링 + close_mosaic=0

v26 대비 변경:
  - fallen area 필터 제거: v24 + unified에서 랜덤 추출 (~1,200장)
  - close_mosaic=0: mosaic 100 epoch 유지 (v26~v28에서 -1%p 역효과 확인)
  - patience=25: close_mosaic 제거에 맞춰 여유 확대
  - v30(stratified)과의 비교 baseline

클래스:
  0: person_with_helmet
  1: person_without_helmet
  2: fallen

사용법:
  python train_go3k_v29.py --prepare   # 데이터셋 빌드
  python train_go3k_v29.py --train     # 학습
  python train_go3k_v29.py --resume    # 이어서 학습
"""
import os
import random
from pathlib import Path

random.seed(42)

HOBAN = "/home/lay/hoban"
OUT_DIR = f"{HOBAN}/datasets_go3k_v29"
MODEL = f"{HOBAN}/yolo26m.pt"

# 소스: 헬멧
L_TIER_TRAIN_IMG = f"{HOBAN}/datasets_minimal_l/train/images"
L_TIER_TRAIN_LBL = f"{HOBAN}/datasets_minimal_l/train/labels"
L_TIER_VAL_IMG = f"{HOBAN}/datasets_minimal_l/valid/images"
L_TIER_VAL_LBL = f"{HOBAN}/datasets_minimal_l/valid/labels"

# 소스: fallen (v24, Roboflow 기반)
V24_TRAIN_IMG = f"{HOBAN}/datasets_go3k_v24/train/images"
V24_TRAIN_LBL = f"{HOBAN}/datasets_go3k_v24/train/labels"
V24_VAL_IMG = f"{HOBAN}/datasets_go3k_v24/valid/images"
V24_VAL_LBL = f"{HOBAN}/datasets_go3k_v24/valid/labels"

# 소스: fallen (unified_safety_all, class 4 → class 2)
UNIFIED_TRAIN_IMG = "/data/unified_safety_all/train/images"
UNIFIED_TRAIN_LBL = "/data/unified_safety_all/train/labels"

# 목표: fallen ~1,200장 (v24 전체 + unified 보충)
TARGET_FALLEN = 1200


def parse_labels(label_path):
    """라벨 파일 파싱"""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append((cls, cx, cy, w, h))
    return bboxes


def extract_v24_fallen_all():
    """v24에서 fallen 전체 추출 (area 필터 없음)"""
    fallen = []
    for lbl_name in os.listdir(V24_TRAIN_LBL):
        if not lbl_name.startswith("fallen_"):
            continue
        lbl_path = os.path.join(V24_TRAIN_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)
        fallen_bboxes = [b for b in bboxes if b[0] == 2]
        if not fallen_bboxes:
            continue
        img_name = lbl_name.replace(".txt", ".jpg")
        if os.path.exists(os.path.join(V24_TRAIN_IMG, img_name)):
            avg_area = sum(b[3]*b[4] for b in fallen_bboxes) / len(fallen_bboxes)
            fallen.append({
                'img': img_name, 'lbl': lbl_name,
                'avg_area': avg_area, 'n_bbox': len(fallen_bboxes),
                'source': 'v24',
                'bboxes': bboxes,
            })
    return fallen


def extract_unified_fallen_random(max_images=500):
    """unified_safety_all에서 fallen(class 4) 랜덤 추출 (area 필터 없음)"""
    fallen = []
    label_files = [f for f in os.listdir(UNIFIED_TRAIN_LBL) if f.endswith('.txt')]
    random.shuffle(label_files)

    scanned = 0
    for lbl_name in label_files:
        lbl_path = os.path.join(UNIFIED_TRAIN_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)

        fallen_bboxes = [b for b in bboxes if b[0] == 4]
        if not fallen_bboxes:
            continue

        img_name = lbl_name.replace(".txt", ".jpg")
        img_path = os.path.join(UNIFIED_TRAIN_IMG, img_name)
        if not os.path.exists(img_path):
            img_name = lbl_name.replace(".txt", ".png")
            img_path = os.path.join(UNIFIED_TRAIN_IMG, img_name)
            if not os.path.exists(img_path):
                continue

        avg_area = sum(b[3]*b[4] for b in fallen_bboxes) / len(fallen_bboxes)
        remapped = [(2, cx, cy, w, h) for _, cx, cy, w, h in fallen_bboxes]
        fallen.append({
            'img': img_name, 'lbl': lbl_name,
            'avg_area': avg_area, 'n_bbox': len(remapped),
            'source': 'unified',
            'bboxes': remapped,
        })

        if len(fallen) >= max_images:
            break

        scanned += 1
        if scanned % 10000 == 0:
            print(f"    스캔 {scanned}... (found {len(fallen)})")

    return fallen


def prepare():
    """v29 데이터셋 빌드"""
    print("=" * 60)
    print("  v29 3-class 데이터셋 빌드")
    print("  랜덤 샘플링 + close_mosaic=0 (v30 비교 baseline)")
    print("=" * 60)

    for subdir in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)

    train_img_dir = os.path.join(OUT_DIR, "train", "images")
    train_lbl_dir = os.path.join(OUT_DIR, "train", "labels")
    val_img_dir = os.path.join(OUT_DIR, "valid", "images")
    val_lbl_dir = os.path.join(OUT_DIR, "valid", "labels")

    # ── 1. L 티어 helmet train ──
    print("\n[1/5] L 티어 helmet train...")
    helmet_count = 0
    for img_name in os.listdir(L_TIER_TRAIN_IMG):
        if not img_name.endswith((".jpg", ".png")):
            continue
        lbl_name = Path(img_name).stem + ".txt"
        src_img = os.path.join(L_TIER_TRAIN_IMG, img_name)
        src_lbl = os.path.join(L_TIER_TRAIN_LBL, lbl_name)
        dst_img = os.path.join(train_img_dir, img_name)
        dst_lbl = os.path.join(train_lbl_dir, lbl_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)
        helmet_count += 1
    print(f"  Helmet: {helmet_count}장")

    # ── 2. v24 fallen (area 필터 없음, 전체) ──
    print("\n[2/5] v24 fallen (전체, area 필터 없음)...")
    v24_fallen = extract_v24_fallen_all()
    random.shuffle(v24_fallen)
    v24_count = 0
    for item in v24_fallen:
        src_img = os.path.join(V24_TRAIN_IMG, item['img'])
        dst_img = os.path.join(train_img_dir, item['img'])
        dst_lbl = os.path.join(train_lbl_dir, item['lbl'])
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            os.symlink(os.path.join(V24_TRAIN_LBL, item['lbl']), dst_lbl)
        v24_count += 1
    print(f"  v24 fallen: {v24_count}장 (avg area: {sum(i['avg_area'] for i in v24_fallen)/len(v24_fallen)*100:.2f}%)")

    # ── 3. unified fallen (랜덤, area 필터 없음) ──
    remaining = max(0, TARGET_FALLEN - v24_count)
    print(f"\n[3/5] unified fallen (랜덤, 최대 {remaining}장)...")
    if remaining > 0:
        unified_fallen = extract_unified_fallen_random(max_images=remaining)
    else:
        unified_fallen = []
    unified_count = 0
    for item in unified_fallen:
        dst_img_name = "uf_" + item['img']
        dst_lbl_name = "uf_" + item['lbl']
        src_img = os.path.join(UNIFIED_TRAIN_IMG, item['img'])
        dst_img = os.path.join(train_img_dir, dst_img_name)
        dst_lbl = os.path.join(train_lbl_dir, dst_lbl_name)

        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            with open(dst_lbl, 'w') as f:
                for cls, cx, cy, w, h in item['bboxes']:
                    f.write(f"{cls} {cx} {cy} {w} {h}\n")
        unified_count += 1
    if unified_fallen:
        print(f"  unified fallen: {unified_count}장 (avg area: {sum(i['avg_area'] for i in unified_fallen)/len(unified_fallen)*100:.2f}%)")
    else:
        print(f"  unified fallen: {unified_count}장 (v24로 충분)")

    # ── 4. Helmet val ──
    print("\n[4/5] Helmet val...")
    val_helmet = 0
    for img_name in os.listdir(L_TIER_VAL_IMG):
        if not img_name.endswith((".jpg", ".png")):
            continue
        lbl_name = Path(img_name).stem + ".txt"
        src_img = os.path.join(L_TIER_VAL_IMG, img_name)
        src_lbl = os.path.join(L_TIER_VAL_LBL, lbl_name)
        dst_img = os.path.join(val_img_dir, img_name)
        dst_lbl = os.path.join(val_lbl_dir, lbl_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)
        val_helmet += 1
    print(f"  Helmet val: {val_helmet}장")

    # ── 5. Fallen val ──
    print("\n[5/5] Fallen val...")
    val_fallen = 0
    for lbl_name in os.listdir(V24_VAL_LBL):
        if not lbl_name.startswith("fallen_"):
            continue
        lbl_path = os.path.join(V24_VAL_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)
        if not any(b[0] == 2 for b in bboxes):
            continue
        img_name = lbl_name.replace(".txt", ".jpg")
        if not os.path.exists(os.path.join(V24_VAL_IMG, img_name)):
            continue
        dst_img = os.path.join(val_img_dir, img_name)
        dst_lbl = os.path.join(val_lbl_dir, lbl_name)
        if not os.path.exists(dst_img):
            os.symlink(os.path.join(V24_VAL_IMG, img_name), dst_img)
        if not os.path.exists(dst_lbl):
            os.symlink(lbl_path, dst_lbl)
        val_fallen += 1
    print(f"  Fallen val: {val_fallen}장")

    # ── data.yaml ──
    data_yaml = os.path.join(OUT_DIR, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: 3\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")
        f.write("  2: fallen\n")

    total_fallen = v24_count + unified_count
    total_train = helmet_count + total_fallen
    fallen_ratio = total_fallen / total_train * 100

    # Area 분포 통계
    all_fallen = v24_fallen + unified_fallen
    areas = [item['avg_area'] for item in all_fallen]
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
    bin_names = ["<1%", "1~5%", "5~10%", "10~20%", ">20%"]

    print(f"\n{'='*60}")
    print(f"  데이터셋 완성: {OUT_DIR}")
    print(f"  Train: {total_train}장")
    print(f"    Helmet: {helmet_count}장 ({helmet_count/total_train*100:.1f}%)")
    print(f"    Fallen: {total_fallen}장 ({fallen_ratio:.1f}%)")
    print(f"      v24: {v24_count}, unified: {unified_count}")
    print(f"  Val: {val_helmet + val_fallen}장 (helmet {val_helmet} + fallen {val_fallen})")
    print(f"\n  Fallen area 분포:")
    for (lo, hi), name in zip(bins, bin_names):
        cnt = sum(1 for a in areas if lo <= a < hi)
        pct = cnt / len(areas) * 100 if areas else 0
        print(f"    {name:>8s}: {cnt:4d}장 ({pct:5.1f}%)")
    print(f"{'='*60}")


def train(batch=4, epochs=100, resume=False):
    """v29 학습"""
    from ultralytics import YOLO

    project = HOBAN
    name = "hoban_go3k_v29"
    data_yaml = os.path.join(OUT_DIR, "data.yaml")

    if not os.path.exists(data_yaml):
        print("데이터셋 없음. 먼저 --prepare 실행")
        return

    if resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        if not os.path.exists(ckpt):
            print(f"체크포인트 없음: {ckpt}")
            return
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print("=" * 60)
    print("  v29 3-class: 랜덤 샘플링 + close_mosaic=0")
    print(f"  Model: yolo26m.pt (COCO pretrained)")
    print(f"  Optimizer: SGD, lr0=0.005, 1280px, batch={batch}")
    print(f"  copy_paste=0.4, scale=0.7, close_mosaic=0")
    print("=" * 60)

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

        # SGD (v26 동일)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (v26 동일 + close_mosaic 변경)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.7,
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.15,
        close_mosaic=0,      # v26: 10 → 0 (mosaic 항상 유지)

        # Early stopping
        patience=25,          # v26: 20 → 25
        amp=True,
        workers=4,
        seed=42,
        plots=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/{name}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="v29 3-class (랜덤 샘플링 + close_mosaic=0)")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.train:
        train(batch=args.batch, epochs=args.epochs)
    elif args.resume:
        train(batch=args.batch, resume=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
