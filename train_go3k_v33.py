#!/usr/bin/env python3
"""v33 3-class: helmet 3k + fallen 3k (small/tiny 우선) + negative 960

v32 대비 변경:
  - Helmet 2,000 → 3,000 (v19에서 +1,000 추가)
  - Fallen 2,000 → 3,000 (미사용 풀에서 small/tiny 우선 +1,000)
  - Augmentation: scale=0.9, multi_scale=0.25, close_mosaic=15,
    erasing=0.3, perspective=0.0005 (tiny 객체 학습 강화)
  - Negative 960장 동일

데이터셋:
  Helmet:    3,000장 (L-tier 2,000 + v19 추가 1,000)
  Fallen:    3,000장 (v32 큐레이션 2,000 + small/tiny 우선 1,000)
  Negative:    960장 (neg_1k_manual, 실제 현장 CCTV)
  Total:     6,960장

클래스:
  0: person_with_helmet
  1: person_without_helmet
  2: fallen

사용법:
  python train_go3k_v33.py --prepare   # 데이터셋 빌드
  python train_go3k_v33.py --train     # 학습
  python train_go3k_v33.py --resume    # 이어서 학습
"""
import json
import os
import random
from pathlib import Path

random.seed(42)

HOBAN = "/home/lay/hoban"
OUT_DIR = f"{HOBAN}/datasets_go3k_v33"
MODEL = f"{HOBAN}/yolo26m.pt"

# ── 소스: 헬멧 ──
L_TIER_TRAIN_IMG = f"{HOBAN}/datasets_minimal_l/train/images"
L_TIER_TRAIN_LBL = f"{HOBAN}/datasets_minimal_l/train/labels"
L_TIER_VAL_IMG = f"{HOBAN}/datasets_minimal_l/valid/images"
L_TIER_VAL_LBL = f"{HOBAN}/datasets_minimal_l/valid/labels"
V19_TRAIN_IMG = f"{HOBAN}/datasets_go3k_v19/train/images"
V19_TRAIN_LBL = f"{HOBAN}/datasets_go3k_v19/train/labels"

# ── 소스: fallen ──
UNIFIED_TRAIN_IMG = "/data/unified_safety_all/train/images"
UNIFIED_TRAIN_LBL = "/data/unified_safety_all/train/labels"
FASTDUP_KEEP = f"{HOBAN}/.omc/fastdup_v32_full/deduped_keep.json"

# ── 소스: fallen val ──
V24_VAL_IMG = f"{HOBAN}/datasets_go3k_v24/valid/images"
V24_VAL_LBL = f"{HOBAN}/datasets_go3k_v24/valid/labels"

# ── 소스: negative ──
NEG_IMG_DIR = f"{HOBAN}/datasets/cvat/neg_1k_manual/images"
NEG_LBL_DIR = f"{HOBAN}/datasets/cvat/neg_1k_manual/labels"

# ── Val 제외 목록 ──
VAL_IMG_DIR = f"{HOBAN}/datasets_go3k_v16/valid/images"

# 타겟
TARGET_HELMET = 3000
TARGET_FALLEN = 3000
TARGET_NEGATIVE = 960

# 배제 소스
EXCLUDED_SOURCES = {"fallen.v2i"}


def get_source(fname):
    """파일명에서 Roboflow 소스 추출"""
    if "fallen.v2i" in fname:
        return "fallen.v2i"
    elif "fall.v4i" in fname:
        return "fall.v4i"
    elif "fall.v1i" in fname and "Fall" not in fname:
        return "fall.v1i"
    elif "Fall.v1i" in fname:
        return "Fall.v1i"
    elif "Fall.v3i" in fname:
        return "Fall.v3i"
    elif "fall.v2i" in fname:
        return "fall.v2i"
    return "other"


def quality_filter(bboxes):
    """품질 기준으로 fallen bbox 필터링"""
    good = []
    for cls, cx, cy, w, h in bboxes:
        if cls != 4:
            continue
        ar = w / h if h > 0 else 0
        area = w * h * 100

        if area < 0.1:
            continue
        if cy < 0.15:
            continue
        if area < 0.5 and cy < 0.3:
            continue
        if area > 25:
            continue
        if ar > 4.0 or ar < 0.2:
            continue
        if cx < 0.02 or cx > 0.98:
            continue

        good.append((2, cx, cy, w, h))  # class 4 → 2 remap
    return good


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


def min_fallen_area(bboxes):
    """fallen bbox 중 최소 area 반환"""
    areas = [w * h for cls, cx, cy, w, h in bboxes if cls == 2]
    return min(areas) if areas else 1.0


def load_fastdup_pool():
    """fastdup dedup 풀 로드 + 품질 필터 + min_area 계산"""
    with open(FASTDUP_KEEP) as f:
        pool = json.load(f)

    items = []
    for entry in pool:
        lbl_path = os.path.join(UNIFIED_TRAIN_LBL, entry['lbl'])
        bboxes = parse_labels(lbl_path)
        good = quality_filter(bboxes)
        if not good:
            continue

        # 이미지 존재 확인
        img_name = None
        stem = Path(entry['lbl']).stem
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = stem + ext
            if os.path.exists(os.path.join(UNIFIED_TRAIN_IMG, candidate)):
                img_name = candidate
                break
        if not img_name:
            continue

        items.append({
            'img': img_name,
            'lbl': entry['lbl'],
            'bboxes': good,
            'source': entry['source'],
            'min_area': min([w * h for _, _, _, w, h in good]),
            'avg_area': sum(w * h for _, _, _, w, h in good) / len(good),
        })

    return items


def prepare():
    """v33 데이터셋 빌드"""
    print("=" * 60)
    print("  v33 3-class 데이터셋 빌드")
    print("  Helmet 3k + Fallen 3k (small/tiny 우선) + Neg 960")
    print("=" * 60)

    for subdir in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)

    train_img_dir = os.path.join(OUT_DIR, "train", "images")
    train_lbl_dir = os.path.join(OUT_DIR, "train", "labels")
    val_img_dir = os.path.join(OUT_DIR, "valid", "images")
    val_lbl_dir = os.path.join(OUT_DIR, "valid", "labels")

    # Val 제외 목록
    val_images = set()
    if os.path.isdir(VAL_IMG_DIR):
        val_images = set(os.listdir(VAL_IMG_DIR))
    print(f"  Val 제외 목록: {len(val_images)}장")

    # ── 1. Helmet: L-tier 2,000 + v19 추가 1,000 ──
    print(f"\n[1/7] Helmet (목표: {TARGET_HELMET}장)...")

    # 1a. L-tier 2,000장
    ltier_imgs = set()
    helmet_count = 0
    for img_name in sorted(os.listdir(L_TIER_TRAIN_IMG)):
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
        ltier_imgs.add(img_name)
        helmet_count += 1
    print(f"  L-tier: {helmet_count}장")

    # 1b. v19에서 추가 1,000장
    extra_needed = TARGET_HELMET - helmet_count
    v19_candidates = []
    for img_name in os.listdir(V19_TRAIN_IMG):
        if not img_name.endswith((".jpg", ".png")):
            continue
        if img_name in ltier_imgs or img_name in val_images:
            continue
        lbl_name = Path(img_name).stem + ".txt"
        lbl_path = os.path.join(V19_TRAIN_LBL, lbl_name)
        if not os.path.exists(lbl_path):
            continue
        bboxes = parse_labels(lbl_path)
        if not any(b[0] in [0, 1] for b in bboxes):
            continue
        v19_candidates.append(img_name)

    random.shuffle(v19_candidates)
    v19_selected = v19_candidates[:extra_needed]
    extra_count = 0
    for img_name in v19_selected:
        lbl_name = Path(img_name).stem + ".txt"
        src_img = os.path.join(V19_TRAIN_IMG, img_name)
        src_lbl = os.path.join(V19_TRAIN_LBL, lbl_name)
        dst_img = os.path.join(train_img_dir, img_name)
        dst_lbl = os.path.join(train_lbl_dir, lbl_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
            os.symlink(src_lbl, dst_lbl)
        extra_count += 1
    helmet_count += extra_count
    print(f"  v19 추가: {extra_count}장")
    print(f"  Helmet 합계: {helmet_count}장")

    # ── 2. Fallen 3,000장 (fastdup 풀에서 small/tiny 우선) ──
    print(f"\n[2/7] Fallen (목표: {TARGET_FALLEN}장, small/tiny 우선)...")
    print("  fastdup 풀 로딩...")
    all_fallen = load_fastdup_pool()
    print(f"  풀 크기: {len(all_fallen)}장")

    # min_area 오름차순 정렬 (tiny/small 우선)
    all_fallen.sort(key=lambda x: x['min_area'])
    selected_fallen = all_fallen[:TARGET_FALLEN]

    # 소스별/area별 통계
    src_counts = {}
    area_bins = {'tiny(<1%)': 0, 'small(1~5%)': 0, 'medium(5~15%)': 0, 'large(>15%)': 0}
    fallen_count = 0
    for item in selected_fallen:
        dst_img_name = f"fallen_{item['img']}"
        dst_lbl_name = f"fallen_{Path(item['img']).stem}.txt"
        src_img = os.path.join(UNIFIED_TRAIN_IMG, item['img'])
        dst_img = os.path.join(train_img_dir, dst_img_name)
        dst_lbl = os.path.join(train_lbl_dir, dst_lbl_name)

        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            with open(dst_lbl, 'w') as f:
                for cls, cx, cy, w, h in item['bboxes']:
                    f.write(f"{cls} {cx} {cy} {w} {h}\n")

        fallen_count += 1
        src = item['source']
        src_counts[src] = src_counts.get(src, 0) + 1

        # area 분류 (avg)
        avg = item['avg_area'] * 100
        if avg < 1:
            area_bins['tiny(<1%)'] += 1
        elif avg < 5:
            area_bins['small(1~5%)'] += 1
        elif avg < 15:
            area_bins['medium(5~15%)'] += 1
        else:
            area_bins['large(>15%)'] += 1

    print(f"  Fallen: {fallen_count}장")
    for src in sorted(src_counts.keys(), key=lambda x: -src_counts[x]):
        print(f"    {src}: {src_counts[src]}장")
    print(f"  Area 분포:")
    for name, cnt in area_bins.items():
        print(f"    {name:>15s}: {cnt:4d}장 ({cnt/fallen_count*100:.1f}%)")

    # ── 3. Negative ──
    print(f"\n[3/7] Negative ({TARGET_NEGATIVE}장)...")
    neg_images = [f for f in os.listdir(NEG_IMG_DIR) if f.endswith((".jpg", ".png"))]
    neg_count = 0
    for img_name in neg_images:
        dst_img_name = f"neg_{img_name}"
        dst_lbl_name = f"neg_{Path(img_name).stem}.txt"
        src_img = os.path.join(NEG_IMG_DIR, img_name)
        dst_img = os.path.join(train_img_dir, dst_img_name)
        dst_lbl = os.path.join(train_lbl_dir, dst_lbl_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            open(dst_lbl, 'w').close()
        neg_count += 1
    print(f"  Negative: {neg_count}장")

    # ── 4. Helmet val ──
    print("\n[4/7] Helmet val...")
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
    print("\n[5/7] Fallen val...")
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

    # ── 6. data.yaml ──
    print("\n[6/7] data.yaml...")
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

    # ── 7. 요약 ──
    total_train = helmet_count + fallen_count + neg_count
    print(f"\n{'='*60}")
    print(f"  v33 데이터셋 완성: {OUT_DIR}")
    print(f"  Train: {total_train}장")
    print(f"    Helmet:   {helmet_count}장 ({helmet_count/total_train*100:.1f}%)")
    print(f"    Fallen:   {fallen_count}장 ({fallen_count/total_train*100:.1f}%)")
    for src in sorted(src_counts.keys(), key=lambda x: -src_counts[x]):
        print(f"      {src}: {src_counts[src]}장")
    print(f"    Negative: {neg_count}장 ({neg_count/total_train*100:.1f}%)")
    print(f"  Val: {val_helmet + val_fallen}장 (helmet {val_helmet} + fallen {val_fallen})")
    print(f"\n  Fallen area 분포:")
    for name, cnt in area_bins.items():
        print(f"    {name:>15s}: {cnt:4d}장 ({cnt/fallen_count*100:.1f}%)")
    print(f"\n  v32 대비 변경:")
    print(f"    Helmet: 2,000 → {helmet_count}")
    print(f"    Fallen: 2,000 → {fallen_count} (small/tiny 우선 추가)")
    print(f"    Augmentation: scale=0.9, multi_scale=0.25, close_mosaic=15")
    print(f"{'='*60}")


def train(batch=4, epochs=150, resume=False):
    """v33 학습"""
    from ultralytics import YOLO

    project = HOBAN
    name = "hoban_go3k_v33"
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
    print("  v33 3-class: helmet 3k + fallen 3k (tiny 강화) + neg 960")
    print(f"  Model: yolo26m.pt (COCO pretrained)")
    print(f"  Optimizer: SGD, lr0=0.005, 1280px, batch={batch}")
    print(f"  scale=0.9, multi_scale=0.25, close_mosaic=15")
    print(f"  epochs={epochs}, patience=35")
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

        # SGD
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (tiny 최적화)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.9,              # v32: 0.7 → tiny 생성 극대화
        multi_scale=0.25,       # 신규: 960~1600px 랜덤 변동
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.3,            # v32: 0.15 → occlusion 강화
        perspective=0.0005,     # 신규: CCTV 부감 시뮬레이션
        close_mosaic=15,        # v32: 0 → 마지막 15ep mosaic 해제

        # Extended training
        patience=35,
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
    parser = argparse.ArgumentParser(description="v33 3-class (helmet 3k + fallen 3k tiny + neg 960)")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
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
