#!/usr/bin/env python3
"""v32 3-class: 큐레이션된 fallen + negative 복원

v29 대비 변경:
  - fallen 소스 변경: v24 → unified_safety_all 직접 선별
  - fallen.v2i 완전 배제 (cam3 FP 원인: 오버헤드 CCTV tiny blob)
  - 품질 필터 적용: area/AR/cy/엣지 기준
  - Roboflow dedup + fastdup 시각적 유사도 dedup (>=0.9)
  - negative 복원: 실제 현장 CCTV neg_1k_manual 960장 (v29에서 누락됨)
  - 150 epoch, patience=35 (v31 동일)

데이터셋:
  Helmet (L-tier):     2,000장
  Fallen (큐레이션):   2,000장 (fall.v4i 1,176 + Fall.v1i 418 + fall.v1i 406)
  Negative:            960장 (neg_1k_manual, 실제 현장 CCTV)
  Total:               4,960장

클래스:
  0: person_with_helmet
  1: person_without_helmet
  2: fallen

사용법:
  python train_go3k_v32.py --prepare   # 데이터셋 빌드
  python train_go3k_v32.py --train     # 학습
  python train_go3k_v32.py --resume    # 이어서 학습
"""
import os
import random
from pathlib import Path

random.seed(42)

HOBAN = "/home/lay/hoban"
OUT_DIR = f"{HOBAN}/datasets_go3k_v32"
MODEL = f"{HOBAN}/yolo26m.pt"

# 소스: 헬멧 (L-tier)
L_TIER_TRAIN_IMG = f"{HOBAN}/datasets_minimal_l/train/images"
L_TIER_TRAIN_LBL = f"{HOBAN}/datasets_minimal_l/train/labels"
L_TIER_VAL_IMG = f"{HOBAN}/datasets_minimal_l/valid/images"
L_TIER_VAL_LBL = f"{HOBAN}/datasets_minimal_l/valid/labels"

# 소스: fallen (unified_safety_all, class 4 → class 2)
UNIFIED_TRAIN_IMG = "/data/unified_safety_all/train/images"
UNIFIED_TRAIN_LBL = "/data/unified_safety_all/train/labels"

# 소스: fallen val (v24에서 가져옴)
V24_VAL_IMG = f"{HOBAN}/datasets_go3k_v24/valid/images"
V24_VAL_LBL = f"{HOBAN}/datasets_go3k_v24/valid/labels"

# 소스: negative (실제 현장 CCTV, CVAT 검증)
NEG_IMG_DIR = f"{HOBAN}/datasets/cvat/neg_1k_manual/images"
NEG_LBL_DIR = f"{HOBAN}/datasets/cvat/neg_1k_manual/labels"

# 타겟
TARGET_FALLEN = 2000
TARGET_NEGATIVE = 960  # neg_1k_manual 전체

# 배제 소스 (FP 원인)
EXCLUDED_SOURCES = {"fallen.v2i"}

# 소스 우선순위 (높을수록 먼저 선택)
SOURCE_PRIORITY = {
    "fall.v4i": 10,   # 최고 품질 (통과율 98.9%)
    "fall.v1i": 7,    # 양호 (67.5%)
    "Fall.v1i": 7,    # 양호 (64.8%)
    "Fall.v3i": 5,    # 보통 (78.3%)
    "fall.v2i": 3,    # 보통 (41.2%)
}


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


def roboflow_base(fname):
    """Roboflow 원본 식별자 추출 (.rf. 해시 제거)"""
    # "fall.v4i.yolo26_img_abc.rf.hash123.jpg" → "fall.v4i.yolo26_img_abc"
    name = Path(fname).stem
    if ".rf." in name:
        return name.split(".rf.")[0]
    return name


def quality_filter(bboxes):
    """품질 기준으로 fallen bbox 필터링. 통과한 bbox만 반환."""
    good = []
    for cls, cx, cy, w, h in bboxes:
        if cls != 4:  # fallen만
            continue
        ar = w / h if h > 0 else 0
        area = w * h * 100  # percentage

        # 제외 조건
        if area < 0.1:          # 극소
            continue
        if cy < 0.15:           # 천장 위치
            continue
        if area < 0.5 and cy < 0.3:  # tiny top blob
            continue
        if area > 25:           # 과대
            continue
        if ar > 4.0 or ar < 0.2:  # 극단 AR
            continue
        if cx < 0.02 or cx > 0.98:  # 엣지 x
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


def extract_curated_fallen():
    """unified_safety_all에서 고품질 fallen 추출

    1. fallen.v2i 완전 배제
    2. 품질 필터 적용
    3. Roboflow dedup (.rf. 해시)
    4. 소스 우선순위에 따라 선별
    """
    print("  스캔 중...")
    candidates = {}  # roboflow_base → best candidate

    label_files = [f for f in os.listdir(UNIFIED_TRAIN_LBL) if f.endswith('.txt')]
    scanned = 0
    skipped_source = 0
    skipped_quality = 0

    for lbl_name in label_files:
        fname = lbl_name.replace('.txt', '')
        src = get_source(fname)

        # 배제 소스 건너뛰기
        if src in EXCLUDED_SOURCES:
            skipped_source += 1
            continue

        if src not in SOURCE_PRIORITY:
            continue

        lbl_path = os.path.join(UNIFIED_TRAIN_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)

        # 품질 필터
        good_bboxes = quality_filter(bboxes)
        if not good_bboxes:
            skipped_quality += 1
            continue

        # 이미지 존재 확인
        img_name = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = fname + ext
            if os.path.exists(os.path.join(UNIFIED_TRAIN_IMG, candidate)):
                img_name = candidate
                break
        if not img_name:
            continue

        # Roboflow dedup: 같은 원본에서 가장 높은 우선순위 소스만
        base = roboflow_base(fname)
        priority = SOURCE_PRIORITY.get(src, 0)
        avg_area = sum(b[3] * b[4] for b in good_bboxes) / len(good_bboxes)

        if base not in candidates or priority > candidates[base]['priority']:
            candidates[base] = {
                'img': img_name,
                'lbl': lbl_name,
                'bboxes': good_bboxes,
                'source': src,
                'priority': priority,
                'avg_area': avg_area,
                'n_bbox': len(good_bboxes),
            }

        scanned += 1
        if scanned % 50000 == 0:
            print(f"    스캔 {scanned}... (unique: {len(candidates)})")

    print(f"  스캔 완료: {scanned}장 처리")
    print(f"  배제: fallen.v2i {skipped_source}장, 품질 미달 {skipped_quality}장")
    print(f"  Roboflow dedup 후: {len(candidates)}장 (unique)")

    # 소스 우선순위 → 랜덤 정렬
    items = list(candidates.values())
    items.sort(key=lambda x: (-x['priority'], random.random()))

    return items


def prepare():
    """v32 데이터셋 빌드"""
    print("=" * 60)
    print("  v32 3-class 데이터셋 빌드")
    print("  큐레이션된 fallen + negative 복원")
    print("=" * 60)

    for subdir in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)

    train_img_dir = os.path.join(OUT_DIR, "train", "images")
    train_lbl_dir = os.path.join(OUT_DIR, "train", "labels")
    val_img_dir = os.path.join(OUT_DIR, "valid", "images")
    val_lbl_dir = os.path.join(OUT_DIR, "valid", "labels")

    # ── 1. L 티어 helmet train ──
    print("\n[1/6] L 티어 helmet train...")
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

    # ── 2. Fallen (큐레이션) ──
    print(f"\n[2/6] Fallen 큐레이션 (목표: {TARGET_FALLEN}장)...")
    all_fallen = extract_curated_fallen()
    selected_fallen = all_fallen[:TARGET_FALLEN]

    # 소스별 통계
    src_counts = {}
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

    print(f"  Fallen: {fallen_count}장")
    for src in sorted(src_counts.keys(), key=lambda x: -src_counts[x]):
        print(f"    {src}: {src_counts[src]}장")

    # ── 3. Negative ──
    print(f"\n[3/6] Negative (목표: {TARGET_NEGATIVE}장)...")
    neg_images = [f for f in os.listdir(NEG_IMG_DIR) if f.endswith((".jpg", ".png"))]
    random.shuffle(neg_images)
    neg_selected = neg_images[:TARGET_NEGATIVE]
    neg_count = 0
    for img_name in neg_selected:
        dst_img_name = f"neg_{img_name}"
        dst_lbl_name = f"neg_{Path(img_name).stem}.txt"
        src_img = os.path.join(NEG_IMG_DIR, img_name)
        dst_img = os.path.join(train_img_dir, dst_img_name)
        dst_lbl = os.path.join(train_lbl_dir, dst_lbl_name)
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            # 빈 라벨 파일 (negative)
            open(dst_lbl, 'w').close()
        neg_count += 1
    print(f"  Negative: {neg_count}장")

    # ── 4. Helmet val ──
    print("\n[4/6] Helmet val...")
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
    print("\n[5/6] Fallen val...")
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
    print("\n[6/6] data.yaml 생성...")
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

    total_train = helmet_count + fallen_count + neg_count

    # Fallen area 분포
    areas = [item['avg_area'] for item in selected_fallen]
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
    bin_names = ["<1%", "1~5%", "5~10%", "10~20%", ">20%"]

    print(f"\n{'='*60}")
    print(f"  데이터셋 완성: {OUT_DIR}")
    print(f"  Train: {total_train}장")
    print(f"    Helmet:   {helmet_count}장 ({helmet_count/total_train*100:.1f}%)")
    print(f"    Fallen:   {fallen_count}장 ({fallen_count/total_train*100:.1f}%)")
    for src in sorted(src_counts.keys(), key=lambda x: -src_counts[x]):
        print(f"      {src}: {src_counts[src]}장")
    print(f"    Negative: {neg_count}장 ({neg_count/total_train*100:.1f}%)")
    print(f"  Val: {val_helmet + val_fallen}장 (helmet {val_helmet} + fallen {val_fallen})")
    if areas:
        print(f"\n  Fallen area 분포:")
        for (lo, hi), name in zip(bins, bin_names):
            cnt = sum(1 for a in areas if lo <= a < hi)
            pct = cnt / len(areas) * 100
            print(f"    {name:>8s}: {cnt:4d}장 ({pct:5.1f}%)")
    print(f"\n  ※ fallen.v2i 완전 배제됨")
    print(f"  ※ 품질 필터: area 0.1~25%, AR 0.2~4.0, cy>0.15, 엣지 제외")
    print(f"{'='*60}")


def train(batch=4, epochs=150, resume=False):
    """v32 학습"""
    from ultralytics import YOLO

    project = HOBAN
    name = "hoban_go3k_v32"
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
    print("  v32 3-class: 큐레이션 fallen + negative 복원")
    print(f"  Model: yolo26m.pt (COCO pretrained)")
    print(f"  Optimizer: SGD, lr0=0.005, 1280px, batch={batch}")
    print(f"  close_mosaic=0, epochs={epochs}, patience=35")
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

        # SGD (v29 동일)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (v29 동일)
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
        close_mosaic=0,

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
    parser = argparse.ArgumentParser(description="v32 3-class (큐레이션 fallen + negative)")
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
