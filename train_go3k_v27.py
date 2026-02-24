#!/usr/bin/env python3
"""v27 3-class: 1:1 비율 (helmet 2K + fallen 2K) + 50% 합성 데이터

v26 대비 변경:
  - helmet:fallen = 1:1 비율 (2,000:2,000)
  - fallen 구성: 실제 1,000 + 합성 1,000
  - 실제 fallen: v24 고품질 (area 5~20%, 학습 효과 높은 것만)
  - 합성 fallen: v24 크롭 → CCTV 배경에 축소 붙여넣기

클래스:
  0: person_with_helmet
  1: person_without_helmet
  2: fallen

사용법:
  python train_go3k_v27.py --prepare   # 데이터셋 빌드 (합성 포함)
  python train_go3k_v27.py --train     # 학습
  python train_go3k_v27.py --resume    # 이어서 학습
"""
import os
import random
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

HOBAN = "/home/lay/hoban"
OUT_DIR = f"{HOBAN}/datasets_go3k_v27"
MODEL = f"{HOBAN}/yolo26m.pt"

# 소스: 헬멧
L_TIER_TRAIN_IMG = f"{HOBAN}/datasets_minimal_l/train/images"
L_TIER_TRAIN_LBL = f"{HOBAN}/datasets_minimal_l/train/labels"
L_TIER_VAL_IMG = f"{HOBAN}/datasets_minimal_l/valid/images"
L_TIER_VAL_LBL = f"{HOBAN}/datasets_minimal_l/valid/labels"

# 소스: fallen (v24)
V24_TRAIN_IMG = f"{HOBAN}/datasets_go3k_v24/train/images"
V24_TRAIN_LBL = f"{HOBAN}/datasets_go3k_v24/train/labels"
V24_VAL_IMG = f"{HOBAN}/datasets_go3k_v24/valid/images"
V24_VAL_LBL = f"{HOBAN}/datasets_go3k_v24/valid/labels"

# 합성 설정
REAL_FALLEN = 1000
SYNTH_FALLEN = 1000
# 실제 fallen 필터: area 5~20% (너무 작으면 품질↓, 너무 크면 CCTV 비현실적)
REAL_MIN_AREA = 0.02
REAL_MAX_AREA = 0.20
# 합성 크롭 소스: area > 5% (크롭 해상도 확보)
CROP_MIN_AREA = 0.05
# 합성 붙여넣기 스케일: CCTV 프레임 대비 면적
PASTE_MIN_AREA = 0.005
PASTE_MAX_AREA = 0.03


def parse_labels(label_path):
    """라벨 파일 파싱 → [(class_id, cx, cy, w, h), ...]"""
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


def select_real_fallen():
    """v24에서 고품질 fallen 선별 (area 5~20%, bbox 명확한 것)"""
    candidates = []
    for lbl_name in os.listdir(V24_TRAIN_LBL):
        if not lbl_name.startswith("fallen_"):
            continue
        lbl_path = os.path.join(V24_TRAIN_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)
        fallen_bboxes = [b for b in bboxes if b[0] == 2]
        if not fallen_bboxes:
            continue

        # area 5~20% 범위의 fallen bbox가 있는 이미지만
        valid = [b for b in fallen_bboxes if REAL_MIN_AREA <= b[3] * b[4] <= REAL_MAX_AREA]
        if not valid:
            continue

        img_name = lbl_name.replace(".txt", ".jpg")
        if not os.path.exists(os.path.join(V24_TRAIN_IMG, img_name)):
            continue

        avg_area = sum(b[3] * b[4] for b in valid) / len(valid)
        candidates.append({
            'img': img_name, 'lbl': lbl_name,
            'avg_area': avg_area, 'n_bbox': len(valid),
            'bboxes': bboxes,
        })

    # bbox 수 많은 것 우선 (다양한 포즈), 그 다음 area 중간 크기 우선
    candidates.sort(key=lambda x: (-x['n_bbox'], abs(x['avg_area'] - 0.08)))
    return candidates[:REAL_FALLEN]


def extract_fallen_crops():
    """v24에서 fallen bbox 크롭 추출 (합성용)"""
    import cv2

    crops = []
    for lbl_name in os.listdir(V24_TRAIN_LBL):
        if not lbl_name.startswith("fallen_"):
            continue
        lbl_path = os.path.join(V24_TRAIN_LBL, lbl_name)
        bboxes = parse_labels(lbl_path)
        fallen_bboxes = [b for b in bboxes if b[0] == 2]
        if not fallen_bboxes:
            continue

        # 큰 bbox만 (크롭 품질 확보)
        large = [b for b in fallen_bboxes if b[3] * b[4] >= CROP_MIN_AREA]
        if not large:
            continue

        img_name = lbl_name.replace(".txt", ".jpg")
        img_path = os.path.join(V24_TRAIN_IMG, img_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        for _, cx, cy, w, h in large:
            x1 = max(0, int((cx - w / 2) * w_img))
            y1 = max(0, int((cy - h / 2) * h_img))
            x2 = min(w_img, int((cx + w / 2) * w_img))
            y2 = min(h_img, int((cy + h / 2) * h_img))
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] >= 30 and crop.shape[1] >= 30:
                crops.append(crop)

    random.shuffle(crops)
    return crops


def generate_synthetic_fallen(bg_img_dir, crops, output_img_dir, output_lbl_dir, count=1000):
    """CCTV 배경에 fallen 크롭을 축소 붙여넣기하여 합성 이미지 생성"""
    import cv2

    bg_files = [f for f in os.listdir(bg_img_dir) if f.endswith(('.jpg', '.png'))]
    if not bg_files:
        print("  배경 이미지 없음!")
        return 0
    if not crops:
        print("  크롭 없음!")
        return 0

    print(f"  배경: {len(bg_files)}장, 크롭: {len(crops)}개")
    generated = 0

    for i in range(count):
        # 랜덤 배경 선택
        bg_name = random.choice(bg_files)
        bg_path = os.path.join(bg_img_dir, bg_name)
        # symlink resolve
        if os.path.islink(bg_path):
            bg_path = os.path.realpath(bg_path)
        bg = cv2.imread(bg_path)
        if bg is None:
            continue
        bg_h, bg_w = bg.shape[:2]

        # 기존 helmet 라벨 읽기 (배경 이미지의 원본 라벨 보존)
        bg_stem = Path(bg_name).stem
        bg_lbl_path = os.path.join(
            L_TIER_TRAIN_LBL, bg_stem + ".txt"
        )
        existing_bboxes = parse_labels(bg_lbl_path)

        # 1~3개 fallen 배치
        n_paste = random.randint(1, 3)
        new_bboxes = list(existing_bboxes)

        for _ in range(n_paste):
            crop = random.choice(crops)
            crop_h, crop_w = crop.shape[:2]

            # 목표 면적 (CCTV 스케일)
            target_area = random.uniform(PASTE_MIN_AREA, PASTE_MAX_AREA)
            target_pixels = target_area * bg_w * bg_h
            aspect = crop_w / max(crop_h, 1)
            new_h = int(np.sqrt(target_pixels / max(aspect, 0.1)))
            new_w = int(new_h * aspect)
            new_h = max(15, min(new_h, bg_h // 3))
            new_w = max(15, min(new_w, bg_w // 3))

            # 리사이즈
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 위치: 바닥 레벨 (이미지 하단 50~85%)
            max_y = bg_h - new_h
            min_y = int(bg_h * 0.50)
            if min_y >= max_y:
                min_y = max(0, max_y - 10)
            y = random.randint(min_y, max(min_y, max_y))
            x = random.randint(0, max(0, bg_w - new_w))

            # 밝기 매칭: 배경 영역의 평균 밝기에 크롭 맞추기
            bg_region = bg[y:y + new_h, x:x + new_w]
            if bg_region.shape[0] > 0 and bg_region.shape[1] > 0:
                bg_mean = np.mean(bg_region).astype(np.float32)
                crop_mean = np.mean(resized).astype(np.float32)
                if crop_mean > 0:
                    ratio = bg_mean / crop_mean
                    ratio = np.clip(ratio, 0.5, 2.0)
                    resized = np.clip(resized.astype(np.float32) * ratio, 0, 255).astype(np.uint8)

            # Gaussian blur (경계 자연스럽게)
            resized = cv2.GaussianBlur(resized, (3, 3), 0.5)

            # alpha blending (경계 부분 페더링)
            mask = np.ones((new_h, new_w), dtype=np.float32)
            feather = max(2, min(new_w, new_h) // 8)
            for fi in range(feather):
                alpha = (fi + 1) / feather
                mask[fi, :] *= alpha
                mask[-(fi + 1), :] *= alpha
                mask[:, fi] *= alpha
                mask[:, -(fi + 1)] *= alpha
            mask = mask[:, :, np.newaxis]

            # 붙여넣기
            paste_h = min(new_h, bg_h - y)
            paste_w = min(new_w, bg_w - x)
            if paste_h < 10 or paste_w < 10:
                continue

            region = bg[y:y + paste_h, x:x + paste_w].astype(np.float32)
            patch = resized[:paste_h, :paste_w].astype(np.float32)
            m = mask[:paste_h, :paste_w]
            blended = region * (1 - m) + patch * m
            bg[y:y + paste_h, x:x + paste_w] = blended.astype(np.uint8)

            # YOLO 라벨 (normalized)
            cx = (x + paste_w / 2) / bg_w
            cy = (y + paste_h / 2) / bg_h
            nw = paste_w / bg_w
            nh = paste_h / bg_h
            new_bboxes.append((2, cx, cy, nw, nh))

        # 저장
        out_name = f"syn_fallen_{i:04d}.jpg"
        out_img = os.path.join(output_img_dir, out_name)
        out_lbl = os.path.join(output_lbl_dir, out_name.replace(".jpg", ".txt"))

        cv2.imwrite(out_img, bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(out_lbl, 'w') as f:
            for cls, cx, cy, w, h in new_bboxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        generated += 1

        if (i + 1) % 200 == 0:
            print(f"    합성 {i + 1}/{count}...")

    return generated


def prepare():
    """v27 데이터셋 빌드"""
    import cv2  # noqa: F401 - 합성에 필요

    print("=" * 60)
    print("  v27 3-class 데이터셋 빌드")
    print("  Helmet 2,000 : Fallen 2,000 (실제 1K + 합성 1K)")
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

    # ── 2. 실제 fallen (v24 고품질, area 2~20%) ──
    print(f"\n[2/6] 실제 fallen 선별 (area {REAL_MIN_AREA*100:.0f}~{REAL_MAX_AREA*100:.0f}%)...")
    real_fallen = select_real_fallen()
    real_count = 0
    real_bbox_count = 0
    for item in real_fallen:
        src_img = os.path.join(V24_TRAIN_IMG, item['img'])
        dst_img = os.path.join(train_img_dir, item['img'])
        dst_lbl = os.path.join(train_lbl_dir, item['lbl'])
        if not os.path.exists(dst_img):
            os.symlink(src_img, dst_img)
        if not os.path.exists(dst_lbl):
            os.symlink(os.path.join(V24_TRAIN_LBL, item['lbl']), dst_lbl)
        real_count += 1
        real_bbox_count += item['n_bbox']
    print(f"  실제 fallen: {real_count}장 ({real_bbox_count} bbox)")
    if real_fallen:
        print(f"  Area range: {real_fallen[0]['avg_area']:.4f} ~ {real_fallen[-1]['avg_area']:.4f}")

    # ── 3. 합성 fallen 크롭 추출 ──
    print(f"\n[3/6] fallen 크롭 추출 (area > {CROP_MIN_AREA*100:.0f}%)...")
    crops = extract_fallen_crops()
    print(f"  크롭 추출: {len(crops)}개")

    # ── 4. 합성 fallen 생성 ──
    print(f"\n[4/6] 합성 fallen 생성 ({SYNTH_FALLEN}장)...")
    synth_count = generate_synthetic_fallen(
        bg_img_dir=L_TIER_TRAIN_IMG,
        crops=crops,
        output_img_dir=train_img_dir,
        output_lbl_dir=train_lbl_dir,
        count=SYNTH_FALLEN,
    )
    print(f"  합성 fallen: {synth_count}장")

    # ── 5. Helmet val ──
    print("\n[5/6] Helmet val...")
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

    # ── 6. Fallen val ──
    print("\n[6/6] Fallen val...")
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

    total_fallen = real_count + synth_count
    total_train = helmet_count + total_fallen

    print(f"\n{'='*60}")
    print(f"  데이터셋 완성: {OUT_DIR}")
    print(f"  Train: {total_train}장")
    print(f"    Helmet: {helmet_count}장 ({helmet_count/total_train*100:.1f}%)")
    print(f"    Fallen: {total_fallen}장 ({total_fallen/total_train*100:.1f}%)")
    print(f"      실제: {real_count}장, 합성: {synth_count}장")
    print(f"  Val: {val_helmet + val_fallen}장 (helmet {val_helmet} + fallen {val_fallen})")
    print(f"  Ratio: helmet:fallen = 1:{total_fallen/max(helmet_count,1):.2f}")
    print(f"{'='*60}")


def train(batch=4, epochs=100, resume=False):
    """v27 학습"""
    from ultralytics import YOLO

    project = HOBAN
    name = "hoban_go3k_v27"
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
    print("  v27 3-class: 1:1 비율 + 50% 합성 fallen")
    print(f"  Model: yolo26m.pt (COCO pretrained)")
    print(f"  Optimizer: SGD, lr0=0.005, 1280px, batch={batch}")
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

        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.7,
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="v27 3-class (1:1 + 합성 fallen)")
    parser.add_argument("--prepare", action="store_true", help="데이터셋 빌드")
    parser.add_argument("--train", action="store_true", help="학습 시작")
    parser.add_argument("--resume", action="store_true", help="학습 재개")
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
