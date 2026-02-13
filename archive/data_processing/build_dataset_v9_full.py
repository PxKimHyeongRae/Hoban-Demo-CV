"""
datasets_v9 풀 빌드 (30K bbox/cls)
- helmet_o/x: helmet_60k_yolo (YOLO format, 이미 area 필터링됨)
- person: coco_person_filtered (area >= 2%)
- fallen: fallen_pool_filtered (0.5% <= area <= 70%)
- negative: 빈 라벨 이미지 1K
- 클래스당 30,000 bbox 균형 (fallen은 최대 ~27K)
"""
import shutil
import random
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

# === 설정 ===
TARGET_BBOX = 30000
VALID_RATIO = 0.2
SEED = 42
OUT = Path(r"D:\task\hoban\datasets_v9")

# 소스 경로
HELMET_DIR = Path(r"D:\task\hoban\dataset\helmet_60k_yolo")
PERSON_DIR = Path(r"D:\task\hoban\dataset\coco_person_filtered")
FALLEN_DIR = Path(r"D:\task\hoban\dataset\fallen_pool_filtered")
NEG_DIR = Path(r"D:\task\hoban\dataset\negative_samples")


def scan_labels(lbl_dir):
    """라벨 스캔 → {stem: {cls: count}}"""
    file_cls = {}
    for lf in lbl_dir.iterdir():
        if lf.suffix != ".txt":
            continue
        cls_counts = Counter()
        with open(lf) as f:
            for line in f:
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1
        if cls_counts:
            file_cls[lf.stem] = cls_counts
    return file_cls


def select_by_bbox(file_cls_map, target_cls, target_bbox):
    """target_cls bbox가 target_bbox 이상이 될 때까지 이미지 선택"""
    candidates = list(file_cls_map.items())
    random.shuffle(candidates)
    selected = []
    bbox_sum = 0
    for stem, cls_counts in candidates:
        if target_cls not in cls_counts:
            continue
        selected.append(stem)
        bbox_sum += cls_counts[target_cls]
        if bbox_sum >= target_bbox:
            break
    return selected, bbox_sum


def find_image(img_dir, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def copy_files(stems, src_img, src_lbl, dst_img, dst_lbl, prefix="", cls_remap=None):
    """파일 복사. cls_remap: {src_cls: dst_cls}"""
    copied = 0
    for stem in stems:
        img = find_image(src_img, stem)
        lf = src_lbl / f"{stem}.txt"
        if not img or not lf.exists():
            continue

        new_stem = f"{prefix}{stem}" if prefix else stem

        if cls_remap:
            with open(lf) as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in cls_remap:
                    parts[0] = str(cls_remap[cls])
                    new_lines.append(" ".join(parts))
            with open(dst_lbl / f"{new_stem}.txt", "w") as f:
                f.write("\n".join(new_lines) + "\n")
        else:
            shutil.copy2(lf, dst_lbl / f"{new_stem}.txt")

        shutil.copy2(img, dst_img / f"{new_stem}{img.suffix}")
        copied += 1
    return copied


def main():
    random.seed(SEED)

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)

    print("=== datasets_v9 풀 빌드 (30K bbox/cls) ===\n")

    # === 1. Helmet (cls 0, 1) ===
    print("[1] Helmet 데이터 (helmet_60k_yolo)")
    helmet_cls = scan_labels(HELMET_DIR / "labels")
    print(f"    스캔: {len(helmet_cls)}장")

    h0_stems, h0_bbox = select_by_bbox(helmet_cls, 0, TARGET_BBOX)
    h1_stems, h1_bbox = select_by_bbox(helmet_cls, 1, TARGET_BBOX)
    helmet_stems = list(set(h0_stems) | set(h1_stems))
    random.shuffle(helmet_stems)
    split = int(len(helmet_stems) * (1 - VALID_RATIO))
    h_train, h_valid = helmet_stems[:split], helmet_stems[split:]

    ct = copy_files(h_train, HELMET_DIR / "images", HELMET_DIR / "labels",
                    OUT / "train/images", OUT / "train/labels")
    cv = copy_files(h_valid, HELMET_DIR / "images", HELMET_DIR / "labels",
                    OUT / "valid/images", OUT / "valid/labels")
    print(f"    cls0 선택: {len(h0_stems)}장 / {h0_bbox} bbox")
    print(f"    cls1 선택: {len(h1_stems)}장 / {h1_bbox} bbox")
    print(f"    복사: train {ct} / valid {cv}")

    # === 2. Person (cls 2) ===
    print("\n[2] Person 데이터 (COCO filtered)")
    person_cls = scan_labels(PERSON_DIR / "labels")
    print(f"    스캔: {len(person_cls)}장")

    p_stems, p_bbox = select_by_bbox(person_cls, 2, TARGET_BBOX)
    random.shuffle(p_stems)
    split = int(len(p_stems) * (1 - VALID_RATIO))
    p_train, p_valid = p_stems[:split], p_stems[split:]

    ct = copy_files(p_train, PERSON_DIR / "images", PERSON_DIR / "labels",
                    OUT / "train/images", OUT / "train/labels", prefix="coco_")
    cv = copy_files(p_valid, PERSON_DIR / "images", PERSON_DIR / "labels",
                    OUT / "valid/images", OUT / "valid/labels", prefix="coco_")
    print(f"    선택: {len(p_stems)}장 / {p_bbox} bbox")
    print(f"    복사: train {ct} / valid {cv}")

    # === 3. Fallen (cls 3, remapped from 0) ===
    print("\n[3] Fallen 데이터 (filtered, 전량 사용)")
    fallen_cls = scan_labels(FALLEN_DIR / "labels")
    print(f"    스캔: {len(fallen_cls)}장")

    # fallen은 27K로 30K 미달 → 전량 사용
    f_stems, f_bbox = select_by_bbox(fallen_cls, 0, TARGET_BBOX)
    random.shuffle(f_stems)
    split = int(len(f_stems) * (1 - VALID_RATIO))
    f_train, f_valid = f_stems[:split], f_stems[split:]

    ct = copy_files(f_train, FALLEN_DIR / "images", FALLEN_DIR / "labels",
                    OUT / "train/images", OUT / "train/labels",
                    prefix="fallen_", cls_remap={0: 3})
    cv = copy_files(f_valid, FALLEN_DIR / "images", FALLEN_DIR / "labels",
                    OUT / "valid/images", OUT / "valid/labels",
                    prefix="fallen_", cls_remap={0: 3})
    print(f"    선택: {len(f_stems)}장 / {f_bbox} bbox (cls 0→3 remap)")
    print(f"    복사: train {ct} / valid {cv}")

    # === 4. Negative samples ===
    print("\n[4] Negative samples (빈 라벨)")
    neg_count = 0
    if NEG_DIR.exists() and (NEG_DIR / "images").exists():
        neg_imgs = list((NEG_DIR / "images").iterdir())
        random.shuffle(neg_imgs)
        split = int(len(neg_imgs) * (1 - VALID_RATIO))
        for i, img in enumerate(neg_imgs):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            target = "train" if i < split else "valid"
            shutil.copy2(img, OUT / target / "images" / f"neg_{img.name}")
            with open(OUT / target / "labels" / f"neg_{img.stem}.txt", "w") as f:
                pass
            neg_count += 1
        print(f"    추가: {neg_count}장 (train {min(split, neg_count)} / valid {max(0, neg_count - split)})")
    else:
        print(f"    {NEG_DIR} 없음 - 건너뜀")

    # === 5. data.yaml ===
    yaml_content = """path: datasets_v9
train: train/images
val: valid/images
nc: 4
names:
  0: person_with_helmet
  1: person_without_helmet
  2: person
  3: fallen
"""
    with open(OUT / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # === 6. 최종 통계 ===
    print("\n=== 최종 클래스 분포 ===")
    names = {0: "helmet_o", 1: "helmet_x", 2: "person", 3: "fallen"}
    for sn in ["train", "valid"]:
        counter = Counter()
        img_count = len(list((OUT / sn / "images").iterdir()))
        for lf in (OUT / sn / "labels").iterdir():
            if lf.suffix != ".txt":
                continue
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        counter[int(line.split()[0])] += 1
        total_bbox = sum(counter.values())
        print(f"  {sn} ({img_count}장, {total_bbox} bbox):")
        for cls in sorted(counter):
            print(f"    {cls} ({names.get(cls, '?')}): {counter[cls]} bbox")
        empty = sum(1 for lf in (OUT / sn / "labels").iterdir()
                    if lf.suffix == ".txt" and lf.stat().st_size == 0)
        if empty:
            print(f"    negative (빈 라벨): {empty}장")

    print("\ndone!")


if __name__ == "__main__":
    freeze_support()
    main()
