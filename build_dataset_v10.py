"""
datasets_v10 빌드 (v9 대비 개선점)
1. helmet 이미지에 person auto-label 추가 (helmet_60k_labeled)
2. negative samples 10K (helmet_negative_10k)
3. 클래스당 30K bbox 균형

소스:
- helmet_60k_labeled: cls 0(helmet_o) + cls 1(helmet_x) + cls 2(person auto)
- coco_person_filtered: cls 2(person)
- fallen_pool_filtered: cls 0→3 remap (fallen)
- helmet_negative_10k: 빈 라벨 (aihub JSON, images only)
- negative_samples: 빈 라벨 1K (기존)
"""
import json
import shutil
import random
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

# === 설정 ===
TARGET_BBOX = 30000
VALID_RATIO = 0.2
SEED = 42
OUT = Path(r"D:\task\hoban\datasets_v10")

HELMET_DIR = Path(r"D:\task\hoban\dataset\helmet_60k_labeled")
PERSON_DIR = Path(r"D:\task\hoban\dataset\coco_person_filtered")
FALLEN_DIR = Path(r"D:\task\hoban\dataset\fallen_pool_filtered")
NEG_AIHUB = Path(r"D:\task\hoban\dataset\helmet_negative_10k")
NEG_OLD = Path(r"D:\task\hoban\dataset\negative_samples")


def scan_labels(lbl_dir):
    """라벨 스캔 → {stem: {cls: count}}"""
    result = {}
    for lf in lbl_dir.iterdir():
        if lf.suffix != ".txt":
            continue
        cls_counts = Counter()
        with open(lf) as f:
            for line in f:
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1
        if cls_counts:
            result[lf.stem] = cls_counts
    return result


def select_by_bbox(file_cls_map, target_cls, target_bbox):
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


def copy_files(stems, src_img, src_lbl, dst_img, dst_lbl,
               prefix="", cls_remap=None):
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

    print("=== datasets_v10 빌드 ===\n")

    # === 1. Helmet + Person auto-label (cls 0, 1, 2) ===
    print("[1] Helmet + Person auto-label (helmet_60k_labeled)")
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

    # helmet 데이터 내 person bbox 수 집계
    h_person_bbox = sum(
        helmet_cls[s].get(2, 0) for s in helmet_stems if s in helmet_cls)
    print(f"    cls0 선택: {len(h0_stems)}장 / {h0_bbox} bbox")
    print(f"    cls1 선택: {len(h1_stems)}장 / {h1_bbox} bbox")
    print(f"    cls2 (person auto): {h_person_bbox} bbox")
    print(f"    복사: train {ct} / valid {cv}")

    # === 2. COCO Person (cls 2) — helmet에서 부족한 만큼 보충 ===
    print(f"\n[2] COCO Person (coco_person_filtered)")
    person_cls = scan_labels(PERSON_DIR / "labels")
    print(f"    스캔: {len(person_cls)}장")

    person_need = max(0, TARGET_BBOX - h_person_bbox)
    if person_need > 0:
        p_stems, p_bbox = select_by_bbox(person_cls, 2, person_need)
    else:
        p_stems, p_bbox = [], 0

    random.shuffle(p_stems)
    split = int(len(p_stems) * (1 - VALID_RATIO))
    p_train, p_valid = p_stems[:split], p_stems[split:]

    ct = copy_files(p_train, PERSON_DIR / "images", PERSON_DIR / "labels",
                    OUT / "train/images", OUT / "train/labels", prefix="coco_")
    cv = copy_files(p_valid, PERSON_DIR / "images", PERSON_DIR / "labels",
                    OUT / "valid/images", OUT / "valid/labels", prefix="coco_")
    print(f"    helmet person auto: {h_person_bbox} bbox")
    print(f"    COCO 보충 필요: {person_need} bbox")
    print(f"    COCO 선택: {len(p_stems)}장 / {p_bbox} bbox")
    print(f"    복사: train {ct} / valid {cv}")

    # === 3. Fallen (cls 3, remapped from 0) ===
    print(f"\n[3] Fallen (fallen_pool_filtered, 전량 사용)")
    fallen_cls = scan_labels(FALLEN_DIR / "labels")
    print(f"    스캔: {len(fallen_cls)}장")

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
    print(f"    선택: {len(f_stems)}장 / {f_bbox} bbox (cls 0→3)")
    print(f"    복사: train {ct} / valid {cv}")

    # === 4. Negative samples (aihub 10K + 기존 1K) ===
    print(f"\n[4] Negative samples")

    neg_count = 0

    # 4a) aihub helmet_negative_10k (JSON → 빈 라벨)
    if NEG_AIHUB.exists() and (NEG_AIHUB / "images").exists():
        neg_imgs = sorted((NEG_AIHUB / "images").iterdir())
        random.shuffle(neg_imgs)
        split = int(len(neg_imgs) * (1 - VALID_RATIO))
        for i, img in enumerate(neg_imgs):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            target = "train" if i < split else "valid"
            shutil.copy2(img, OUT / target / "images" / f"neg10k_{img.name}")
            with open(OUT / target / "labels" / f"neg10k_{img.stem}.txt", "w"):
                pass
            neg_count += 1
        print(f"    aihub negative: {neg_count}장")

    # 4b) 기존 negative_samples 1K
    old_neg = 0
    if NEG_OLD.exists() and (NEG_OLD / "images").exists():
        old_imgs = list((NEG_OLD / "images").iterdir())
        random.shuffle(old_imgs)
        split = int(len(old_imgs) * (1 - VALID_RATIO))
        for i, img in enumerate(old_imgs):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            target = "train" if i < split else "valid"
            shutil.copy2(img, OUT / target / "images" / f"neg1k_{img.name}")
            with open(OUT / target / "labels" / f"neg1k_{img.stem}.txt", "w"):
                pass
            old_neg += 1
        print(f"    기존 negative: {old_neg}장")

    neg_total = neg_count + old_neg
    print(f"    총 negative: {neg_total}장")

    # === 5. data.yaml ===
    yaml_content = """path: datasets_v10
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
    print(f"\n=== 최종 클래스 분포 ===")
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
        empty = sum(1 for lf in (OUT / sn / "labels").iterdir()
                    if lf.suffix == ".txt" and lf.stat().st_size == 0)
        print(f"  {sn} ({img_count}장, {total_bbox} bbox):")
        for cls in sorted(counter):
            print(f"    {cls} ({names.get(cls, '?')}): {counter[cls]} bbox")
        if empty:
            print(f"    negative (빈 라벨): {empty}장")

    print("\ndone!")


if __name__ == "__main__":
    freeze_support()
    main()
