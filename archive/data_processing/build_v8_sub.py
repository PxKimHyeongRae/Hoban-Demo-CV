"""
v8 서브셋: datasets_v8에서 클래스당 3,000 bbox 추출
로컬 30에폭 실험용
"""
import shutil
import random
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

SRC = Path(r"D:\task\hoban\datasets_v8")
OUT = Path(r"D:\task\hoban\datasets_v8_sub")
TARGET_BBOX = 3000
VALID_RATIO = 0.2


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


def scan_labels(lbl_dir):
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


def copy_split(src_split, out_split, target):
    file_cls = scan_labels(SRC / src_split / "labels")
    print(f"\n  {src_split} 스캔: {len(file_cls)}장")

    selected = set()
    for cls in range(4):
        sel, bbox = select_by_bbox(file_cls, cls, target)
        selected |= set(sel)
        print(f"    cls{cls}: {len(sel)}장 / {bbox} bbox")

    copied = 0
    for stem in selected:
        # 이미지 찾기
        img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = SRC / src_split / "images" / f"{stem}{ext}"
            if p.exists():
                img = p
                break
        lf = SRC / src_split / "labels" / f"{stem}.txt"
        if not img or not lf.exists():
            continue

        shutil.copy2(img, OUT / out_split / "images" / f"{stem}{img.suffix}")
        shutil.copy2(lf, OUT / out_split / "labels" / f"{stem}.txt")
        copied += 1

    return copied


def main():
    random.seed(42)

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)

    print("=== v8 서브셋 빌드 (3K bbox/cls) ===")

    train_count = copy_split("train", "train", TARGET_BBOX)
    print(f"\n  train 복사: {train_count}장")

    valid_count = copy_split("valid", "valid", int(TARGET_BBOX * VALID_RATIO))
    print(f"\n  valid 복사: {valid_count}장")

    # data.yaml
    yaml_content = """path: datasets_v8_sub
train: train/images
val: valid/images
nc: 4
names:
  0: person_with_helmet
  1: person_without_helmet
  2: person
  3: fallen
"""
    with open(OUT / "data.yaml", "w") as f:
        f.write(yaml_content)

    # 최종 통계
    print("\n=== 최종 클래스 분포 ===")
    names = {0: "helmet_o", 1: "helmet_x", 2: "person", 3: "fallen"}
    for sn in ["train", "valid"]:
        counter = Counter()
        total = 0
        for lf in (OUT / sn / "labels").iterdir():
            if lf.suffix != ".txt":
                continue
            total += 1
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        counter[int(line.split()[0])] += 1
        print(f"  {sn} ({total}장):")
        for cls in sorted(counter):
            print(f"    {cls} ({names.get(cls, '?')}): {counter[cls]} bbox")

    print("\ndone!")


if __name__ == "__main__":
    freeze_support()
    main()
