"""
v7 풀 데이터셋: aihub(헬멧) + COCO person(사람) + robo(fallen만)
- 4클래스: helmet_o(0), helmet_x(1), person(2), fallen(3)
- 클래스별 bbox 10,000개 기준 균형
"""
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict

AIHUB_BASE = Path(r"D:\task\hoban\datasets_merged")
COCO = Path(r"D:\task\hoban\coco_person")
OUT = Path(r"D:\task\hoban\datasets_v7")
TARGET_BBOX = 10000
VALID_RATIO = 0.2  # valid = train의 20%

def select_by_bbox(file_class_map, target_cls, target_bbox):
    """이미지를 셔플 후 target_cls의 bbox가 target_bbox에 도달할 때까지 선택"""
    candidates = list(file_class_map.items())
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

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1. aihub 헬멧 데이터 - bbox 기준 만개씩
    # ========================================
    print("=== 1. aihub 헬멧 데이터 (bbox 기준 만개) ===")
    for split_src, split_dst, target in [("train", "train", TARGET_BBOX), ("valid", "valid", int(TARGET_BBOX * VALID_RATIO))]:
        lbl_dir = AIHUB_BASE / split_src / "labels"
        img_dir = AIHUB_BASE / split_src / "images"

        # 각 파일별 클래스별 bbox 수 집계
        file_cls = {}
        for lf in lbl_dir.iterdir():
            if lf.suffix != ".txt" or not lf.stem.startswith("aihub"):
                continue
            cls_counts = Counter()
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        cls_counts[int(line.split()[0])] += 1
            if cls_counts:
                file_cls[lf.stem] = cls_counts

        # class 0, class 1 각각 bbox 기준 선택
        sel_0, bbox_0 = select_by_bbox(file_cls, 0, target)
        sel_1, bbox_1 = select_by_bbox(file_cls, 1, target)
        selected = set(sel_0) | set(sel_1)
        print(f"  {split_dst} class 0: {len(sel_0)}장 ({bbox_0} bbox)")
        print(f"  {split_dst} class 1: {len(sel_1)}장 ({bbox_1} bbox)")

        copied = 0
        for stem in selected:
            lf = lbl_dir / f"{stem}.txt"
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img = p
                    break
            if not img or not lf.exists():
                continue
            shutil.copy2(img, OUT / split_dst / "images" / img.name)
            shutil.copy2(lf, OUT / split_dst / "labels" / lf.name)
            copied += 1
        print(f"  {split_dst} aihub 복사: {copied}장")

    # ========================================
    # 2. COCO person (class 2) - bbox 기준 만개
    # ========================================
    print("\n=== 2. COCO person 데이터 (bbox 기준 만개) ===")
    coco_labels = list(COCO.glob("labels/*.txt"))
    random.shuffle(coco_labels)

    for split_name, target in [("train", TARGET_BBOX), ("valid", int(TARGET_BBOX * VALID_RATIO))]:
        copied = 0
        bbox_sum = 0
        for lf in coco_labels:
            if bbox_sum >= target:
                break
            stem = lf.stem
            img = COCO / "images" / f"{stem}.jpg"
            if not img.exists():
                continue
            with open(lf) as f:
                n_bbox = sum(1 for line in f if line.strip())
            shutil.copy2(img, OUT / split_name / "images" / f"coco_{stem}.jpg")
            shutil.copy2(lf, OUT / split_name / "labels" / f"coco_{stem}.txt")
            bbox_sum += n_bbox
            copied += 1
        # 사용한 라벨 제거 (train/valid 중복 방지)
        coco_labels = coco_labels[copied:]
        print(f"  {split_name}: {copied}장 ({bbox_sum} bbox)")

    # ========================================
    # 3. robo fallen (class 3) - bbox 기준 만개
    # ========================================
    print("\n=== 3. robo fallen 데이터 (bbox 기준 만개) ===")
    for split_src, split_dst, target in [("train", "train", TARGET_BBOX), ("valid", "valid", int(TARGET_BBOX * VALID_RATIO))]:
        lbl_dir = AIHUB_BASE / split_src / "labels"
        img_dir = AIHUB_BASE / split_src / "images"

        candidates = []
        for lf in lbl_dir.iterdir():
            if lf.suffix != ".txt" or not lf.stem.startswith("robo"):
                continue
            with open(lf) as f:
                lines = [l.strip() for l in f if l.strip()]
            fallen_lines = [l for l in lines if len(l.split()) == 5 and int(l.split()[0]) == 3]
            if fallen_lines:
                candidates.append((lf.stem, fallen_lines))

        random.shuffle(candidates)
        copied = 0
        bbox_sum = 0
        for stem, fallen_lines in candidates:
            if bbox_sum >= target:
                break
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img = p
                    break
            if not img:
                continue

            with open(OUT / split_dst / "labels" / f"{stem}.txt", "w") as f:
                f.write("\n".join(fallen_lines) + "\n")
            shutil.copy2(img, OUT / split_dst / "images" / img.name)
            bbox_sum += len(fallen_lines)
            copied += 1
        print(f"  {split_dst}: {copied}장 ({bbox_sum} bbox)")

    # ========================================
    # 4. 클래스 분포 확인
    # ========================================
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
            print(f"    {cls} ({names.get(cls,'?')}): {counter[cls]} bbox")

    # ========================================
    # 5. data.yaml (상대경로 - 서버용)
    # ========================================
    yaml_content = """path: datasets_v7
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
    print("\ndone!")

if __name__ == "__main__":
    main()
