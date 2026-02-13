"""
v7: aihub(헬멧) + COCO person(사람) + robo(fallen만)
- WiderPerson 대신 COCO person 사용
- 4클래스: helmet_o(0), helmet_x(1), person(2), fallen(3)
"""
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict

AIHUB_BASE = Path(r"D:\task\hoban\datasets_merged")
COCO = Path(r"D:\task\hoban\coco_person")
OUT = Path(r"D:\task\hoban\datasets_v7_sub")
PER_CLASS = 1000
VALID_PER_CLASS = 250

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1. aihub 헬멧 데이터 (class 0, 1) - 클래스별 1000장
    # ========================================
    print("=== 1. aihub 헬멧 데이터 ===")
    for split_src, split_dst, per_cls in [("train", "train", PER_CLASS), ("valid", "valid", VALID_PER_CLASS)]:
        lbl_dir = AIHUB_BASE / split_src / "labels"
        img_dir = AIHUB_BASE / split_src / "images"

        class_imgs = defaultdict(list)
        for lf in lbl_dir.iterdir():
            if lf.suffix != ".txt" or not lf.stem.startswith("aihub"):
                continue
            with open(lf) as f:
                classes = set(int(l.split()[0]) for l in f if l.strip())
            for c in classes:
                if c in [0, 1]:
                    class_imgs[c].append(lf.stem)

        selected = set()
        for c in [0, 1]:
            pool = [s for s in class_imgs[c] if s not in selected]
            random.shuffle(pool)
            pick = pool[:per_cls]
            selected.update(pick)
            print(f"  {split_dst} class {c}: {len(pick)}장 선택")

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
    # 2. COCO person (class 2) - 1000장
    # ========================================
    print("\n=== 2. COCO person 데이터 ===")
    coco_labels = list(COCO.glob("labels/*.txt"))
    random.shuffle(coco_labels)

    # train/valid 분리
    train_pick = coco_labels[:PER_CLASS]
    valid_pick = coco_labels[PER_CLASS:PER_CLASS + VALID_PER_CLASS]

    for split_name, picks in [("train", train_pick), ("valid", valid_pick)]:
        copied = 0
        for lf in picks:
            stem = lf.stem
            img = COCO / "images" / f"{stem}.jpg"
            if not img.exists():
                continue
            shutil.copy2(img, OUT / split_name / "images" / f"coco_{stem}.jpg")
            shutil.copy2(lf, OUT / split_name / "labels" / f"coco_{stem}.txt")
            copied += 1
        print(f"  {split_name}: {copied}장")

    # ========================================
    # 3. robo fallen (class 3) - 1000장
    # ========================================
    print("\n=== 3. robo fallen 데이터 ===")
    for split_src, split_dst, per_cls in [("train", "train", PER_CLASS), ("valid", "valid", VALID_PER_CLASS)]:
        lbl_dir = AIHUB_BASE / split_src / "labels"
        img_dir = AIHUB_BASE / split_src / "images"

        fallen_stems = []
        for lf in lbl_dir.iterdir():
            if lf.suffix != ".txt" or not lf.stem.startswith("robo"):
                continue
            with open(lf) as f:
                lines = [l.strip() for l in f if l.strip()]
            has_fallen = any(int(l.split()[0]) == 3 for l in lines if len(l.split()) == 5)
            if has_fallen:
                fallen_stems.append(lf.stem)

        random.shuffle(fallen_stems)
        picks = fallen_stems[:per_cls]

        copied = 0
        for stem in picks:
            lf = lbl_dir / f"{stem}.txt"
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img = p
                    break
            if not img:
                continue

            # fallen만 추출
            with open(lf) as f:
                fallen_lines = [l.strip() for l in f if l.strip() and len(l.split()) == 5 and int(l.split()[0]) == 3]
            if not fallen_lines:
                continue

            with open(OUT / split_dst / "labels" / f"{stem}.txt", "w") as f:
                f.write("\n".join(fallen_lines) + "\n")
            shutil.copy2(img, OUT / split_dst / "images" / img.name)
            copied += 1
        print(f"  {split_dst}: {copied}장")

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
    # 5. data.yaml
    # ========================================
    yaml_content = """path: D:/task/hoban/datasets_v7_sub
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
