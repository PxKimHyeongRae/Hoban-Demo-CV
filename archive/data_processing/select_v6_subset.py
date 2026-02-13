"""
v6 데이터셋에서 클래스별 1000장 서브셋 추출
- 각 클래스가 포함된 이미지를 1000장씩 선별
- 중복 이미지는 한번만 포함
"""
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path(r"D:\task\hoban\datasets_v6")
OUT = Path(r"D:\task\hoban\datasets_v6_sub")
PER_CLASS = 1000

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid"]:
        per_class = PER_CLASS if split == "train" else 250

        lbl_dir = SRC / split / "labels"
        img_dir = SRC / split / "images"

        # 클래스별 이미지 분류
        class_images = defaultdict(list)
        for lf in lbl_dir.iterdir():
            if lf.suffix != ".txt":
                continue
            with open(lf) as f:
                classes = set()
                for line in f:
                    if line.strip():
                        classes.add(int(line.split()[0]))
            for cls in classes:
                class_images[cls].append(lf.stem)

        print(f"\n=== {split} 클래스별 이미지 수 ===")
        for cls in sorted(class_images):
            print(f"  class {cls}: {len(class_images[cls])}장")

        # 클래스별 선택
        selected = set()
        for cls in sorted(class_images):
            pool = [s for s in class_images[cls] if s not in selected]
            random.shuffle(pool)
            pick = pool[:per_class]
            selected.update(pick)
            print(f"  class {cls}: {len(pick)}장 선택")

        # 복사
        copied = 0
        for stem in selected:
            lbl = lbl_dir / f"{stem}.txt"
            if not lbl.exists():
                continue
            # 이미지 찾기
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img = p
                    break
            if not img:
                continue
            shutil.copy2(img, OUT / split / "images" / img.name)
            shutil.copy2(lbl, OUT / split / "labels" / lbl.name)
            copied += 1

        print(f"  총 복사: {copied}장")

        # 최종 분포
        counter = Counter()
        for lf in (OUT / split / "labels").iterdir():
            if lf.suffix != ".txt":
                continue
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        counter[int(line.split()[0])] += 1
        names = {0: "helmet_o", 1: "helmet_x", 2: "person", 3: "fallen"}
        print(f"  bbox 분포:")
        for cls in sorted(counter):
            print(f"    {cls} ({names.get(cls,'?')}): {counter[cls]}")

    # data.yaml
    yaml_content = """path: D:/task/hoban/datasets_v6_sub
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
