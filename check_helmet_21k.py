"""helmet_21k 데이터 확인 - 상위 0.1% 이상치 체크 + 전체 분포"""
from pathlib import Path
from collections import Counter

BASE = Path(r"D:\task\hoban\dataset\helmet_21k")

def analyze(name, cls_filter=None):
    lbl_dir = BASE / name / "labels"
    img_dir = BASE / name / "images"

    if not lbl_dir.exists():
        print(f"  {name}: labels 폴더 없음")
        return

    imgs = len(list(img_dir.iterdir())) if img_dir.exists() else 0

    areas = []
    cls_counter = Counter()
    for lf in lbl_dir.iterdir():
        if lf.suffix != ".txt":
            continue
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                cls_counter[cls] += 1
                w, h = float(parts[3]), float(parts[4])
                areas.append((w * h, w, h, lf.stem, cls))

    areas.sort(key=lambda x: -x[0])  # 큰 순
    total = len(areas)

    print(f"\n=== {name} ===")
    print(f"  이미지: {imgs}장, bbox: {total}개")
    print(f"  클래스 분포: {dict(cls_counter)}")

    if not areas:
        return

    # 상위 0.1% (이상치 후보)
    top_n = max(1, int(total * 0.001))
    print(f"\n  상위 0.1% ({top_n}개) - 이상치 후보:")
    for i, (area, w, h, stem, cls) in enumerate(areas[:min(top_n, 20)]):
        print(f"    {i+1}. area={area*100:.2f}% w={w:.4f} h={h:.4f} cls={cls} | {stem[:50]}")

    # 전체 분포
    areas_only = [a[0] for a in areas]
    areas_only.sort()
    print(f"\n  전체 area 분포 (작은→큰):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int(total * p / 100)
        a = areas_only[idx]
        px = (a ** 0.5) * 640
        print(f"    {p:>2}%ile: {a*100:.3f}% (~{px:.0f}px)")

for name in ["helmet_wearing", "helmet_not_wearing", "background"]:
    analyze(name)

print("\ndone!")
