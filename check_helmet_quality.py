"""helmet_pool bbox 크기 분포 분석"""
from pathlib import Path
from collections import Counter

POOL = Path(r"D:\task\hoban\helmet_pool\labels")

def analyze_cls(labels, cls_id, name):
    areas = []
    for lf in labels:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == cls_id:
                    w, h = float(parts[3]), float(parts[4])
                    areas.append(w * h)

    if not areas:
        return

    areas.sort()
    total = len(areas)
    tiny = sum(1 for a in areas if a < 0.005)  # < 0.5%
    small = sum(1 for a in areas if 0.005 <= a < 0.01)  # 0.5~1%
    ok = sum(1 for a in areas if 0.01 <= a < 0.02)  # 1~2%
    medium = sum(1 for a in areas if 0.02 <= a < 0.10)  # 2~10%
    large = sum(1 for a in areas if a >= 0.10)  # > 10%

    print(f"\n=== cls {cls_id} ({name}) - {total}개 bbox ===")
    print(f"  area < 0.5% (탐지불가):    {tiny:>6}개 ({tiny/total*100:.1f}%)")
    print(f"  area 0.5~1% (매우 작음):   {small:>6}개 ({small/total*100:.1f}%)")
    print(f"  area 1~2% (작음):          {ok:>6}개 ({ok/total*100:.1f}%)")
    print(f"  area 2~10% (보통):         {medium:>6}개 ({medium/total*100:.1f}%)")
    print(f"  area > 10% (양호):         {large:>6}개 ({large/total*100:.1f}%)")

    for p in [5, 10, 25, 50, 75, 90, 95]:
        idx = int(total * p / 100)
        px = (areas[idx] ** 0.5) * 640
        print(f"  {p:>2}%ile: area={areas[idx]:.4f} (~{px:.0f}px)")

labels = [l for l in POOL.iterdir() if l.suffix == '.txt']
print(f"총 라벨 파일: {len(labels)}개")

analyze_cls(labels, 0, "helmet_o")
analyze_cls(labels, 1, "helmet_x")

print("\ndone!")
