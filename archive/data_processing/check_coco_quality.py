"""COCO person 데이터 품질 분석"""
from pathlib import Path
from collections import Counter

COCO = Path(r"D:\task\hoban\dataset\coco_person\labels")
V8 = Path(r"D:\task\hoban\datasets_v8")

def analyze(label_dir, name):
    areas = []
    bbox_per_img = []
    tiny_count = 0  # area < 1%
    small_count = 0  # area < 2%
    medium_count = 0  # 2~10%
    large_count = 0  # > 10%

    for lf in label_dir.iterdir():
        if lf.suffix != ".txt":
            continue
        with open(lf) as f:
            lines = [l.strip() for l in f if l.strip()]

        count = 0
        for line in lines:
            parts = line.split()
            if int(parts[0]) != 2:  # person only
                continue
            w, h = float(parts[3]), float(parts[4])
            area = w * h
            areas.append(area)
            count += 1
            if area < 0.01:
                tiny_count += 1
            elif area < 0.02:
                small_count += 1
            elif area < 0.10:
                medium_count += 1
            else:
                large_count += 1
        if count > 0:
            bbox_per_img.append(count)

    if not areas:
        print(f"  {name}: person bbox 없음")
        return

    areas.sort()
    total = len(areas)
    print(f"\n=== {name} person bbox 분석 ===")
    print(f"  총 bbox: {total}개, 이미지: {len(bbox_per_img)}개")
    print(f"  이미지당 평균 bbox: {sum(bbox_per_img)/len(bbox_per_img):.1f}개")
    print(f"  area 분포:")
    print(f"    tiny  (< 1%):  {tiny_count}개 ({tiny_count/total*100:.1f}%)")
    print(f"    small (1~2%):  {small_count}개 ({small_count/total*100:.1f}%)")
    print(f"    medium(2~10%): {medium_count}개 ({medium_count/total*100:.1f}%)")
    print(f"    large (> 10%): {large_count}개 ({large_count/total*100:.1f}%)")
    print(f"  area percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        idx = int(total * p / 100)
        print(f"    {p}%ile: {areas[idx]:.4f} ({areas[idx]*100:.2f}%)")

    # 640px에서 실제 픽셀 크기 추정
    print(f"\n  640px 기준 실제 크기 추정:")
    for p in [5, 25, 50]:
        idx = int(total * p / 100)
        a = areas[idx]
        px_w = (a ** 0.5) * 640
        print(f"    {p}%ile: ~{px_w:.0f}x{px_w:.0f}px")

# COCO 전체 pool
analyze(COCO, "COCO person pool")

# datasets_v8 train에서 person
analyze(V8 / "train" / "labels", "datasets_v8 train")

# datasets_v8 valid에서 person
analyze(V8 / "valid" / "labels", "datasets_v8 valid")

print("\ndone!")
