"""
fallen_pool에서 극단 bbox 제거
- area < 0.5% 제거 (탐지 불가)
- area > 70% 제거 (이미지 전체 bbox, 잘못된 라벨 가능)
- 이미지당 bbox 5개 초과 제거 (비현실적)
- 결과: fallen_pool_filtered/
"""
import shutil
from pathlib import Path
from multiprocessing import freeze_support

SRC_IMG = Path(r"D:\task\hoban\fallen_pool\images")
SRC_LBL = Path(r"D:\task\hoban\fallen_pool\labels")
OUT = Path(r"D:\task\hoban\fallen_pool_filtered")
MIN_AREA = 0.005   # 0.5%
MAX_AREA = 0.70    # 70%
MAX_BBOX = 5       # 이미지당 최대 bbox


def main():
    OUT_IMG = OUT / "images"
    OUT_LBL = OUT / "labels"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT_IMG.mkdir(parents=True)
    OUT_LBL.mkdir(parents=True)

    total_imgs = 0
    kept_imgs = 0
    total_bbox = 0
    kept_bbox = 0
    removed_tiny = 0
    removed_huge = 0
    removed_overflow = 0

    for lf in sorted(SRC_LBL.iterdir()):
        if lf.suffix != ".txt":
            continue
        total_imgs += 1

        kept_lines = []
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                total_bbox += 1
                w, h = float(parts[3]), float(parts[4])
                area = w * h
                if area < MIN_AREA:
                    removed_tiny += 1
                    continue
                if area > MAX_AREA:
                    removed_huge += 1
                    continue
                kept_lines.append(line.strip())

        # 이미지당 bbox 제한
        if len(kept_lines) > MAX_BBOX:
            removed_overflow += len(kept_lines) - MAX_BBOX
            kept_lines = kept_lines[:MAX_BBOX]

        if not kept_lines:
            continue

        # 이미지 찾기
        img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = SRC_IMG / f"{lf.stem}{ext}"
            if p.exists():
                img = p
                break
        if not img:
            continue

        shutil.copy2(img, OUT_IMG / img.name)
        with open(OUT_LBL / lf.name, "w") as f:
            f.write("\n".join(kept_lines) + "\n")

        kept_imgs += 1
        kept_bbox += len(kept_lines)

    print("=== fallen_pool 필터링 완료 ===")
    print(f"  필터: {MIN_AREA*100:.1f}% <= area <= {MAX_AREA*100:.0f}%, bbox <= {MAX_BBOX}/img")
    print(f"  이미지: {total_imgs} → {kept_imgs} ({total_imgs - kept_imgs}개 제거)")
    print(f"  bbox: {total_bbox} → {kept_bbox}")
    print(f"    제거 - tiny: {removed_tiny}, huge: {removed_huge}, overflow: {removed_overflow}")

    # area 분포 확인
    areas = []
    for lf in OUT_LBL.iterdir():
        if lf.suffix != ".txt":
            continue
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    areas.append(float(parts[3]) * float(parts[4]))
    if areas:
        areas.sort()
        print(f"\n  필터 후 area 분포:")
        for p in [5, 25, 50, 75, 95]:
            idx = int(len(areas) * p / 100)
            px = (areas[idx] ** 0.5) * 640
            print(f"    {p}%ile: {areas[idx]*100:.2f}% (~{px:.0f}px)")

    print("\ndone!")


if __name__ == "__main__":
    freeze_support()
    main()
