"""
COCO person pool에서 area >= 2% bbox만 남기고 필터링
- area < 2% bbox 라인 제거
- bbox 0개가 된 이미지는 제외
- 결과: coco_person_filtered/
"""
import shutil
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

SRC_IMG = Path(r"D:\task\hoban\dataset\coco_person\images")
SRC_LBL = Path(r"D:\task\hoban\dataset\coco_person\labels")
OUT = Path(r"D:\task\hoban\coco_person_filtered")
MIN_AREA = 0.02  # 2%


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
    bbox_per_img = []

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
                if area >= MIN_AREA:
                    kept_lines.append(line.strip())
                    kept_bbox += 1

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

        # 복사
        shutil.copy2(img, OUT_IMG / img.name)
        with open(OUT_LBL / lf.name, "w") as f:
            f.write("\n".join(kept_lines) + "\n")

        kept_imgs += 1
        bbox_per_img.append(len(kept_lines))

    print("=== COCO person 필터링 완료 ===")
    print(f"  필터: area >= {MIN_AREA*100:.0f}%")
    print(f"  이미지: {total_imgs} → {kept_imgs} ({total_imgs - kept_imgs}개 제거)")
    print(f"  bbox: {total_bbox} → {kept_bbox} ({total_bbox - kept_bbox}개 제거)")
    if bbox_per_img:
        print(f"  이미지당 bbox: 평균 {sum(bbox_per_img)/len(bbox_per_img):.1f}개, "
              f"최대 {max(bbox_per_img)}개")

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
