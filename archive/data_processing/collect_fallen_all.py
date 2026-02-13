"""
8개 fall 폴더에서 fallen bbox만 수집하여 fallen_pool/에 통합
- 각 소스의 fallen 클래스만 추출, class 0으로 통일
- train + valid + test 모두 수집 (데이터 최대화)
- prefix로 소스 구분
"""
import shutil
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

FALL_BASE = Path(r"D:\task\dataset")
OUT = Path(r"D:\task\hoban\fallen_pool")

# 소스 정의: (폴더명, prefix, {원본class: 포함여부})
SOURCES = [
    {
        "name": "fall detection ip camera.v3i.yolo26",
        "prefix": "fall1",
        "fallen_classes": {0},  # class 0 = fall
    },
    {
        "name": "Fall Detection.v4-resized640_aug3x-accurate.yolo26",
        "prefix": "fall2",
        "fallen_classes": {0},  # class 0 = Fall-Detected
    },
    {
        "name": "Fall.v1i.yolo26",
        "prefix": "fall3",
        "fallen_classes": {1},  # class 1 = down (class 0 = "10-" 제외)
    },
    {
        "name": "fall.v1i.yolo26 (2)",
        "prefix": "fall4",
        "fallen_classes": {0},  # class 0 = fall
    },
    {
        "name": "fall.v2i.yolo26",
        "prefix": "fall5",
        "fallen_classes": {0},  # class 0 = fall
    },
    {
        "name": "Fall.v3i.yolo26",
        "prefix": "fall6",
        "fallen_classes": {0, 1},  # class 0 = Fall-person, class 1 = fall
    },
    {
        "name": "fall.v4i.yolo26",
        "prefix": "fall7",
        "fallen_classes": {0},  # class 0 = falling
    },
    {
        "name": "fallen.v2i.yolo26",
        "prefix": "fall8",
        "fallen_classes": {1},  # class 1 = 2_fallen
    },
]


def collect_source(src_dir, prefix, fallen_classes):
    """하나의 소스에서 fallen bbox 추출"""
    collected = 0
    bbox_count = 0
    empty_skip = 0

    for split in ["train", "valid", "test"]:
        img_dir = src_dir / split / "images"
        lbl_dir = src_dir / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.iterdir():
            if lbl_file.suffix != ".txt":
                continue

            # 라벨에서 fallen 클래스만 추출
            fallen_lines = []
            with open(lbl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id in fallen_classes:
                        # class 0으로 통일
                        fallen_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

            if not fallen_lines:
                empty_skip += 1
                continue

            # 이미지 찾기
            stem = lbl_file.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
            if not img_path:
                continue

            # 저장
            out_name = f"{prefix}_{split}_{stem}"
            shutil.copy2(img_path, OUT / "images" / f"{out_name}{img_path.suffix}")
            with open(OUT / "labels" / f"{out_name}.txt", "w") as f:
                f.write("\n".join(fallen_lines) + "\n")

            collected += 1
            bbox_count += len(fallen_lines)

    return collected, bbox_count, empty_skip


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    total_imgs = 0
    total_bbox = 0

    print("=== fallen 데이터 수집 ===\n")
    for src in SOURCES:
        src_dir = FALL_BASE / src["name"]
        if not src_dir.exists():
            print(f"  [SKIP] {src['name']} - 폴더 없음")
            continue

        imgs, bbox, skipped = collect_source(src_dir, src["prefix"], src["fallen_classes"])
        total_imgs += imgs
        total_bbox += bbox
        print(f"  [{src['prefix']}] {src['name']}")
        print(f"    수집: {imgs}장, {bbox} bbox (스킵: {skipped})")

    # 최종 통계
    print(f"\n=== 수집 완료 ===")
    print(f"총 이미지: {total_imgs}장")
    print(f"총 fallen bbox: {total_bbox}개")

    # 검증: 실제 파일 수
    actual_imgs = len(list((OUT / "images").iterdir()))
    actual_lbls = len(list((OUT / "labels").iterdir()))
    print(f"실제 파일: images={actual_imgs}, labels={actual_lbls}")
    print("done!")


if __name__ == "__main__":
    freeze_support()
    main()
