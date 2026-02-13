"""
helmet_60k JSON → YOLO 변환
- helmet_wearing: class "07" → cls 0
- helmet_not_wearing: class "08" → cls 1
- 이미 area 필터링 + 이상치 제거 완료 (summary.json 참조)
- 결과: dataset/helmet_60k_yolo/ (images + labels)
"""
import json
import shutil
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

BASE = Path(r"D:\task\hoban\dataset\helmet_60k\helmet_60k")
OUT = Path(r"D:\task\hoban\dataset\helmet_60k_yolo")

CLS_MAP = {"07": 0, "08": 1}


def convert_folder(folder_name, out_img, out_lbl):
    img_dir = BASE / folder_name / "images"
    lbl_dir = BASE / folder_name / "labels"

    converted = 0
    skipped = 0
    cls_counts = Counter()

    for jf in sorted(lbl_dir.iterdir()):
        if jf.suffix != ".json":
            continue

        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        resolution = data.get("image", {}).get("resolution", [1920, 1080])
        img_w, img_h = resolution[0], resolution[1]
        annotations = data.get("annotations", [])

        lines = []
        for ann in annotations:
            cls_str = ann.get("class", "")
            if cls_str not in CLS_MAP:
                continue
            cls = CLS_MAP[cls_str]

            box = ann.get("box", [])
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box
            cx = max(0, min(1, (x1 + x2) / 2 / img_w))
            cy = max(0, min(1, (y1 + y2) / 2 / img_h))
            bw = max(0, min(1, (x2 - x1) / img_w))
            bh = max(0, min(1, (y2 - y1) / img_h))

            if bw <= 0 or bh <= 0:
                continue

            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            cls_counts[cls] += 1

        if not lines:
            skipped += 1
            continue

        img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = img_dir / f"{jf.stem}{ext}"
            if p.exists():
                img = p
                break
        if not img:
            skipped += 1
            continue

        shutil.copy2(img, out_img / img.name)
        with open(out_lbl / f"{jf.stem}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        converted += 1

    return converted, skipped, cls_counts


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    print("=== helmet_60k → YOLO 변환 ===\n")

    total_cls = Counter()

    for folder, desc in [("helmet_wearing", "착용(cls0)"), ("helmet_not_wearing", "미착용(cls1)")]:
        c, s, cls_c = convert_folder(folder, OUT / "images", OUT / "labels")
        total_cls += cls_c
        print(f"  {desc}: {c}장 변환, {s}장 스킵")
        for k in sorted(cls_c):
            print(f"    cls {k}: {cls_c[k]} bbox")

    print(f"\n=== 총 결과 ===")
    print(f"  이미지: {len(list((OUT/'images').iterdir()))}장")
    for cls in sorted(total_cls):
        print(f"  cls {cls}: {total_cls[cls]} bbox")
    print("\ndone!")


if __name__ == "__main__":
    freeze_support()
    main()
