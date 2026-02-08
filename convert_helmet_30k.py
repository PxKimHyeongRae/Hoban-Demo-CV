"""
helmet_30k JSON → YOLO txt 변환
- JSON box [x1,y1,x2,y2] pixel → YOLO [x_center, y_center, w, h] normalized
- class "07" (wearing) → 0 (helmet_o)
- class "08" (not_wearing) → 1 (helmet_x)
- 출력: helmet_pool/images, helmet_pool/labels
"""
import json
import shutil
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

SRC = Path(r"D:\task\hoban\dataset\helmet_30k")
OUT = Path(r"D:\task\hoban\helmet_pool")

CLASS_MAP = {"07": 0, "08": 1}  # wearing→0, not_wearing→1
FOLDER_CLASS = {
    "helmet_wearing": 0,
    "helmet_not_wearing": 1,
}


def convert_json_to_yolo(json_path, img_w, img_h):
    """JSON annotation → YOLO lines"""
    data = json.load(open(json_path, encoding="utf-8"))
    lines = []
    for ann in data.get("annotations", []):
        cls_str = ann.get("class", "")
        if cls_str not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_str]
        box = ann["box"]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        # normalize
        x_center = (x1 + x2) / 2.0 / img_w
        y_center = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        # clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    stats = Counter()
    bbox_stats = Counter()
    errors = 0

    for split in ["training", "validation"]:
        for cls_folder in ["helmet_wearing", "helmet_not_wearing"]:
            img_dir = SRC / split / cls_folder / "images"
            lbl_dir = SRC / split / cls_folder / "labels"
            if not img_dir.exists():
                continue

            prefix = f"{split[:5]}_{cls_folder[:3]}_"  # e.g. "train_hel_", "valid_hel_"

            for json_file in sorted(lbl_dir.iterdir()):
                if json_file.suffix != ".json":
                    continue
                stem = json_file.stem

                # find image
                img_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    p = img_dir / f"{stem}{ext}"
                    if p.exists():
                        img_path = p
                        break
                if not img_path:
                    errors += 1
                    continue

                # get resolution from JSON
                try:
                    data = json.load(open(json_file, encoding="utf-8"))
                    res = data.get("image", {}).get("resolution", None)
                    if res:
                        img_w, img_h = res[0], res[1]
                    else:
                        from PIL import Image
                        with Image.open(img_path) as im:
                            img_w, img_h = im.size
                except Exception:
                    errors += 1
                    continue

                # convert
                yolo_lines = convert_json_to_yolo(json_file, img_w, img_h)
                if not yolo_lines:
                    errors += 1
                    continue

                # save
                out_name = f"{prefix}{stem}"
                shutil.copy2(img_path, OUT / "images" / f"{out_name}{img_path.suffix}")
                with open(OUT / "labels" / f"{out_name}.txt", "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")

                for line in yolo_lines:
                    bbox_stats[int(line.split()[0])] += 1
                stats[f"{split}/{cls_folder}"] += 1

    # 통계 출력
    print("=== helmet_30k → YOLO 변환 완료 ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}장")
    print(f"\n총 이미지: {sum(stats.values())}장")
    print(f"총 bbox: {sum(bbox_stats.values())}개")
    for cls in sorted(bbox_stats):
        name = {0: "helmet_o", 1: "helmet_x"}.get(cls, "?")
        print(f"  cls {cls} ({name}): {bbox_stats[cls]}")
    if errors:
        print(f"에러/스킵: {errors}건")
    print("done!")


if __name__ == "__main__":
    freeze_support()
    main()
