"""
helmet_21k JSON → YOLO 변환 + 이상치 분석
- helmet_wearing: class "07" → cls 0
- helmet_not_wearing: class "08" → cls 1
- background: 빈 라벨 (negative sample)
- 상위 0.1% 이상치 보고
- 결과: helmet_v9/ (images + labels)
"""
import json
import shutil
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

BASE = Path(r"D:\task\hoban\dataset\helmet_21k")
OUT = Path(r"D:\task\hoban\helmet_v9")
NEG_OUT = Path(r"D:\task\hoban\negative_samples")

CLS_MAP = {"07": 0, "08": 1}


def convert_folder(folder_name, out_img, out_lbl):
    img_dir = BASE / folder_name / "images"
    lbl_dir = BASE / folder_name / "labels"

    converted = 0
    skipped = 0
    all_bboxes = []  # (area, w, h, stem, cls)

    for jf in sorted(lbl_dir.iterdir()):
        if jf.suffix != ".json":
            continue

        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except:
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
            # YOLO format
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            # 범위 클리핑
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))

            if bw <= 0 or bh <= 0:
                continue

            area = bw * bh
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            all_bboxes.append((area, bw, bh, jf.stem, cls))

        if not lines:
            skipped += 1
            continue

        # 이미지 찾기
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

    return converted, skipped, all_bboxes


def main():
    # helmet_v9 출력
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)

    # negative_samples 출력
    if NEG_OUT.exists():
        shutil.rmtree(NEG_OUT)
    (NEG_OUT / "images").mkdir(parents=True)

    print("=== helmet_21k → YOLO 변환 ===\n")

    all_bboxes = []

    # helmet_wearing (cls 0)
    c, s, bboxes = convert_folder("helmet_wearing", OUT / "images", OUT / "labels")
    cls0_bboxes = [b for b in bboxes if b[4] == 0]
    all_bboxes.extend(bboxes)
    print(f"  helmet_wearing: {c}장 변환, {s}장 스킵, {len(cls0_bboxes)} bbox (cls 0)")

    # helmet_not_wearing (cls 1)
    c, s, bboxes = convert_folder("helmet_not_wearing", OUT / "images", OUT / "labels")
    cls1_bboxes = [b for b in bboxes if b[4] == 1]
    all_bboxes.extend(bboxes)
    print(f"  helmet_not_wearing: {c}장 변환, {s}장 스킵, {len(cls1_bboxes)} bbox (cls 1)")

    # background (negative samples)
    bg_dir = BASE / "background" / "images"
    neg_count = 0
    if bg_dir.exists():
        for img in bg_dir.iterdir():
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy2(img, NEG_OUT / "images" / img.name)
                neg_count += 1
    print(f"  background: {neg_count}장 → negative_samples/")

    # 이상치 분석
    print(f"\n=== 이상치 분석 ===")

    for cls, name, bboxes_cls in [(0, "helmet_o", cls0_bboxes), (1, "helmet_x", cls1_bboxes)]:
        if not bboxes_cls:
            continue
        bboxes_cls.sort(key=lambda x: -x[0])
        total = len(bboxes_cls)
        top_n = max(1, int(total * 0.001))

        print(f"\n  cls {cls} ({name}) - {total}개 bbox")
        print(f"  상위 0.1% ({top_n}개):")
        for i, (area, w, h, stem, _) in enumerate(bboxes_cls[:min(top_n, 20)]):
            px = (area ** 0.5) * 640
            print(f"    {i+1}. area={area*100:.2f}% ({px:.0f}px) w={w:.4f} h={h:.4f} | {stem}")

        # 분포
        areas = sorted([b[0] for b in bboxes_cls])
        print(f"  area 분포:")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            idx = int(total * p / 100)
            a = areas[idx]
            px = (a ** 0.5) * 640
            print(f"    {p:>2}%ile: {a*100:.3f}% (~{px:.0f}px)")

    print(f"\n=== 총 결과 ===")
    print(f"  helmet_v9: {len(list((OUT/'images').iterdir()))}장")
    total_cls = Counter()
    for b in all_bboxes:
        total_cls[b[4]] += 1
    for cls in sorted(total_cls):
        print(f"    cls {cls}: {total_cls[cls]} bbox")
    print(f"  negative_samples: {neg_count}장")

    print("\ndone!")


if __name__ == "__main__":
    freeze_support()
    main()
