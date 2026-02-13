"""bbox 시각화 - helmet_o, helmet_x 각 10장씩 저장"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from multiprocessing import freeze_support

POOL_IMG = Path(r"D:\task\hoban\helmet_pool\images")
POOL_LBL = Path(r"D:\task\hoban\helmet_pool\labels")
OUT = Path(r"D:\task\hoban\bbox_check")

COLORS = {0: (0, 255, 0), 1: (255, 0, 0)}  # green=helmet_o, red=helmet_x
NAMES = {0: "helmet_o", 1: "helmet_x"}


def draw_bbox(img, label_path):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = COLORS.get(cls, (255, 255, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            area = bw * bh
            label = f"{NAMES.get(cls, str(cls))} {area*100:.2f}%"
            draw.text((x1, max(0, y1 - 15)), label, fill=color)
    return img


def find_images_for_cls(cls_id, count=10):
    """특정 클래스가 포함된 라벨 파일 찾기 (다양한 크기로)"""
    candidates = []
    for lf in POOL_LBL.iterdir():
        if lf.suffix != ".txt":
            continue
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == cls_id:
                    area = float(parts[3]) * float(parts[4])
                    candidates.append((lf.stem, area))
                    break

    # 크기별로 정렬해서 고르게 선택
    candidates.sort(key=lambda x: x[1])
    n = len(candidates)
    if n <= count:
        return [c[0] for c in candidates]

    # 균등 간격으로 선택
    indices = [int(i * (n - 1) / (count - 1)) for i in range(count)]
    return [candidates[i][0] for i in indices]


def main():
    OUT.mkdir(exist_ok=True)

    for cls_id in [0, 1]:
        cls_name = NAMES[cls_id]
        stems = find_images_for_cls(cls_id, 10)
        print(f"\n=== {cls_name} (cls {cls_id}) - {len(stems)}장 ===")

        for i, stem in enumerate(stems):
            # 이미지 찾기
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = POOL_IMG / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
            if not img_path:
                continue

            lbl_path = POOL_LBL / f"{stem}.txt"
            img = Image.open(img_path)
            img = draw_bbox(img, lbl_path)

            out_path = OUT / f"{cls_name}_{i+1:02d}_{stem[:40]}.jpg"
            img.save(out_path, quality=90)

            # bbox 정보 출력
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) == cls_id:
                        area = float(parts[3]) * float(parts[4])
                        px = (area ** 0.5) * 640
                        print(f"  {i+1:2d}. area={area*100:.3f}% (~{px:.0f}px) | {img.size[0]}x{img.size[1]} | {stem[:50]}")

    print(f"\n저장 위치: {OUT}")
    print("done!")


if __name__ == "__main__":
    freeze_support()
    main()
