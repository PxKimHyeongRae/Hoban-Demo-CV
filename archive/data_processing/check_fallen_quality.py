"""fallen_pool 라벨 품질 검증"""
from pathlib import Path
from collections import Counter
import random

POOL = Path(r"D:\task\hoban\fallen_pool")

def main():
    labels = list((POOL / "labels").iterdir())
    labels = [l for l in labels if l.suffix == ".txt"]

    print(f"=== fallen_pool 라벨 품질 검증 ===")
    print(f"총 라벨 파일: {len(labels)}개\n")

    # 1. 빈 라벨 파일 체크
    empty = []
    for lf in labels:
        with open(lf) as f:
            content = f.read().strip()
        if not content:
            empty.append(lf.name)
    print(f"1. 빈 라벨 파일: {len(empty)}개")
    if empty[:5]:
        print(f"   예시: {empty[:5]}")

    # 2. bbox 크기 분석
    widths = []
    heights = []
    areas = []
    tiny = []  # < 0.01 area
    huge = []  # > 0.5 area
    out_of_bounds = []
    multi_cls = []
    bbox_per_img = []
    source_counter = Counter()

    for lf in labels:
        with open(lf) as f:
            lines = [l.strip() for l in f if l.strip()]

        bbox_per_img.append(len(lines))
        src = lf.stem.split("_")[0]
        source_counter[src] += 1

        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            if cls != 0:
                multi_cls.append((lf.name, cls))
                continue

            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            widths.append(w)
            heights.append(h)
            area = w * h
            areas.append(area)

            # 범위 체크
            if x - w/2 < -0.01 or x + w/2 > 1.01 or y - h/2 < -0.01 or y + h/2 > 1.01:
                out_of_bounds.append(lf.name)

            if area < 0.005:  # 매우 작은 bbox (이미지의 0.5% 미만)
                tiny.append((lf.name, w, h, area))
            elif area > 0.7:  # 매우 큰 bbox (이미지의 70% 초과)
                huge.append((lf.name, w, h, area))

    print(f"\n2. 클래스 0 이외의 라벨: {len(multi_cls)}개")
    if multi_cls[:5]:
        print(f"   예시: {multi_cls[:5]}")

    print(f"\n3. bbox 크기 분포:")
    if widths:
        widths.sort()
        heights.sort()
        areas.sort()
        print(f"   width  - min: {widths[0]:.4f}, median: {widths[len(widths)//2]:.4f}, max: {widths[-1]:.4f}")
        print(f"   height - min: {heights[0]:.4f}, median: {heights[len(heights)//2]:.4f}, max: {heights[-1]:.4f}")
        print(f"   area   - min: {areas[0]:.6f}, median: {areas[len(areas)//2]:.4f}, max: {areas[-1]:.4f}")

        # 분위수
        p5 = areas[int(len(areas)*0.05)]
        p95 = areas[int(len(areas)*0.95)]
        print(f"   area 5%ile: {p5:.4f}, 95%ile: {p95:.4f}")

    print(f"\n4. 이상 bbox:")
    print(f"   매우 작은 bbox (area < 0.5%): {len(tiny)}개 ({len(tiny)/max(len(areas),1)*100:.1f}%)")
    if tiny[:3]:
        for name, w, h, a in tiny[:3]:
            print(f"     {name}: w={w:.4f}, h={h:.4f}, area={a:.6f}")
    print(f"   매우 큰 bbox (area > 70%): {len(huge)}개 ({len(huge)/max(len(areas),1)*100:.1f}%)")
    if huge[:3]:
        for name, w, h, a in huge[:3]:
            print(f"     {name}: w={w:.4f}, h={h:.4f}, area={a:.4f}")
    print(f"   범위 초과 (좌표 > 1.0): {len(out_of_bounds)}개")

    print(f"\n5. 이미지당 bbox 수:")
    bbox_per_img.sort()
    c = Counter(bbox_per_img)
    print(f"   min: {bbox_per_img[0]}, median: {bbox_per_img[len(bbox_per_img)//2]}, max: {bbox_per_img[-1]}")
    for k in sorted(c.keys()):
        if c[k] >= 10 or k <= 5:
            print(f"   {k}개: {c[k]}장 ({c[k]/len(bbox_per_img)*100:.1f}%)")

    print(f"\n6. 소스별 분포:")
    for src, cnt in sorted(source_counter.items(), key=lambda x: -x[1]):
        print(f"   {src}: {cnt}장 ({cnt/len(labels)*100:.1f}%)")

    # 7. 이미지 존재 확인 (랜덤 100개)
    random.seed(42)
    sample = random.sample(labels, min(100, len(labels)))
    missing = 0
    for lf in sample:
        found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            if (POOL / "images" / f"{lf.stem}{ext}").exists():
                found = True
                break
        if not found:
            missing += 1
    print(f"\n7. 이미지-라벨 매칭 (100개 샘플): 누락 {missing}개")

    # 8. 이미지 해상도 샘플 체크
    try:
        from PIL import Image
        sample_imgs = random.sample(list((POOL / "images").iterdir()), min(50, len(labels)))
        sizes = []
        for img_path in sample_imgs:
            try:
                with Image.open(img_path) as im:
                    sizes.append(im.size)
            except:
                pass
        if sizes:
            ws = [s[0] for s in sizes]
            hs = [s[1] for s in sizes]
            print(f"\n8. 이미지 해상도 (50개 샘플):")
            print(f"   width  - min: {min(ws)}, max: {max(ws)}")
            print(f"   height - min: {min(hs)}, max: {max(hs)}")
            small = sum(1 for w,h in sizes if w < 200 or h < 200)
            print(f"   200px 미만: {small}개")
    except ImportError:
        print("\n8. PIL 미설치, 해상도 체크 건너뜀")

    print("\ndone!")

if __name__ == "__main__":
    main()
