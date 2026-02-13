#!/usr/bin/env python3
"""
go500 CVAT XML mask → YOLO bbox 변환 + 학습 데이터셋 빌드

- CVAT mask(RLE) 어노테이션에서 bounding box 추출
- YOLO format으로 변환
- train/valid split (80/20)

실행: python build_go500.py
출력: /home/lay/hoban/datasets/go500_yolo/
"""

import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

SEED = 42
random.seed(SEED)

SRC_DIR = "/home/lay/hoban/datasets/go500"
OUT_DIR = "/home/lay/hoban/datasets/go500_yolo"
VALID_RATIO = 0.2

# 클래스 매핑 (typo 포함)
CLASS_MAP = {
    "person_with_helemt": 0,
    "person_with_helmet": 0,
    "person_without_helmet": 1,
}


def parse_cvat_xml(xml_path):
    """CVAT XML에서 이미지별 bbox 추출 (mask RLE → bbox)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {}  # {filename: [(cls, cx, cy, w, h), ...]}

    for img_elem in root.findall("image"):
        fname = img_elem.get("name")
        img_w = int(img_elem.get("width"))
        img_h = int(img_elem.get("height"))
        bboxes = []

        for mask in img_elem.findall("mask"):
            label = mask.get("label")
            cls = CLASS_MAP.get(label)
            if cls is None:
                continue

            # mask의 left, top, width, height로 bbox 계산
            left = int(mask.get("left"))
            top = int(mask.get("top"))
            mw = int(mask.get("width"))
            mh = int(mask.get("height"))

            # YOLO normalized format
            cx = (left + mw / 2) / img_w
            cy = (top + mh / 2) / img_h
            w = mw / img_w
            h = mh / img_h

            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = min(w, 1.0)
            h = min(h, 1.0)

            if w > 0.001 and h > 0.001:
                bboxes.append((cls, cx, cy, w, h))

        # box 어노테이션도 처리
        for box in img_elem.findall("box"):
            label = box.get("label")
            cls = CLASS_MAP.get(label)
            if cls is None:
                continue

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            cx = (xtl + xbr) / 2 / img_w
            cy = (ytl + ybr) / 2 / img_h
            w = (xbr - xtl) / img_w
            h = (ybr - ytl) / img_h

            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = min(w, 1.0)
            h = min(h, 1.0)

            if w > 0.001 and h > 0.001:
                bboxes.append((cls, cx, cy, w, h))

        data[fname] = bboxes

    return data


def main():
    xml_path = os.path.join(SRC_DIR, "annotations.xml")
    img_dir = os.path.join(SRC_DIR, "images")

    print("=" * 60)
    print("go500 CVAT → YOLO Dataset Builder")
    print(f"  Source: {SRC_DIR}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    # 파싱
    data = parse_cvat_xml(xml_path)
    print(f"\nParsed: {len(data)} images")

    # 라벨링된 이미지만 필터
    labeled = {k: v for k, v in data.items() if v}
    unlabeled = {k: v for k, v in data.items() if not v}
    print(f"  Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}")

    # 클래스 통계
    cls_count = defaultdict(int)
    area_stats = []
    for fname, bboxes in labeled.items():
        for cls, cx, cy, w, h in bboxes:
            cls_count[cls] += 1
            area_stats.append(w * h * 100)

    print(f"  Class 0 (helmet_o): {cls_count[0]}")
    print(f"  Class 1 (helmet_x): {cls_count[1]}")

    if area_stats:
        area_stats.sort()
        n = len(area_stats)
        print(f"  Bbox Area P25={area_stats[n//4]:.4f}%, P50={area_stats[n//2]:.4f}%, P75={area_stats[3*n//4]:.4f}%")

    # 출력 디렉토리
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for d in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

    # Train/Valid split
    labeled_files = sorted(labeled.keys())
    random.shuffle(labeled_files)
    n_val = max(1, int(len(labeled_files) * VALID_RATIO))
    val_files = set(labeled_files[:n_val])
    train_files = set(labeled_files[n_val:])

    print(f"\n  Train: {len(train_files)}, Valid: {len(val_files)}")

    # 저장
    for fname in labeled_files:
        split = "valid" if fname in val_files else "train"
        bboxes = labeled[fname]
        stem = os.path.splitext(fname)[0]

        # 이미지 복사
        src_img = os.path.join(img_dir, fname)
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(OUT_DIR, split, "images", fname))
        else:
            continue

        # 라벨 저장
        lbl_path = os.path.join(OUT_DIR, split, "labels", stem + ".txt")
        with open(lbl_path, "w") as f:
            for cls, cx, cy, w, h in bboxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # data.yaml (서버)
    data_yaml = (
        f"path: {OUT_DIR}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: person_with_helmet\n"
        f"  1: person_without_helmet\n"
    )
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(data_yaml)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"  Train: {len(train_files)} images")
    print(f"  Valid: {len(val_files)} images")
    print(f"  data.yaml: {os.path.join(OUT_DIR, 'data.yaml')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
