#!/usr/bin/env python3
"""
수작업 라벨링 데이터(CVAT XML) → YOLO 변환 + 학습 스크립트

데이터: Z:\home\lay\hoban\datasets\manual_first_250 (CVAT XML, mask annotations)
변환: mask bbox → YOLO format labels
학습: yolo26m.pt에서 30 epochs

실행 (Windows 로컬):
  python train_manual_v1.py
  python train_manual_v1.py --epochs 50
  python train_manual_v1.py --convert-only   # 변환만
"""

import os
import sys
import random
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# ===== Config =====
SEED = 42
random.seed(SEED)

# 경로 (Windows)
BASE_DIR = r"Z:\home\lay\hoban"
SRC_DIR = os.path.join(BASE_DIR, "datasets", "manual_first_250")
OUT_DIR = os.path.join(BASE_DIR, "datasets_manual_v1")
MODEL = os.path.join(BASE_DIR, "yolo26m.pt")

# 클래스 매핑 (CVAT label → YOLO class id)
CLASS_MAP = {
    "person_with_helemt": 0,    # typo in CVAT
    "person_with_helmet": 0,
    "person_without_helmet": 1,
}

VALID_RATIO = 0.2  # 20% validation


def convert_cvat_to_yolo():
    """CVAT XML → YOLO labels 변환"""
    xml_path = os.path.join(SRC_DIR, "annotations.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_src = os.path.join(SRC_DIR, "images")
    img_out_train = os.path.join(OUT_DIR, "train", "images")
    lbl_out_train = os.path.join(OUT_DIR, "train", "labels")
    img_out_valid = os.path.join(OUT_DIR, "valid", "images")
    lbl_out_valid = os.path.join(OUT_DIR, "valid", "labels")

    for d in [img_out_train, lbl_out_train, img_out_valid, lbl_out_valid]:
        os.makedirs(d, exist_ok=True)

    images = root.findall(".//image")
    print(f"XML images: {len(images)}")

    # 라벨 데이터 수집
    data = []  # (img_name, img_w, img_h, [(cls, cx, cy, w, h), ...])

    for img_elem in images:
        img_name = img_elem.get("name")
        img_w = int(img_elem.get("width"))
        img_h = int(img_elem.get("height"))

        bboxes = []

        # mask → bbox 변환 (left, top, width, height)
        for mask in img_elem.findall("mask"):
            label = mask.get("label")
            cls_id = CLASS_MAP.get(label)
            if cls_id is None:
                print(f"  Warning: unknown label '{label}', skipping")
                continue

            left = int(mask.get("left"))
            top = int(mask.get("top"))
            w = int(mask.get("width"))
            h = int(mask.get("height"))

            # YOLO normalized (center x, center y, w, h)
            cx = (left + w / 2) / img_w
            cy = (top + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # clamp
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = min(nw, 1.0)
            nh = min(nh, 1.0)

            bboxes.append((cls_id, cx, cy, nw, nh))

        # box annotations (있으면)
        for box in img_elem.findall("box"):
            label = box.get("label")
            cls_id = CLASS_MAP.get(label)
            if cls_id is None:
                continue

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            cx = (xtl + xbr) / 2 / img_w
            cy = (ytl + ybr) / 2 / img_h
            nw = (xbr - xtl) / img_w
            nh = (ybr - ytl) / img_h

            bboxes.append((cls_id, cx, cy, nw, nh))

        data.append((img_name, img_w, img_h, bboxes))

    # 실제 이미지 파일 존재 확인
    existing = set(os.listdir(img_src))
    data = [(n, w, h, b) for n, w, h, b in data if n in existing]
    print(f"Images with files: {len(data)}")

    # 라벨링된 이미지만 사용 (negative 제외)
    positives = [(n, w, h, b) for n, w, h, b in data if b]
    skipped = len(data) - len(positives)
    print(f"  Annotated (사용): {len(positives)}")
    print(f"  Unannotated (제외): {skipped}")

    # train/valid split
    random.shuffle(positives)

    n_val = max(1, int(len(positives) * VALID_RATIO))
    val_data = positives[:n_val]
    train_data = positives[n_val:]
    random.shuffle(val_data)
    random.shuffle(train_data)

    # 저장
    stats = {"train": {0: 0, 1: 0, "neg": 0}, "valid": {0: 0, 1: 0, "neg": 0}}

    for split, split_data in [("train", train_data), ("valid", val_data)]:
        img_out = os.path.join(OUT_DIR, split, "images")
        lbl_out = os.path.join(OUT_DIR, split, "labels")

        for img_name, img_w, img_h, bboxes in split_data:
            # 이미지 복사
            src = os.path.join(img_src, img_name)
            dst = os.path.join(img_out, img_name)
            shutil.copy2(src, dst)

            # 라벨 저장
            lbl_name = Path(img_name).stem + ".txt"
            with open(os.path.join(lbl_out, lbl_name), "w") as f:
                for cls, cx, cy, w, h in bboxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    stats[split][cls] = stats[split].get(cls, 0) + 1

            if not bboxes:
                stats[split]["neg"] += 1

    # data.yaml
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

    print(f"\n=== Dataset Created ===")
    for split in ["train", "valid"]:
        s = stats[split]
        total = s.get(0, 0) + s.get(1, 0)
        print(f"  [{split}] helmet_o={s.get(0,0)}, helmet_x={s.get(1,0)}, negative={s.get('neg',0)} (bbox total: {total})")
    print(f"  data.yaml: {os.path.join(OUT_DIR, 'data.yaml')}")

    return os.path.join(OUT_DIR, "data.yaml")


def train(data_yaml, epochs=30, batch=16):
    """YOLO 학습"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"Manual Data Training")
    print(f"  Model: {MODEL}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}, Batch: {batch}")
    print(f"  imgsz: 640")
    print(f"{'='*60}")

    model = YOLO(MODEL)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device="0",
        project=BASE_DIR,
        name="hoban_manual_v1",
        exist_ok=True,

        # 학습 설정
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        cos_lr=True,

        # 소량 데이터 → 강한 augmentation
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.15,
        degrees=10.0,
        fliplr=0.5,
        erasing=0.3,

        # 기타
        patience=20,
        amp=True,
        workers=4,
        seed=SEED,
        deterministic=True,
        plots=True,
        verbose=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {BASE_DIR}\\hoban_manual_v1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--convert-only", action="store_true", help="변환만 하고 학습 안 함")
    args = parser.parse_args()

    # 기존 출력 삭제
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # 1. 변환
    data_yaml = convert_cvat_to_yolo()

    if args.convert_only:
        print("\nConversion complete. Skipping training.")
        return

    # 2. 학습
    train(data_yaml, epochs=args.epochs, batch=args.batch)


if __name__ == "__main__":
    main()
