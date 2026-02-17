#!/usr/bin/env python3
"""
snapshots에서 헬멧_미착용 이미지 30분 간격으로 추출 → CVAT 패키징

실행: python prepare_cvat_helmet_off.py [--server]
"""
import os
import zipfile
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--min-gap", type=int, default=180, help="최소 간격 (초, 기본 3분)")
    args = parser.parse_args()

    BASE = "/" if args.server else "Z:/"
    SNAP_DIR = os.path.join(BASE, "home/lay/video_indoor/static/snapshots")
    OUT_DIR = os.path.join(BASE, "home/lay/hoban/datasets/cvat_helmet_off")
    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]

    os.makedirs(OUT_DIR, exist_ok=True)

    # 헬멧_미착용만 수집
    off_files = sorted(f for f in os.listdir(SNAP_DIR)
                       if "헬멧_미착용" in f and f.startswith(("cam1", "cam2")))
    print(f"헬멧_미착용 전체: {len(off_files)}장")

    # 카메라+날짜별로 그룹핑
    by_cam_date = defaultdict(list)
    for f in off_files:
        parts = f.split("_")
        cam, date, time_str = parts[0], parts[1], parts[2]
        h, m, s = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
        sec = h * 3600 + m * 60 + s
        by_cam_date[f"{cam}_{date}"].append((sec, f))

    # 30분 간격 필터
    selected = []
    for key in sorted(by_cam_date.keys()):
        entries = sorted(by_cam_date[key])
        last_sec = -99999
        for sec, fname in entries:
            if sec - last_sec >= args.min_gap:
                selected.append(fname)
                last_sec = sec

    print(f"30분 간격 필터 후: {len(selected)}장")

    # 날짜 분포
    date_dist = defaultdict(int)
    cam_dist = defaultdict(int)
    for f in selected:
        parts = f.split("_")
        date_dist[parts[1]] += 1
        cam_dist[parts[0]] += 1

    print(f"\n카메라별: {dict(cam_dist)}")
    print(f"날짜별:")
    for d, c in sorted(date_dist.items()):
        print(f"  {d}: {c}장")

    # CVAT 패키징
    print(f"\n패키징 중...", flush=True)

    ann_path = os.path.join(OUT_DIR, "annotations.zip")
    with zipfile.ZipFile(ann_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.data",
                     f"classes = {len(CLASS_NAMES)}\n"
                     f"train = data/train.txt\n"
                     f"names = data/obj.names\n"
                     f"backup = backup/\n")
        zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")

        train_lines = []
        for fname in selected:
            train_lines.append(f"data/obj_train_data/{fname}")
            lbl_name = fname.replace(".jpg", ".txt")
            zf.writestr(f"obj_train_data/{lbl_name}", "")
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    img_zip_path = os.path.join(OUT_DIR, "images.zip")
    with zipfile.ZipFile(img_zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, fname in enumerate(selected):
            if i % 20 == 0:
                print(f"  {i}/{len(selected)}...", flush=True)
            zf.write(os.path.join(SNAP_DIR, fname), fname)

    ann_size = os.path.getsize(ann_path) / 1024 / 1024
    img_size = os.path.getsize(img_zip_path) / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"완료! {len(selected)}장")
    print(f"  annotations.zip: {ann_size:.1f}MB")
    print(f"  images.zip: {img_size:.1f}MB")
    print(f"  출력: {OUT_DIR}")
    print(f"\nCVAT:")
    print(f"  1. Create Task → images.zip")
    print(f"  2. Upload annotations → YOLO 1.1 → annotations.zip")
    print(f"  3. 라벨링 (빈 라벨은 삭제)")
    print(f"  4. Export (YOLO 1.1) → 클로드에게 전달")


if __name__ == "__main__":
    main()
