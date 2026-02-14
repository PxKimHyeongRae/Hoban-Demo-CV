#!/usr/bin/env python3
"""
snapshots_raw에 go2k_v2 + SAHI로 pseudo-labeling
go2k_manual 중복 제외, 카메라당 N장 균등 샘플링

실행: python detect_snapshots_sahi.py --limit 2000
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="0")
parser.add_argument("--conf", type=float, default=0.50)
parser.add_argument("--model", default="/home/lay/hoban/hoban_go2k_v2/weights/best.pt")
parser.add_argument("--limit", type=int, default=0, help="카메라당 최대 N장")
args = parser.parse_args()

SNAP_DIR = "/home/lay/video_indoor/static/snapshots_raw"
GO2K_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
OUT_DIR = "/home/lay/hoban/datasets/snapshots_labels"
os.makedirs(OUT_DIR, exist_ok=True)

# go2k 중복 제외용 (파일명 기준)
go2k_files = set(os.listdir(GO2K_DIR))

# 카메라별 수집 (중복 제외)
cam_files = {}
skipped = 0
for f in sorted(os.listdir(SNAP_DIR)):
    if not f.endswith(".jpg"):
        continue
    if f in go2k_files:
        skipped += 1
        continue
    cam = f.split("_")[0]
    cam_files.setdefault(cam, []).append(f)

print(f"go2k 중복 제외: {skipped}장")
for cam in sorted(cam_files):
    print(f"  {cam}: {len(cam_files[cam])}장")

# 카메라당 균등 샘플링
targets = []
for cam in sorted(cam_files):
    files = cam_files[cam]
    if args.limit > 0 and len(files) > args.limit:
        step = len(files) / args.limit
        files = [files[int(i * step)] for i in range(args.limit)]
    for f in files:
        targets.append(f)

print(f"\n대상: {len(targets)}장 (limit={args.limit if args.limit > 0 else '없음'})")
print(f"모델: {args.model}, conf={args.conf}\n")

# 모델 로드
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=args.conf,
    device=args.device,
)

# 추론
t0 = time.time()
total_boxes = 0
img_with_det = 0

for i, fname in enumerate(targets):
    if i % 500 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(targets) - i)
        print(f"  [{i}/{len(targets)}] elapsed={elapsed/60:.1f}m, ETA={eta/60:.1f}m, bbox={total_boxes}")

    img_path = os.path.join(SNAP_DIR, fname)

    result = get_sliced_prediction(
        img_path, model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS",
    )

    preds = result.object_prediction_list
    if not preds:
        continue

    img_with_det += 1
    img = Image.open(img_path)
    img_w, img_h = img.size

    label_name = fname.replace(".jpg", ".txt")
    label_path = os.path.join(OUT_DIR, label_name)
    with open(label_path, "w") as f:
        for p in preds:
            bbox = p.bbox
            cx = (bbox.minx + bbox.maxx) / 2 / img_w
            cy = (bbox.miny + bbox.maxy) / 2 / img_h
            w = (bbox.maxx - bbox.minx) / img_w
            h = (bbox.maxy - bbox.miny) / img_h
            f.write(f"{p.category.id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            total_boxes += 1

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"완료! ({elapsed/60:.1f}분)")
print(f"탐지 이미지: {img_with_det}/{len(targets)}장 ({img_with_det/len(targets)*100:.1f}%)")
print(f"총 bbox: {total_boxes}개")
print(f"라벨 저장: {OUT_DIR}")
