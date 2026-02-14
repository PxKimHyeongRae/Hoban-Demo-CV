#!/usr/bin/env python3
"""
go2k_manual 데이터에 대해 SAHI 추론

최적 조건 (sweep 결과):
  - 모델: go2k_v2 best.pt
  - conf: 0.50
  - SAHI: NMS / match_threshold=0.4 / IOS
  - F1=0.804, P=0.725, R=0.903

실행: python detect_go2k_sahi.py
GPU: python detect_go2k_sahi.py --device 0
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import numpy as np
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="0", help="cpu 또는 0")
parser.add_argument("--conf", type=float, default=0.50)
parser.add_argument("--model", default="/home/lay/hoban/hoban_go2k_v2/weights/best.pt")
args = parser.parse_args()

model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=args.conf,
    device=args.device,
)

img_dir = "/home/lay/hoban/datasets/go2k_manual/images"
images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
print(f"총 {len(images)}장 SAHI detect (go2k_v2, conf={args.conf}, device={args.device})")

cls_counter = Counter()
img_with_det = 0
total_boxes = 0
all_confs = []

for i, fname in enumerate(images):
    if i % 50 == 0:
        print(f"  {i}/{len(images)}...")

    result = get_sliced_prediction(
        os.path.join(img_dir, fname),
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS",
    )

    preds = result.object_prediction_list
    if preds:
        img_with_det += 1
        total_boxes += len(preds)
        for p in preds:
            cls_counter[p.category.id] += 1
            all_confs.append(p.score.value)

names = {0: "person_with_helmet", 1: "person_without_helmet"}
print(f"\n=== go2k_v2 + SAHI 결과 (go2k_manual) ===")
print(f"탐지 이미지: {img_with_det}/{len(images)}장 ({img_with_det/len(images)*100:.1f}%)")
print(f"총 bbox: {total_boxes}개")
for cls_id, cnt in sorted(cls_counter.items()):
    print(f"  {names.get(cls_id, cls_id)}: {cnt}개")

if all_confs:
    arr = np.array(all_confs)
    print(f"\nConfidence: 평균={arr.mean():.3f}, 중앙값={np.median(arr):.3f}, min={arr.min():.3f}, max={arr.max():.3f}")
