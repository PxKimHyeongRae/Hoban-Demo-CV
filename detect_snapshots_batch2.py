#!/usr/bin/env python3
"""
snapshots_raw 2차 배치: cam1+cam2 미사용분에서 20초 간격 선별 → SAHI 탐지
기존 라벨링 이미지와 20초 이상 간격 보장

실행: python detect_snapshots_batch2.py
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import os
import time
from collections import defaultdict

SNAP_DIR = "/home/lay/video_indoor/static/snapshots_raw"
LABEL_DIR = "/home/lay/hoban/datasets/snapshots_labels"
GO2K_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
OUT_DIR = "/home/lay/hoban/datasets/snapshots_labels_batch2"
MODEL_PATH = "/home/lay/hoban/hoban_go2k_v2/weights/best.pt"
MIN_INTERVAL = 20  # 초
CONF = 0.50

os.makedirs(OUT_DIR, exist_ok=True)


def parse_ts(fname):
    parts = fname.split("_")
    if len(parts) < 4:
        return None
    try:
        date = int(parts[1])
        t = parts[2]
        h, m, s = int(t[:2]), int(t[2:4]), int(t[4:6])
        ms = int(parts[3])
        return date * 86400 + h * 3600 + m * 60 + s + ms / 1000
    except (ValueError, IndexError):
        return None


# 이미 사용된 파일
used = set(f.replace(".txt", ".jpg") for f in os.listdir(LABEL_DIR))
used.update(os.listdir(GO2K_DIR))

# 기존 라벨 타임스탬프
used_ts = defaultdict(list)
for f in sorted(used):
    cam = f.split("_")[0]
    ts = parse_ts(f)
    if ts:
        used_ts[cam].append(ts)
for cam in used_ts:
    used_ts[cam].sort()

# cam1+cam2 미사용 후보
candidates = [f for f in sorted(os.listdir(SNAP_DIR))
              if f.endswith(".jpg") and f not in used and f.split("_")[0] in ("cam1", "cam2")]

# 20초 간격 필터 (기존 라벨과도 20초 이상)
cam_cands = defaultdict(list)
for f in candidates:
    cam_cands[f.split("_")[0]].append(f)

targets = []
for cam in ["cam1", "cam2"]:
    existing_ts = used_ts.get(cam, [])
    last_ts = None
    count = 0
    for f in cam_cands[cam]:
        ts = parse_ts(f)
        if ts is None:
            continue
        if any(abs(ts - ets) < MIN_INTERVAL for ets in existing_ts):
            continue
        if last_ts and (ts - last_ts) < MIN_INTERVAL:
            continue
        targets.append(f)
        last_ts = ts
        count += 1
    print(f"{cam}: {count}장 선별")

print(f"\n대상: {len(targets)}장")
print(f"모델: {MODEL_PATH}, conf={CONF}")
print(f"SAHI: 640x640, overlap=0.2, NMS/0.4/IOS\n")

# 모델 로드
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=MODEL_PATH,
    confidence_threshold=CONF,
    device="0",
)

# 추론
t0 = time.time()
total_boxes = 0
img_with_det = 0

for i, fname in enumerate(targets):
    if i % 200 == 0:
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
        # 빈 라벨도 저장 (negative sample)
        label_name = fname.replace(".jpg", ".txt")
        with open(os.path.join(OUT_DIR, label_name), "w") as f:
            pass
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
