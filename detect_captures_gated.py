#!/usr/bin/env python3
"""
captures cam1+cam2 → SAHI + 풀이미지 게이트 탐지 → 라벨 저장
Gate Strategy #2: conf=0.20, radius=40px

전체시간, 10초 간격, go2k 타임스탬프 제외
3000장 탐지 도달 시 조기 종료

실행: python detect_captures_gated.py
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from PIL import Image
import os
import time
from collections import defaultdict

CAP_DIR = "/home/lay/video_indoor/static/captures"
GO2K_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
OUT_DIR = "/home/lay/hoban/datasets/captures_labels_3k"
MODEL_PATH = "/home/lay/hoban/hoban_go2k_v2/weights/best.pt"
TARGET_COUNT = 3000
MIN_INTERVAL = 10
SAHI_CONF = 0.50

# Gate Strategy #2
GATE_CONF = 0.20
GATE_RADIUS = 40

os.makedirs(OUT_DIR, exist_ok=True)


def parse_ts(fname):
    parts = fname.split("_")
    if len(parts) < 3:
        return None, None
    try:
        date = int(parts[1])
        t = parts[2]
        h, m, s = int(t[:2]), int(t[2:4]), int(t[4:6])
        return date * 86400 + h * 3600 + m * 60 + s, h
    except (ValueError, IndexError):
        return None, None


def point_near_any(px, py, gates, radius):
    for gx1, gy1, gx2, gy2 in gates:
        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        if abs(px - gcx) <= radius and abs(py - gcy) <= radius:
            return True
    return False


# go2k 타임스탬프 제외
go2k_keys = set()
for f in os.listdir(GO2K_DIR):
    parts = f.split("_")
    if len(parts) >= 3:
        go2k_keys.add("_".join(parts[:3]))

# cam1+cam2 이미지 수집 (카메라별 10초 간격) → 타임스탬프로 섞기
targets = []
for cam in ["cam1", "cam2"]:
    cam_dir = os.path.join(CAP_DIR, cam)
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))

    last_ts = None
    count = 0
    for f in files:
        ts, h = parse_ts(f)
        if ts is None:
            continue
        key = "_".join(f.split("_")[:3])
        if key in go2k_keys:
            continue
        if last_ts and (ts - last_ts) < MIN_INTERVAL:
            continue
        targets.append((cam, f, ts))
        last_ts = ts
        count += 1

    print(f"{cam}: {count}장 (전체시간, 10초 간격, go2k 제외)")

# 타임스탬프 기준 정렬 (cam1+cam2 섞기)
targets.sort(key=lambda x: x[2])
targets = [(cam, fname) for cam, fname, ts in targets]

print(f"\n대상: {len(targets)}장, 목표: {TARGET_COUNT}장 탐지")
print(f"모델: {MODEL_PATH}")
print(f"SAHI: conf={SAHI_CONF}, 640x640, overlap=0.2, NMS/0.4/IOS")
print(f"Gate: conf={GATE_CONF}, radius={GATE_RADIUS}px\n")

# 모델 로드
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=MODEL_PATH,
    confidence_threshold=SAHI_CONF,
    device="0",
)
full_model = YOLO(MODEL_PATH)
full_model.to("cuda:0")
print("모델 로드 완료\n")

# 추론
t0 = time.time()
total_boxes = 0
total_boxes_before_gate = 0
img_with_det = 0
processed_files = []

for i, (cam, fname) in enumerate(targets):
    if img_with_det >= TARGET_COUNT:
        print(f"\n  목표 {TARGET_COUNT}장 도달! 조기 종료.")
        break

    if i % 500 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(targets) - i)
        print(f"  [{i}/{len(targets)}] elapsed={elapsed/60:.1f}m, ETA={eta/60:.1f}m, "
              f"탐지={img_with_det}, bbox={total_boxes}")

    img_path = os.path.join(CAP_DIR, cam, fname)

    # 1) 풀이미지 추론 (게이트용)
    full_results = full_model.predict(img_path, imgsz=640, conf=GATE_CONF, device="0", verbose=False)
    gates = []
    for r in full_results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            gates.append((float(x1), float(y1), float(x2), float(y2)))

    # 풀이미지에서 사람 후보 없으면 스킵 (게이트 효과)
    if not gates:
        continue

    # 2) SAHI 추론
    result = get_sliced_prediction(
        img_path, sahi_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="NMS",
        postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS",
    )

    preds = result.object_prediction_list
    if not preds:
        continue

    total_boxes_before_gate += len(preds)

    # 3) 게이트 필터링
    img = Image.open(img_path)
    img_w, img_h = img.size

    filtered = []
    for p in preds:
        bbox = p.bbox
        cx = (bbox.minx + bbox.maxx) / 2
        cy = (bbox.miny + bbox.maxy) / 2
        if point_near_any(cx, cy, gates, GATE_RADIUS):
            filtered.append(p)

    if not filtered:
        continue

    img_with_det += 1

    label_name = fname.replace(".jpg", ".txt")
    with open(os.path.join(OUT_DIR, label_name), "w") as f:
        for p in filtered:
            bbox = p.bbox
            cx = (bbox.minx + bbox.maxx) / 2 / img_w
            cy = (bbox.miny + bbox.maxy) / 2 / img_h
            w = (bbox.maxx - bbox.minx) / img_w
            h = (bbox.maxy - bbox.miny) / img_h
            f.write(f"{p.category.id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            total_boxes += 1

    processed_files.append((cam, fname))

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"완료! ({elapsed/60:.1f}분)")
print(f"처리: {i+1}/{len(targets)}장")
print(f"탐지 이미지: {img_with_det}장 ({img_with_det/(i+1)*100:.1f}%)")
print(f"게이트 전 bbox: {total_boxes_before_gate}개")
print(f"게이트 후 bbox: {total_boxes}개 ({total_boxes_before_gate - total_boxes}개 필터)")
print(f"라벨 저장: {OUT_DIR}")

# 카메라별 통계
cam_counts = defaultdict(int)
for cam, fname in processed_files:
    cam_counts[cam] += 1
for cam in sorted(cam_counts):
    print(f"  {cam}: {cam_counts[cam]}장")
