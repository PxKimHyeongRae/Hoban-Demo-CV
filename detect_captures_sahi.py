#!/usr/bin/env python3
"""
captures cam1~3 이미지에 go2k_v2 + SAHI로 pseudo-labeling
go2k_manual과 중복되는 타임스탬프 이미지 제외

출력: /home/lay/hoban/datasets/captures_labels/ (YOLO format)

실행:
  python detect_captures_sahi.py --count                          # 통계 확인
  python detect_captures_sahi.py --hours 8 18 --limit 2000        # 주간 2000장씩
  python detect_captures_sahi.py --hours 8 18 --interval 300      # 주간 5분 간격
  python detect_captures_sahi.py                                  # 전체
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="0")
parser.add_argument("--conf", type=float, default=0.50)
parser.add_argument("--model", default="/home/lay/hoban/hoban_go2k_v2/weights/best.pt")
parser.add_argument("--cam", nargs="+", default=["cam1", "cam2", "cam3"])
parser.add_argument("--count", action="store_true", help="카메라별 이미지 수만 출력")
parser.add_argument("--limit", type=int, default=0, help="카메라당 최대 N장 detect")
parser.add_argument("--hours", type=int, nargs=2, metavar=("START", "END"),
                    help="시간대 필터 (예: --hours 8 18 → 08:00~18:00)")
parser.add_argument("--interval", type=int, default=0,
                    help="최소 샘플 간격(초). 중복 프레임 방지 (예: 300 → 5분 간격)")
args = parser.parse_args()

CAPTURES_DIR = "/home/lay/video_indoor/static/captures"
SNAPSHOTS_DIR = "/home/lay/video_indoor/static/snapshots_raw"
GO2K_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
OUT_DIR = "/home/lay/hoban/datasets/captures_labels"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_timestamp(fname):
    """cam1_20260210_131225_0000.jpg → (date_str, hour, minute, second, total_seconds)"""
    parts = fname.split("_")
    if len(parts) < 3:
        return None
    time_str = parts[2]  # 131225
    if len(time_str) < 6:
        return None
    h, m, s = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
    date_str = parts[1]  # 20260210
    total = int(date_str) * 86400 + h * 3600 + m * 60 + s
    return {"date": date_str, "hour": h, "min": m, "sec": s, "total": total}


# go2k 타임스탬프 수집 (중복 제외용)
go2k_keys = set()
for f in os.listdir(GO2K_DIR):
    parts = f.split("_")
    if len(parts) >= 3:
        go2k_keys.add("_".join(parts[:3]))

print(f"go2k 타임스탬프: {len(go2k_keys)}개 (제외 대상)")

# 카메라별 이미지 수집
cam_targets = {}
skipped_dup = 0
skipped_hour = 0
for cam in args.cam:
    cam_dir = os.path.join(CAPTURES_DIR, cam)
    files = []
    for f in sorted(os.listdir(cam_dir)):
        if not f.endswith(".jpg"):
            continue
        # go2k 중복 제외
        parts = f.split("_")
        key = "_".join(parts[:3])
        if key in go2k_keys:
            skipped_dup += 1
            continue
        # 시간대 필터
        if args.hours:
            ts = parse_timestamp(f)
            if ts and not (args.hours[0] <= ts["hour"] < args.hours[1]):
                skipped_hour += 1
                continue
        files.append(f)

    # 간격 필터 (중복 프레임 방지)
    if args.interval > 0 and files:
        sampled = [files[0]]
        last_ts = parse_timestamp(files[0])
        for f in files[1:]:
            ts = parse_timestamp(f)
            if ts and last_ts and (ts["total"] - last_ts["total"]) >= args.interval:
                sampled.append(f)
                last_ts = ts
        files = sampled

    cam_targets[cam] = files

# --count: 통계만 출력
if args.count:
    hour_msg = f" (시간대: {args.hours[0]:02d}:00~{args.hours[1]:02d}:00)" if args.hours else ""
    interval_msg = f", 간격: {args.interval}초" if args.interval > 0 else ""
    print(f"\n필터{hour_msg}{interval_msg}")
    print(f"go2k 중복 제외: {skipped_dup}장, 시간대 제외: {skipped_hour}장\n")
    print(f"{'캠':>5} {'필터후':>8} {'첫 파일':<40} {'끝 파일':<40}")
    print("-" * 100)
    total = 0
    for cam in args.cam:
        files = cam_targets[cam]
        total += len(files)
        first = files[0] if files else "-"
        last = files[-1] if files else "-"
        print(f"{cam:>5} {len(files):>8} {first:<40} {last:<40}")
    print("-" * 100)
    print(f"{'합계':>5} {total:>8}")
    if args.limit > 0:
        limited = sum(min(len(cam_targets[c]), args.limit) for c in args.cam)
        print(f"  --limit {args.limit} 적용 시: {limited}장")
    exit(0)

# --limit: 카메라당 균등 샘플링 (전체에서 균등 분포)
targets = []
for cam in args.cam:
    files = cam_targets[cam]
    if args.limit > 0 and len(files) > args.limit:
        # 균등 간격 샘플링 (최신 편향 방지)
        step = len(files) / args.limit
        files = [files[int(i * step)] for i in range(args.limit)]
    for f in files:
        targets.append((cam, f))

# 필터 요약
filters = []
if args.hours:
    filters.append(f"시간대 {args.hours[0]:02d}:00~{args.hours[1]:02d}:00")
if args.interval > 0:
    filters.append(f"간격 {args.interval}초")
if args.limit > 0:
    filters.append(f"카메라당 {args.limit}장")
filter_msg = ", ".join(filters) if filters else "전체"

print(f"대상: {len(targets)}장 ({filter_msg})")
print(f"  go2k 중복 제외: {skipped_dup}장, 시간대 제외: {skipped_hour}장")
print(f"모델: {args.model}")
print(f"설정: conf={args.conf}, NMS/0.4/IOS\n")

# 모델 로드
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=args.conf,
    device=args.device,
)

# 추론
from PIL import Image

t0 = time.time()
total_boxes = 0
img_with_det = 0

for i, (cam, fname) in enumerate(targets):
    if i % 500 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(targets) - i)
        print(f"  [{i}/{len(targets)}] elapsed={elapsed/60:.1f}m, ETA={eta/60:.1f}m, bbox={total_boxes}")

    img_path = os.path.join(CAPTURES_DIR, cam, fname)

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

    # YOLO format 라벨 저장
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
