#!/usr/bin/env python3
"""
captures cam1+cam2에서 SAHI로 helmet_off 탐지 → CVAT 패키징

- v16 모델 + SAHI (1280x720 타일)
- 이전 결과 자동 스킵 (이어서 추출 가능)
- 3k train/val 제외, 3분 간격 필터

로컬: python extract_helmet_off.py
서버: python extract_helmet_off.py --server
"""
import os
import sys
import zipfile
import time
import logging
import argparse
from collections import defaultdict

# SAHI verbose 끄기
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--interval", type=int, default=5, help="캡처 샘플링 간격")
    parser.add_argument("--target", type=int, default=400, help="목표 helmet_off 이미지 수")
    parser.add_argument("--min-gap", type=int, default=180, help="최소 시간 간격 (초)")
    args = parser.parse_args()

    BASE = "/" if args.server else "Z:/"

    CAP_DIR = os.path.join(BASE, "home/lay/video_indoor/static/captures")
    TRAIN_DIR = os.path.join(BASE, "home/lay/hoban/datasets/3k_finetune/train/images")
    VAL_DIR = os.path.join(BASE, "home/lay/hoban/datasets/3k_finetune/val/images")
    MODEL_PATH = os.path.join(BASE, "home/lay/hoban/hoban_go3k_v16_640/weights/best.pt")
    OUT_DIR = os.path.join(BASE, "home/lay/hoban/datasets/cvat_helmet_off")
    RESULT_DIR = os.path.join(OUT_DIR, "results")  # 개별 결과 저장
    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
    CAMERAS = ["cam1", "cam2"]

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "labels"), exist_ok=True)

    # 1. 제외 목록: 3k + 이전 결과
    used = set()
    for d in [TRAIN_DIR, VAL_DIR]:
        used.update(os.listdir(d))

    already_found = set(os.listdir(os.path.join(RESULT_DIR, "images")))
    print(f"3k 제외: {len(used)}장, 이전 결과: {len(already_found)}장 (스킵)")

    # 2. 캡처 수집 (주간, 순차)
    candidates = []
    for cam in CAMERAS:
        cam_dir = os.path.join(CAP_DIR, cam)
        all_files = sorted(f for f in os.listdir(cam_dir)
                          if f.endswith(".jpg") and f not in used and f not in already_found)
        sampled = all_files[::args.interval]
        candidates.extend([(cam, f) for f in sampled])
        print(f"  {cam}: 전체 {len(all_files)} → 샘플 {len(sampled)}")

    print(f"\n스캔 대상: {len(candidates)}장")

    # 3. SAHI 모델 로드
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from PIL import Image
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"SAHI 모델 로드 (device={device})")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=MODEL_PATH,
        confidence_threshold=args.conf,
        device=device,
    )

    # 4. SAHI 추론 (burst 구간 스킵: 100장당 10개 이상 탐지 시 다음 200장 건너뜀)
    BURST_WINDOW = 100
    BURST_THRESHOLD = 10
    BURST_SKIP = 200

    new_found = 0
    processed = 0
    window_hits = 0
    window_start = 0
    skip_until = 0
    t_start = time.time()

    for idx, (cam, fname) in enumerate(candidates):
        if idx < skip_until:
            continue

        img_path = os.path.join(CAP_DIR, cam, fname)
        processed += 1

        # burst 윈도우 리셋
        if processed - window_start >= BURST_WINDOW:
            if window_hits >= BURST_THRESHOLD:
                skip_until = idx + BURST_SKIP
                print(f"  ⚠ burst 감지 ({window_hits}개/100장) → {BURST_SKIP}장 스킵")
            window_hits = 0
            window_start = processed

        try:
            result = get_sliced_prediction(
                img_path, sahi_model,
                slice_height=720, slice_width=1280,
                overlap_height_ratio=0.15, overlap_width_ratio=0.15,
                perform_standard_pred=True,
                postprocess_type="NMS",
                postprocess_match_threshold=0.4,
                postprocess_match_metric="IOS",
                verbose=0,
            )
        except Exception:
            continue

        has_off = False
        label_lines = []
        img = Image.open(img_path)
        img_w, img_h = img.size

        for pred in result.object_prediction_list:
            cls_id = pred.category.id
            x1, y1 = pred.bbox.minx, pred.bbox.miny
            x2, y2 = pred.bbox.maxx, pred.bbox.maxy
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if cls_id == 1:
                has_off = True

        if has_off:
            new_found += 1
            window_hits += 1
            # 이미지 복사 + 라벨 저장
            import shutil
            shutil.copy2(img_path, os.path.join(RESULT_DIR, "images", fname))
            lbl_name = fname.replace(".jpg", ".txt")
            with open(os.path.join(RESULT_DIR, "labels", lbl_name), "w") as f:
                f.write("\n".join(label_lines) + "\n")

        if processed % 100 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed
            eta = (len(candidates) - processed) / rate if rate > 0 else 0
            total = len(already_found) + new_found
            print(f"  [{processed}/{len(candidates)}] "
                  f"신규: {new_found} | 누적: {total}장 "
                  f"({rate:.1f} img/s, ETA: {eta/60:.1f}분)")

            if total >= args.target:
                print(f"\n  목표 달성! ({total}장)")
                break

        if len(already_found) + new_found >= args.target:
            break

    # 5. 전체 결과 (이전 + 신규) 시간 간격 필터
    all_images = list(os.listdir(os.path.join(RESULT_DIR, "images")))
    print(f"\n전체 결과: {len(all_images)}장 (이전 {len(already_found)} + 신규 {new_found})")

    by_cam_date = defaultdict(list)
    for fname in all_images:
        parts = fname.split("_")
        cam = parts[0]
        h, m, s = int(parts[2][:2]), int(parts[2][2:4]), int(parts[2][4:6])
        sec = h * 3600 + m * 60 + s
        lbl_path = os.path.join(RESULT_DIR, "labels", fname.replace(".jpg", ".txt"))
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                labels = f.read().strip()
        else:
            continue
        n_off = sum(1 for l in labels.split("\n") if l.startswith("1 "))
        by_cam_date[f"{cam}_{parts[1]}"].append((sec, cam, fname, labels, n_off))

    filtered = []
    for key in sorted(by_cam_date.keys()):
        entries = sorted(by_cam_date[key])
        last_sec = -99999
        for sec, cam, fname, labels, n_off in entries:
            if sec - last_sec >= args.min_gap:
                filtered.append((cam, fname, labels, n_off))
                last_sec = sec

    filtered.sort(key=lambda x: -x[3])
    if len(filtered) > args.target:
        filtered = filtered[:args.target]

    print(f"3분 간격 필터 후: {len(filtered)}장")

    # 6. CVAT 패키징
    ann_path = os.path.join(OUT_DIR, "annotations.zip")
    with zipfile.ZipFile(ann_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.data",
                     f"classes = {len(CLASS_NAMES)}\n"
                     f"train = data/train.txt\n"
                     f"names = data/obj.names\n"
                     f"backup = backup/\n")
        zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")
        train_lines = []
        for cam, fname, labels, _ in filtered:
            train_lines.append(f"data/obj_train_data/{fname}")
            zf.writestr(f"obj_train_data/{fname.replace('.jpg', '.txt')}", labels + "\n")
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    img_zip_path = os.path.join(OUT_DIR, "images.zip")
    with zipfile.ZipFile(img_zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, (cam, fname, _, _) in enumerate(filtered):
            if i % 50 == 0:
                print(f"  압축: {i}/{len(filtered)}...", flush=True)
            img_path = os.path.join(RESULT_DIR, "images", fname)
            if os.path.exists(img_path):
                zf.write(img_path, fname)
            else:
                zf.write(os.path.join(CAP_DIR, cam, fname), fname)

    print(f"\n{'='*60}")
    print(f"완료! {len(filtered)}장")
    print(f"  annotations.zip: {os.path.getsize(ann_path)/1024/1024:.1f}MB")
    print(f"  images.zip: {os.path.getsize(img_zip_path)/1024/1024:.1f}MB")
    print(f"  출력: {OUT_DIR}")


if __name__ == "__main__":
    main()
