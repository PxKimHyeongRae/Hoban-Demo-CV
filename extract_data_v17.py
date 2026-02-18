#!/usr/bin/env python3
"""v17 데이터 수집: helmet_off 추출 + hard negative 마이닝 (v3 - 실서비스 파이프라인)

실서비스(video_indoor)와 동일한 L2+full 파이프라인 사용:
  - 1280x1280 2타일 배치 슬라이스 + full-image 1280px 추론
  - cross_slice_nms + cross_class_nms
  - SAHI 사용 안 함

Mode 1 (helmet_off): captures에서 helmet_off 탐지 이미지 수집
  - 주간(07-17) 이미지만 대상
  - v17 L2+full 추론 → 후처리 → helmet_off 있는 이미지 저장
  - 3k train/val 제외, burst 스킵, 3분 간격 필터

Mode 2 (hard_neg): 사람 없는 배경에서 v17이 오탐하는 이미지 수집
  - 주간(07-17) 이미지만 대상
  - COCO person detector (yolo26m.pt)로 "사람 없음" 확인
  - v17 L2+full에서 탐지 있음 → 오탐 = hard negative
  - 빈 라벨(.txt)과 함께 저장

사용법:
  python extract_data_v17.py                    # 둘 다 순차 실행
  python extract_data_v17.py --mode helmet_off  # helmet_off만
  python extract_data_v17.py --mode hard_neg    # hard negative만
  python extract_data_v17.py --server           # 서버에서 실행
  python extract_data_v17.py --clear            # 이전 결과 삭제 후 재수집
"""
import os
import sys
import zipfile
import time
import logging
import argparse
import shutil
import numpy as np
import cv2
from collections import defaultdict

logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ============================================================================
#  실서비스 파이프라인 (video_indoor 동일)
# ============================================================================

def _calc_slices(img_h, img_w, slice_h, slice_w, overlap_h, overlap_w):
    step_h = int(slice_h * (1 - overlap_h))
    step_w = int(slice_w * (1 - overlap_w))
    slices = []
    y = 0
    while y < img_h:
        y_end = min(y + slice_h, img_h)
        x = 0
        while x < img_w:
            x_end = min(x + slice_w, img_w)
            slices.append((x, y, x_end, y_end))
            if x_end >= img_w:
                break
            x += step_w
        if y_end >= img_h:
            break
        y += step_h
    return slices


def _letterbox(img, target_size=1280):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    dh = (target_size - nh) // 2
    dw = (target_size - nw) // 2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    return canvas, scale, dw, dh


def _compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def _cross_slice_nms(detections, iou_threshold=0.5):
    if not detections:
        return []
    by_class = {}
    for det in detections:
        by_class.setdefault(det[0], []).append(det)
    result = []
    for cls_id, dets in by_class.items():
        dets.sort(key=lambda x: -x[1])
        kept = []
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets
                    if _compute_iou(best[2:], d[2:]) < iou_threshold]
        result.extend(kept)
    return result


def _cross_class_nms(detections, iou_threshold=0.3):
    if len(detections) <= 1:
        return detections
    sorted_dets = sorted(detections, key=lambda x: -x[1])
    keep, suppressed = [], set()
    for i in range(len(sorted_dets)):
        if i in suppressed:
            continue
        keep.append(sorted_dets[i])
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            if sorted_dets[i][0] != sorted_dets[j][0]:
                iou = _compute_iou(sorted_dets[i][2:], sorted_dets[j][2:])
                if iou >= iou_threshold:
                    suppressed.add(j)
    return keep


def _batch_sliced_predict(frame, yolo_model, slice_size=1280, overlap=0.1,
                           conf=0.15, device="cuda:0"):
    """video_indoor 동일: 배치 슬라이스 추론"""
    import torch
    img_h, img_w = frame.shape[:2]
    slices = _calc_slices(img_h, img_w, slice_size, slice_size, overlap, overlap)

    batch_list, metas = [], []
    for (sx, sy, ex, ey) in slices:
        crop = frame[sy:ey, sx:ex]
        lb, scale, dw, dh = _letterbox(crop, slice_size)
        t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        batch_list.append(t)
        metas.append((sx, sy, ex-sx, ey-sy, scale, dw, dh))

    batch_np = np.ascontiguousarray(np.stack(batch_list))
    batch_tensor = torch.from_numpy(batch_np)
    if "cuda" in device:
        batch_tensor = batch_tensor.half().cuda()

    with torch.no_grad():
        raw_preds = yolo_model.model(batch_tensor)
    preds_tensor = raw_preds[0]

    all_dets = []
    for i, (sx, sy, sw, sh, scale, dw, dh) in enumerate(metas):
        preds = preds_tensor[i]
        valid = preds[preds[:, 4] >= conf]
        if len(valid) == 0:
            continue
        for det in valid:
            x1, y1, x2, y2 = det[:4].cpu().numpy()
            conf_val = float(det[4])
            cls_id = int(det[5])
            x1 = (x1 - dw) / scale + sx
            y1 = (y1 - dh) / scale + sy
            x2 = (x2 - dw) / scale + sx
            y2 = (y2 - dh) / scale + sy
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            if x2 > x1 and y2 > y1:
                all_dets.append((cls_id, conf_val, x1, y1, x2, y2))

    return _cross_slice_nms(all_dets, 0.5)


def _full_image_predict(frame, yolo_model, conf=0.15, device="cuda:0"):
    """video_indoor 동일: 전체 이미지 추론"""
    import torch
    img_h, img_w = frame.shape[:2]
    lb, scale, dw, dh = _letterbox(frame, 1280)
    t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    batch_tensor = torch.from_numpy(t[np.newaxis])
    if "cuda" in device:
        batch_tensor = batch_tensor.half().cuda()

    with torch.no_grad():
        raw_preds = yolo_model.model(batch_tensor)
    preds_tensor = raw_preds[0][0]
    valid = preds_tensor[preds_tensor[:, 4] >= conf]

    dets = []
    for det in valid:
        x1, y1, x2, y2 = det[:4].cpu().numpy()
        conf_val = float(det[4])
        cls_id = int(det[5])
        x1 = (x1 - dw) / scale
        y1 = (y1 - dh) / scale
        x2 = (x2 - dw) / scale
        y2 = (y2 - dh) / scale
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 > x1 and y2 > y1:
            dets.append((cls_id, conf_val, x1, y1, x2, y2))
    return dets


def predict_l2_full(frame, yolo_model, conf=0.15, device="cuda:0"):
    """L2+full 파이프라인: 슬라이스 2타일 + full-image → 병합 → NMS"""
    slice_dets = _batch_sliced_predict(frame, yolo_model, 1280, 0.1, conf, device)
    full_dets = _full_image_predict(frame, yolo_model, conf, device)
    merged = _cross_slice_nms(slice_dets + full_dets, 0.5)
    merged = _cross_class_nms(merged, 0.3)
    return merged


# ============================================================================
#  유틸리티
# ============================================================================

def parse_timestamp(fname):
    """파일명에서 시간(초) 추출: cam1_20260212_122355_xxxxx.jpg"""
    parts = fname.split("_")
    if len(parts) < 3:
        return -1
    ts = parts[2]
    if len(ts) < 6:
        return -1
    try:
        h, m, s = int(ts[:2]), int(ts[2:4]), int(ts[4:6])
        return h * 3600 + m * 60 + s
    except ValueError:
        return -1


def parse_hour(fname):
    parts = fname.split("_")
    if len(parts) < 3:
        return -1
    ts = parts[2]
    if len(ts) < 2:
        return -1
    try:
        return int(ts[:2])
    except ValueError:
        return -1


def is_daytime(fname, start_hour=7, end_hour=17):
    hour = parse_hour(fname)
    return start_hour <= hour < end_hour


def load_exclusion_set(base, extra_dirs=None):
    used = set()
    dirs = [
        os.path.join(base, "home/lay/hoban/datasets/3k_finetune/train/images"),
        os.path.join(base, "home/lay/hoban/datasets/3k_finetune/val/images"),
        os.path.join(base, "home/lay/hoban/datasets_go3k_v16/train/images"),
        os.path.join(base, "home/lay/hoban/datasets_go3k_v16/valid/images"),
    ]
    if extra_dirs:
        dirs.extend(extra_dirs)
    for d in dirs:
        if os.path.isdir(d):
            used.update(os.listdir(d))
    return used


def collect_candidates(cap_dir, cameras, used, already_found, interval=5,
                       daytime_only=True, start_hour=7, end_hour=17):
    candidates = []
    for cam in cameras:
        cam_dir = os.path.join(cap_dir, cam)
        if not os.path.isdir(cam_dir):
            print(f"  {cam}: 디렉터리 없음 ({cam_dir})")
            continue
        all_files = sorted(f for f in os.listdir(cam_dir)
                          if f.endswith(".jpg") and f not in used and f not in already_found)
        if daytime_only:
            day_files = [f for f in all_files if is_daytime(f, start_hour, end_hour)]
            sampled = day_files[::interval]
            print(f"  {cam}: 전체 {len(all_files)} -> 주간({start_hour}-{end_hour}시) "
                  f"{len(day_files)} -> 샘플 {len(sampled)}")
        else:
            sampled = all_files[::interval]
            print(f"  {cam}: 전체 {len(all_files)} -> 샘플 {len(sampled)}")
        candidates.extend([(cam, f) for f in sampled])
    return candidates


def time_gap_filter(result_dir, images, min_gap=180, target=None):
    by_cam_date = defaultdict(list)
    for fname in images:
        parts = fname.split("_")
        if len(parts) < 3:
            continue
        cam = parts[0]
        sec = parse_timestamp(fname)
        if sec < 0:
            continue
        lbl_path = os.path.join(result_dir, "labels", fname.replace(".jpg", ".txt"))
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                labels = f.read().strip()
        else:
            labels = ""
        n_off = sum(1 for l in labels.split("\n") if l.startswith("1 ")) if labels else 0
        by_cam_date[f"{cam}_{parts[1]}"].append((sec, cam, fname, labels, n_off))

    filtered = []
    for key in sorted(by_cam_date.keys()):
        entries = sorted(by_cam_date[key])
        last_sec = -99999
        for sec, cam, fname, labels, n_off in entries:
            if sec - last_sec >= min_gap:
                filtered.append((cam, fname, labels, n_off))
                last_sec = sec

    filtered.sort(key=lambda x: -x[3])
    if target and len(filtered) > target:
        filtered = filtered[:target]
    return filtered


def package_cvat(out_dir, filtered, class_names, prefix=""):
    ann_path = os.path.join(out_dir, f"{prefix}annotations.zip")
    with zipfile.ZipFile(ann_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.data",
                     f"classes = {len(class_names)}\n"
                     f"train = data/train.txt\n"
                     f"names = data/obj.names\n"
                     f"backup = backup/\n")
        zf.writestr("obj.names", "\n".join(class_names) + "\n")
        train_lines = []
        for cam, fname, labels, _ in filtered:
            train_lines.append(f"data/obj_train_data/{fname}")
            zf.writestr(f"obj_train_data/{fname.replace('.jpg', '.txt')}",
                        (labels + "\n") if labels else "\n")
        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    img_zip_path = os.path.join(out_dir, f"{prefix}images.zip")
    result_img_dir = os.path.join(out_dir, "results/images")
    with zipfile.ZipFile(img_zip_path, "w", zipfile.ZIP_STORED) as zf:
        for i, (cam, fname, _, _) in enumerate(filtered):
            if i % 50 == 0:
                print(f"  zip: {i}/{len(filtered)}...", flush=True)
            img_path = os.path.join(result_img_dir, fname)
            if os.path.exists(img_path):
                zf.write(img_path, fname)

    print(f"  annotations: {os.path.getsize(ann_path)/1024/1024:.1f}MB")
    print(f"  images: {os.path.getsize(img_zip_path)/1024/1024:.1f}MB")
    return ann_path, img_zip_path


# ============================================================================
#  Mode 1: helmet_off 추출 (L2+full 파이프라인)
# ============================================================================
def run_helmet_off(args, base, model_path, cap_dir):
    print("\n" + "=" * 60)
    print("  MODE 1: helmet_off 추출 (L2+full, 주간 07-17시)")
    print("=" * 60)

    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
    CAMERAS = ["cam1", "cam2"]

    OUT_DIR = os.path.join(base, "home/lay/hoban/datasets/helmet_off_v17")
    RESULT_DIR = os.path.join(OUT_DIR, "results")
    os.makedirs(os.path.join(RESULT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "labels"), exist_ok=True)

    used = load_exclusion_set(base)
    already_found = set(os.listdir(os.path.join(RESULT_DIR, "images")))
    print(f"3k 제외: {len(used)}장, 이전 결과: {len(already_found)}장")

    candidates = collect_candidates(
        cap_dir, CAMERAS, used, already_found, args.interval,
        daytime_only=True, start_hour=args.start_hour, end_hour=args.end_hour)
    print(f"스캔 대상: {len(candidates)}장")

    if not candidates:
        print("스캔할 이미지 없음")
        return

    from ultralytics import YOLO
    print(f"v17 모델 로드 (FP16+fuse, device={args.device})")
    yolo_model = YOLO(model_path)
    yolo_model.fuse()
    if "cuda" in args.device:
        yolo_model.model.to(args.device)
        yolo_model.model.half()

    BURST_WINDOW, BURST_THRESHOLD, BURST_SKIP = 100, 40, 200
    new_found, processed = 0, 0
    window_hits, window_start, skip_until = 0, 0, 0
    t_start = time.time()

    for idx, (cam, fname) in enumerate(candidates):
        if idx < skip_until:
            continue

        img_path = os.path.join(cap_dir, cam, fname)
        processed += 1

        # burst 윈도우
        if processed - window_start >= BURST_WINDOW:
            if window_hits >= BURST_THRESHOLD:
                skip_until = idx + BURST_SKIP
                print(f"  burst 감지 ({window_hits}개/100장) -> {BURST_SKIP}장 스킵")
            window_hits = 0
            window_start = processed

        # L2+full 추론 (실서비스 동일)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]
        img_area = img_h * img_w

        preds = predict_l2_full(frame, yolo_model, conf=0.15, device=args.device)

        # per-class conf 필터
        preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in preds
                 if s >= (0.40 if c == 0 else 0.15)]

        has_off = any(c == 1 for c, *_ in preds)
        if not has_off:
            continue

        new_found += 1
        window_hits += 1

        # YOLO format 라벨
        label_lines = []
        for cls_id, conf, x1, y1, x2, y2 in preds:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        shutil.copy2(img_path, os.path.join(RESULT_DIR, "images", fname))
        with open(os.path.join(RESULT_DIR, "labels", fname.replace(".jpg", ".txt")), "w") as f:
            f.write("\n".join(label_lines) + "\n")

        if processed % 100 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed
            total = len(already_found) + new_found
            print(f"  [{processed}/{len(candidates)}] "
                  f"신규: {new_found} | 누적: {total}장 "
                  f"({rate:.1f} img/s, ETA: {(len(candidates)-processed)/rate/60:.1f}분)")
            if total >= args.target_off:
                print(f"  목표 달성! ({total}장)")
                break

        if len(already_found) + new_found >= args.target_off:
            break

    elapsed = time.time() - t_start
    total = len(already_found) + new_found
    print(f"\nhelmet_off 수집 완료: {total}장 (신규 {new_found}, {elapsed:.0f}s)")

    all_images = list(os.listdir(os.path.join(RESULT_DIR, "images")))
    filtered = time_gap_filter(RESULT_DIR, all_images, args.min_gap, args.target_off)
    print(f"3분 간격 필터 후: {len(filtered)}장")

    package_cvat(OUT_DIR, filtered, CLASS_NAMES)
    print(f"출력: {OUT_DIR}")
    return new_found


# ============================================================================
#  Mode 2: hard negative 마이닝 (COCO + L2+full)
# ============================================================================
def run_hard_neg(args, base, model_path, cap_dir):
    """COCO에서 사람 없음 + v17 L2+full에서 탐지 있음 = hard negative"""
    print("\n" + "=" * 60)
    print("  MODE 2: hard negative 마이닝 (COCO 검증 + L2+full, 주간 07-17시)")
    print("=" * 60)

    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
    CAMERAS = ["cam1", "cam2"]

    OUT_DIR = os.path.join(base, "home/lay/hoban/datasets/hard_neg_v17")
    RESULT_DIR = os.path.join(OUT_DIR, "results")
    os.makedirs(os.path.join(RESULT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "labels"), exist_ok=True)

    used = load_exclusion_set(base)
    already_found = set(os.listdir(os.path.join(RESULT_DIR, "images")))
    print(f"3k 제외: {len(used)}장, 이전 결과: {len(already_found)}장")

    candidates = collect_candidates(
        cap_dir, CAMERAS, used, already_found, args.interval * 2,
        daytime_only=True, start_hour=args.start_hour, end_hour=args.end_hour)
    print(f"스캔 대상: {len(candidates)}장")

    if not candidates:
        print("스캔할 이미지 없음")
        return

    from ultralytics import YOLO

    # COCO person detector
    coco_model_path = os.path.join(base, "home/lay/hoban/yolo26m.pt")
    print(f"COCO person detector 로드: {coco_model_path}")
    coco_model = YOLO(coco_model_path)
    coco_model.fuse()
    if "cuda" in args.device:
        coco_model.model.to(args.device)
        coco_model.model.half()

    # v17 모델 (L2+full 추론용)
    print(f"v17 모델 로드 (FP16+fuse, device={args.device})")
    v17_model = YOLO(model_path)
    v17_model.fuse()
    if "cuda" in args.device:
        v17_model.model.to(args.device)
        v17_model.model.half()

    COCO_PERSON_CLASS = 0
    COCO_PERSON_CONF = 0.25
    V17_MIN_CONF = 0.30  # v17에서 이 conf 이상 탐지가 있으면 오탐 후보

    new_found, processed, skipped_has_person = 0, 0, 0
    t_start = time.time()

    for idx, (cam, fname) in enumerate(candidates):
        img_path = os.path.join(cap_dir, cam, fname)
        processed += 1

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Step 1: COCO person detector로 사람 유무 확인 (full-image)
        coco_dets = _full_image_predict(frame, coco_model, conf=0.10, device=args.device)
        has_person = any(cls_id == COCO_PERSON_CLASS and conf >= COCO_PERSON_CONF
                        for cls_id, conf, *_ in coco_dets)

        if has_person:
            skipped_has_person += 1
            continue

        # Step 2: COCO 사람 없음 → v17 L2+full 추론
        v17_dets = predict_l2_full(frame, v17_model, conf=0.15, device=args.device)

        # per-class conf 적용 후 탐지가 있으면 = 오탐
        v17_filtered = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in v17_dets
                        if s >= V17_MIN_CONF]

        if not v17_filtered:
            continue

        # Hard negative!
        new_found += 1
        shutil.copy2(img_path, os.path.join(RESULT_DIR, "images", fname))
        with open(os.path.join(RESULT_DIR, "labels", fname.replace(".jpg", ".txt")), "w") as f:
            f.write("")

        if processed % 100 == 0 or new_found % 10 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            total = len(already_found) + new_found
            print(f"  [{processed}/{len(candidates)}] "
                  f"hard_neg: {new_found} | COCO사람: {skipped_has_person} "
                  f"({rate:.1f} img/s)")
            if total >= args.target_neg:
                print(f"  목표 달성! ({total}장)")
                break

        if len(already_found) + new_found >= args.target_neg:
            break

    elapsed = time.time() - t_start
    total = len(already_found) + new_found
    print(f"\nhard negative 수집 완료: {total}장 (신규 {new_found}, {elapsed:.0f}s)")
    print(f"  COCO 사람 탐지로 스킵: {skipped_has_person}장")

    all_images = list(os.listdir(os.path.join(RESULT_DIR, "images")))
    filtered = time_gap_filter(RESULT_DIR, all_images, args.min_gap, args.target_neg)
    print(f"3분 간격 필터 후: {len(filtered)}장")

    package_cvat(OUT_DIR, filtered, CLASS_NAMES)
    print(f"출력: {OUT_DIR}")
    return new_found


# ============================================================================
#  Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="v17 데이터 수집 (v3 - 실서비스 L2+full 파이프라인)")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--mode", choices=["all", "helmet_off", "hard_neg"],
                        default="all")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--target-off", type=int, default=400)
    parser.add_argument("--target-neg", type=int, default=300)
    parser.add_argument("--min-gap", type=int, default=180)
    parser.add_argument("--start-hour", type=int, default=7)
    parser.add_argument("--end-hour", type=int, default=17)
    parser.add_argument("--clear", action="store_true",
                        help="이전 결과 삭제 후 재수집")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    BASE = "/" if args.server else "Z:/"
    MODEL_PATH = os.path.join(BASE, "home/lay/hoban/hoban_go3k_v17/weights/best.pt")
    CAP_DIR = os.path.join(BASE, "home/lay/video_indoor/static/captures")

    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device: {args.device}")

    if not os.path.exists(MODEL_PATH):
        print(f"모델 없음: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.isdir(CAP_DIR):
        print(f"캡처 디렉터리 없음: {CAP_DIR}")
        sys.exit(1)

    print(f"모델: {MODEL_PATH}")
    print(f"캡처: {CAP_DIR}")
    print(f"파이프라인: L2+full (실서비스 동일)")
    print(f"주간 필터: {args.start_hour:02d}:00 ~ {args.end_hour:02d}:00")

    if args.clear:
        for mode_dir in ["helmet_off_v17", "hard_neg_v17"]:
            result_dir = os.path.join(BASE, f"home/lay/hoban/datasets/{mode_dir}/results")
            if os.path.isdir(result_dir):
                shutil.rmtree(result_dir)
                print(f"삭제: {result_dir}")

    if args.mode in ("all", "helmet_off"):
        run_helmet_off(args, BASE, MODEL_PATH, CAP_DIR)

    if args.mode in ("all", "hard_neg"):
        run_hard_neg(args, BASE, MODEL_PATH, CAP_DIR)

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
