#!/usr/bin/env python3
"""v17 데이터 수집 (v5 - 멀티프로세스 GPU 병렬 추론)

실서비스(video_indoor)와 동일한 L2+full 파이프라인 + 멀티프로세스 병렬:
  - N개 워커 프로세스가 각각 모델 로드 → 후보 분할 처리
  - 단일 GPU를 N개 프로세스가 공유 (CUDA 드라이버가 스케줄링)
  - RTX 4080 16GB: helmet_off 4워커, hard_neg 2워커

사용법:
  python extract_data_v17.py --server                       # 서버 (자동 워커 수)
  python extract_data_v17.py --server --workers 4            # 워커 수 수동
  python extract_data_v17.py --server --clear                # 이전 결과 삭제 후 재수집
  python extract_data_v17.py --server --mode helmet_off      # helmet_off만
  python extract_data_v17.py --server --mode hard_neg        # hard negative만
  python extract_data_v17.py                                 # 로컬 PC (자동)
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

# stdout 버퍼링 방지 (백그라운드 실행 시 필수)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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


# ============================================================================
#  GPU 배치 추론
# ============================================================================

def _parse_preds(preds_single, conf, img_h, img_w, scale, dw, dh,
                 offset_x=0, offset_y=0):
    """단일 텐서에서 탐지 결과 파싱"""
    valid = preds_single[preds_single[:, 4] >= conf]
    dets = []
    for det in valid:
        x1, y1, x2, y2 = det[:4].cpu().numpy()
        conf_val = float(det[4])
        cls_id = int(det[5])
        x1 = (x1 - dw) / scale + offset_x
        y1 = (y1 - dh) / scale + offset_y
        x2 = (x2 - dw) / scale + offset_x
        y2 = (y2 - dh) / scale + offset_y
        x1, y1 = max(0, min(x1, img_w)), max(0, min(y1, img_h))
        x2, y2 = max(0, min(x2, img_w)), max(0, min(y2, img_h))
        if x2 > x1 and y2 > y1:
            dets.append((cls_id, conf_val, x1, y1, x2, y2))
    return dets


def _batch_predict_l2_full(frames, yolo_model, conf=0.15, device="cuda:0",
                            max_gpu_batch=16):
    """N개 프레임 L2+full 배치 추론"""
    import torch
    n = len(frames)
    if n == 0:
        return []

    slice_tiles, slice_metas = [], []
    full_tiles, full_metas = [], []
    frame_info = []

    for fi, frame in enumerate(frames):
        if frame is None:
            frame_info.append((0, 0))
            continue
        img_h, img_w = frame.shape[:2]
        frame_info.append((img_h, img_w))

        slices = _calc_slices(img_h, img_w, 1280, 1280, 0.1, 0.1)
        for (sx, sy, ex, ey) in slices:
            crop = frame[sy:ey, sx:ex]
            lb, scale, dw, dh = _letterbox(crop, 1280)
            t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            slice_tiles.append(t)
            slice_metas.append((fi, sx, sy, scale, dw, dh))

        lb, scale, dw, dh = _letterbox(frame, 1280)
        t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        full_tiles.append(t)
        full_metas.append((fi, scale, dw, dh))

    per_frame_dets = [[] for _ in range(n)]

    for start in range(0, len(slice_tiles), max_gpu_batch):
        batch = slice_tiles[start:start + max_gpu_batch]
        metas = slice_metas[start:start + max_gpu_batch]
        batch_np = np.ascontiguousarray(np.stack(batch))
        batch_tensor = torch.from_numpy(batch_np)
        if "cuda" in device:
            batch_tensor = batch_tensor.half().to(device)
        with torch.no_grad():
            preds = yolo_model.model(batch_tensor)[0]
        for i, (fi, sx, sy, scale, dw, dh) in enumerate(metas):
            img_h, img_w = frame_info[fi]
            dets = _parse_preds(preds[i], conf, img_h, img_w, scale, dw, dh, sx, sy)
            per_frame_dets[fi].extend(dets)

    for start in range(0, len(full_tiles), max_gpu_batch):
        batch = full_tiles[start:start + max_gpu_batch]
        metas = full_metas[start:start + max_gpu_batch]
        batch_np = np.ascontiguousarray(np.stack(batch))
        batch_tensor = torch.from_numpy(batch_np)
        if "cuda" in device:
            batch_tensor = batch_tensor.half().to(device)
        with torch.no_grad():
            preds = yolo_model.model(batch_tensor)[0]
        for i, (fi, scale, dw, dh) in enumerate(metas):
            img_h, img_w = frame_info[fi]
            dets = _parse_preds(preds[i], conf, img_h, img_w, scale, dw, dh)
            per_frame_dets[fi].extend(dets)

    results = []
    for fi in range(n):
        merged = _cross_slice_nms(per_frame_dets[fi], 0.5)
        merged = _cross_class_nms(merged, 0.3)
        results.append(merged)
    return results


def _batch_full_image_predict(frames, yolo_model, conf=0.10, device="cuda:0",
                               max_gpu_batch=16):
    """N개 프레임 full-image 배치 추론"""
    import torch
    n = len(frames)
    per_frame_dets = [[] for _ in range(n)]

    tiles, metas, frame_info = [], [], []
    for fi, frame in enumerate(frames):
        if frame is None:
            frame_info.append((0, 0))
            continue
        img_h, img_w = frame.shape[:2]
        frame_info.append((img_h, img_w))
        lb, scale, dw, dh = _letterbox(frame, 1280)
        t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tiles.append(t)
        metas.append((fi, scale, dw, dh))

    for start in range(0, len(tiles), max_gpu_batch):
        batch = tiles[start:start + max_gpu_batch]
        batch_metas = metas[start:start + max_gpu_batch]
        batch_np = np.ascontiguousarray(np.stack(batch))
        batch_tensor = torch.from_numpy(batch_np)
        if "cuda" in device:
            batch_tensor = batch_tensor.half().to(device)
        with torch.no_grad():
            preds = yolo_model.model(batch_tensor)[0]
        for i, (fi, scale, dw, dh) in enumerate(batch_metas):
            img_h, img_w = frame_info[fi]
            dets = _parse_preds(preds[i], conf, img_h, img_w, scale, dw, dh)
            per_frame_dets[fi].extend(dets)

    return per_frame_dets


# ============================================================================
#  유틸리티
# ============================================================================

def parse_timestamp(fname):
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
            print(f"  {cam}: 디렉터리 없음 ({cam_dir})", flush=True)
            continue
        all_files = sorted(f for f in os.listdir(cam_dir)
                          if f.endswith(".jpg") and f not in used and f not in already_found)
        if daytime_only:
            day_files = [f for f in all_files if is_daytime(f, start_hour, end_hour)]
            sampled = day_files[::interval]
            print(f"  {cam}: 전체 {len(all_files)} -> 주간({start_hour}-{end_hour}시) "
                  f"{len(day_files)} -> 샘플 {len(sampled)}", flush=True)
        else:
            sampled = all_files[::interval]
            print(f"  {cam}: 전체 {len(all_files)} -> 샘플 {len(sampled)}", flush=True)
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

    print(f"  annotations: {os.path.getsize(ann_path)/1024/1024:.1f}MB", flush=True)
    print(f"  images: {os.path.getsize(img_zip_path)/1024/1024:.1f}MB", flush=True)
    return ann_path, img_zip_path


# ============================================================================
#  워커 함수 (서브프로세스에서 실행)
# ============================================================================

def _helmet_off_worker(worker_id, candidates, model_path, cap_dir,
                       img_dir, lbl_dir, device, batch_size, max_gpu_batch):
    """helmet_off 워커: 모델 로드 → 배치 추론 → 결과 저장"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    model.fuse()
    if "cuda" in device:
        model.model.to(device)
        model.model.half()

    found, processed = 0, 0
    t_start = time.time()

    for ci in range(0, len(candidates), batch_size):
        chunk = candidates[ci:ci + batch_size]
        paths = [os.path.join(cap_dir, cam, fname) for cam, fname in chunk]
        frames = [cv2.imread(p) for p in paths]

        valid = [(i, f) for i, f in enumerate(frames) if f is not None]
        processed += len(chunk)

        if not valid:
            continue

        valid_frames = [f for _, f in valid]
        all_dets = _batch_predict_l2_full(
            valid_frames, model, conf=0.15,
            device=device, max_gpu_batch=max_gpu_batch)

        for vi, (ci_local, frame) in enumerate(valid):
            cam, fname = chunk[ci_local]
            img_h, img_w = frame.shape[:2]

            preds = all_dets[vi]
            preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in preds
                     if s >= (0.40 if c == 0 else 0.15)]

            if not any(c == 1 for c, *_ in preds):
                continue

            found += 1
            label_lines = []
            for cls_id, conf_val, x1, y1, x2, y2 in preds:
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            shutil.copy2(os.path.join(cap_dir, cam, fname),
                        os.path.join(img_dir, fname))
            with open(os.path.join(lbl_dir, fname.replace(".jpg", ".txt")), "w") as f:
                f.write("\n".join(label_lines) + "\n")

        # 진행률 (200장마다)
        if processed % 200 < batch_size:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  [W{worker_id}] {processed}/{len(candidates)} "
                  f"found={found} ({rate:.1f} img/s)", flush=True)

    elapsed = time.time() - t_start
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"  [W{worker_id}] 완료: {found}장 발견 "
          f"({processed}장 처리, {rate:.1f} img/s, {elapsed:.0f}s)", flush=True)
    return found


def _hard_neg_worker(worker_id, candidates, v17_path, coco_path, cap_dir,
                     img_dir, lbl_dir, device, batch_size, max_gpu_batch):
    """hard_neg 워커: COCO 사람검출 + v17 오탐 = hard negative"""
    from ultralytics import YOLO

    coco_model = YOLO(coco_path)
    coco_model.fuse()
    if "cuda" in device:
        coco_model.model.to(device)
        coco_model.model.half()

    v17_model = YOLO(v17_path)
    v17_model.fuse()
    if "cuda" in device:
        v17_model.model.to(device)
        v17_model.model.half()

    COCO_PERSON_CLASS = 0
    COCO_PERSON_CONF = 0.25
    V17_MIN_CONF = 0.30

    found, processed, skipped_person = 0, 0, 0
    t_start = time.time()

    for ci in range(0, len(candidates), batch_size):
        chunk = candidates[ci:ci + batch_size]
        paths = [os.path.join(cap_dir, cam, fname) for cam, fname in chunk]
        frames = [cv2.imread(p) for p in paths]

        valid = [(i, f) for i, f in enumerate(frames) if f is not None]
        processed += len(chunk)

        if not valid:
            continue

        valid_frames = [f for _, f in valid]

        # Step 1: COCO person 배치 탐지
        coco_dets = _batch_full_image_predict(
            valid_frames, coco_model, conf=0.10,
            device=device, max_gpu_batch=max_gpu_batch)

        # Step 2: 사람 없는 프레임만 필터
        no_person_indices = []
        for vi, (ci_local, frame) in enumerate(valid):
            has_person = any(cls_id == COCO_PERSON_CLASS and conf_val >= COCO_PERSON_CONF
                           for cls_id, conf_val, *_ in coco_dets[vi])
            if has_person:
                skipped_person += 1
            else:
                no_person_indices.append(vi)

        if not no_person_indices:
            continue

        # Step 3: v17 L2+full 배치 추론 (사람 없는 프레임만)
        no_person_frames = [valid_frames[vi] for vi in no_person_indices]
        v17_dets = _batch_predict_l2_full(
            no_person_frames, v17_model, conf=0.15,
            device=device, max_gpu_batch=max_gpu_batch)

        for idx, vi in enumerate(no_person_indices):
            ci_local = valid[vi][0]
            cam, fname = chunk[ci_local]
            dets = v17_dets[idx]
            filtered_dets = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in dets
                             if s >= V17_MIN_CONF]
            if not filtered_dets:
                continue

            found += 1
            shutil.copy2(os.path.join(cap_dir, cam, fname),
                        os.path.join(img_dir, fname))
            with open(os.path.join(lbl_dir, fname.replace(".jpg", ".txt")), "w") as f:
                f.write("")

        if processed % 200 < batch_size:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  [W{worker_id}] {processed}/{len(candidates)} "
                  f"neg={found} person_skip={skipped_person} ({rate:.1f} img/s)",
                  flush=True)

    elapsed = time.time() - t_start
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"  [W{worker_id}] 완료: {found}장 ({processed}장, "
          f"person_skip={skipped_person}, {rate:.1f} img/s, {elapsed:.0f}s)",
          flush=True)
    return found


# ============================================================================
#  멀티프로세스 코디네이터
# ============================================================================

def _run_workers_subprocess(worker_type, n_workers, candidates, args_dict):
    """N개 서브프로세스로 병렬 처리 (CUDA 안전)"""
    import subprocess
    import threading
    import json
    import tempfile

    # 후보 분할 (라운드 로빈)
    shards = [candidates[i::n_workers] for i in range(n_workers)]
    for i, shard in enumerate(shards):
        print(f"  워커 {i}: {len(shard)}장 할당", flush=True)

    # 각 워커의 후보를 임시 파일로 저장
    shard_files = []
    for i, shard in enumerate(shards):
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(shard, tf)
        tf.close()
        shard_files.append(tf.name)

    # 서브프로세스 실행
    procs = []
    for i in range(n_workers):
        cmd = [sys.executable, "-u", __file__,
               "--_worker", worker_type,
               "--_worker-id", str(i),
               "--_shard-file", shard_files[i]]
        # 필요한 인자 전달
        for key, val in args_dict.items():
            cmd.extend([f"--{key}", str(val)])
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True)
        procs.append(p)

    # 실시간 출력 수집
    def _drain(proc):
        for line in proc.stdout:
            print(line.rstrip(), flush=True)

    threads = []
    for p in procs:
        t = threading.Thread(target=_drain, args=(p,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    for p in procs:
        p.wait()

    # 임시 파일 정리
    for f in shard_files:
        os.unlink(f)


def run_helmet_off(args, base, model_path, cap_dir):
    print("\n" + "=" * 60, flush=True)
    print("  MODE 1: helmet_off 추출 (L2+full, 주간 07-17시)", flush=True)
    print("=" * 60, flush=True)

    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
    CAMERAS = ["cam1", "cam2"]

    OUT_DIR = os.path.join(base, "home/lay/hoban/datasets/helmet_off_v17")
    RESULT_DIR = os.path.join(OUT_DIR, "results")
    IMG_DIR = os.path.join(RESULT_DIR, "images")
    LBL_DIR = os.path.join(RESULT_DIR, "labels")
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    used = load_exclusion_set(base)
    already_found = set(os.listdir(IMG_DIR))
    print(f"3k 제외: {len(used)}장, 이전 결과: {len(already_found)}장", flush=True)

    candidates = collect_candidates(
        cap_dir, CAMERAS, used, already_found, args.interval,
        daytime_only=True, start_hour=args.start_hour, end_hour=args.end_hour)
    print(f"스캔 대상: {len(candidates)}장", flush=True)

    if not candidates:
        print("스캔할 이미지 없음", flush=True)
        return 0

    N = args.workers
    BATCH = max(1, args.batch_size)
    print(f"워커: {N}개, 배치: {BATCH}", flush=True)

    t_start = time.time()

    if N <= 1:
        # 단일 프로세스
        new_found = _helmet_off_worker(
            0, candidates, model_path, cap_dir,
            IMG_DIR, LBL_DIR, args.device, BATCH, args.max_gpu_batch)
    else:
        # 멀티프로세스
        _run_workers_subprocess("helmet_off", N, candidates, {
            "model-path": model_path,
            "cap-dir": cap_dir,
            "img-dir": IMG_DIR,
            "lbl-dir": LBL_DIR,
            "device": args.device,
            "batch-size": BATCH,
            "max-gpu-batch": args.max_gpu_batch,
        })
        new_found = len(os.listdir(IMG_DIR)) - len(already_found)

    elapsed = time.time() - t_start
    total = len(already_found) + new_found
    print(f"\nhelmet_off 수집 완료: {total}장 (신규 {new_found}, {elapsed:.0f}s)", flush=True)

    all_images = list(os.listdir(IMG_DIR))
    filtered = time_gap_filter(RESULT_DIR, all_images, args.min_gap, args.target_off)
    print(f"3분 간격 필터 후: {len(filtered)}장", flush=True)

    package_cvat(OUT_DIR, filtered, CLASS_NAMES)
    print(f"출력: {OUT_DIR}", flush=True)
    return new_found


def run_hard_neg(args, base, model_path, cap_dir):
    print("\n" + "=" * 60, flush=True)
    print("  MODE 2: hard negative 마이닝 (COCO+L2+full, 주간 07-17시)", flush=True)
    print("=" * 60, flush=True)

    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
    CAMERAS = ["cam1", "cam2"]

    OUT_DIR = os.path.join(base, "home/lay/hoban/datasets/hard_neg_v17")
    RESULT_DIR = os.path.join(OUT_DIR, "results")
    IMG_DIR = os.path.join(RESULT_DIR, "images")
    LBL_DIR = os.path.join(RESULT_DIR, "labels")
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    used = load_exclusion_set(base)
    already_found = set(os.listdir(IMG_DIR))
    print(f"3k 제외: {len(used)}장, 이전 결과: {len(already_found)}장", flush=True)

    candidates = collect_candidates(
        cap_dir, CAMERAS, used, already_found, args.interval * 2,
        daytime_only=True, start_hour=args.start_hour, end_hour=args.end_hour)
    print(f"스캔 대상: {len(candidates)}장", flush=True)

    if not candidates:
        print("스캔할 이미지 없음", flush=True)
        return 0

    coco_model_path = os.path.join(base, "home/lay/hoban/yolo26m.pt")
    N = min(args.workers, 2)  # hard_neg은 모델 2개 → 최대 2워커
    BATCH = max(1, args.batch_size)
    print(f"워커: {N}개, 배치: {BATCH}", flush=True)

    t_start = time.time()

    if N <= 1:
        new_found = _hard_neg_worker(
            0, candidates, model_path, coco_model_path, cap_dir,
            IMG_DIR, LBL_DIR, args.device, BATCH, args.max_gpu_batch)
    else:
        _run_workers_subprocess("hard_neg", N, candidates, {
            "model-path": model_path,
            "coco-path": coco_model_path,
            "cap-dir": cap_dir,
            "img-dir": IMG_DIR,
            "lbl-dir": LBL_DIR,
            "device": args.device,
            "batch-size": BATCH,
            "max-gpu-batch": args.max_gpu_batch,
        })
        new_found = len(os.listdir(IMG_DIR)) - len(already_found)

    elapsed = time.time() - t_start
    total = len(already_found) + new_found
    print(f"\nhard negative 수집 완료: {total}장 (신규 {new_found}, {elapsed:.0f}s)", flush=True)

    all_images = list(os.listdir(IMG_DIR))
    filtered = time_gap_filter(RESULT_DIR, all_images, args.min_gap, args.target_neg)
    print(f"3분 간격 필터 후: {len(filtered)}장", flush=True)

    package_cvat(OUT_DIR, filtered, CLASS_NAMES)
    print(f"출력: {OUT_DIR}", flush=True)
    return new_found


# ============================================================================
#  Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="v17 데이터 수집 (v5 - 멀티프로세스 GPU 병렬)")
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
    parser.add_argument("--workers", type=int, default=0,
                        help="병렬 워커 수 (0=자동: CUDA면 4, CPU면 1)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="워커당 이미지 배치 크기")
    parser.add_argument("--max-gpu-batch", type=int, default=16,
                        help="GPU 1회 forward 최대 타일 수")
    parser.add_argument("--io-workers", type=int, default=4)
    # 내부 서브프로세스용
    parser.add_argument("--_worker", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker-id", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_shard-file", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--coco-path", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--cap-dir", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--img-dir", type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--lbl-dir", type=str, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ── 서브프로세스 워커 모드 ──
    if args._worker:
        import json
        with open(args._shard_file) as f:
            candidates = json.load(f)
        # json은 list of list로 로드됨 → tuple로 변환
        candidates = [(c[0], c[1]) for c in candidates]

        if args._worker == "helmet_off":
            _helmet_off_worker(
                args._worker_id, candidates,
                args.model_path, args.cap_dir,
                args.img_dir, args.lbl_dir,
                args.device, args.batch_size, args.max_gpu_batch)
        elif args._worker == "hard_neg":
            _hard_neg_worker(
                args._worker_id, candidates,
                args.model_path, args.coco_path, args.cap_dir,
                args.img_dir, args.lbl_dir,
                args.device, args.batch_size, args.max_gpu_batch)
        return

    # ── 코디네이터 모드 ──
    BASE = "/" if args.server else "Z:/"
    MODEL_PATH = os.path.join(BASE, "home/lay/hoban/hoban_go3k_v17/weights/best.pt")
    CAP_DIR = os.path.join(BASE, "home/lay/video_indoor/static/captures")

    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.workers == 0:
        args.workers = 4 if "cuda" in args.device else 1

    print(f"Device: {args.device}", flush=True)
    print(f"모델: {MODEL_PATH}", flush=True)
    print(f"캡처: {CAP_DIR}", flush=True)
    print(f"파이프라인: L2+full (실서비스 동일)", flush=True)
    print(f"병렬: {args.workers}워커 × batch={args.batch_size}", flush=True)
    print(f"주간 필터: {args.start_hour:02d}:00 ~ {args.end_hour:02d}:00", flush=True)

    if not os.path.exists(MODEL_PATH):
        print(f"모델 없음: {MODEL_PATH}", flush=True)
        sys.exit(1)
    if not os.path.isdir(CAP_DIR):
        print(f"캡처 디렉터리 없음: {CAP_DIR}", flush=True)
        sys.exit(1)

    if args.clear:
        for mode_dir in ["helmet_off_v17", "hard_neg_v17"]:
            result_dir = os.path.join(BASE, f"home/lay/hoban/datasets/{mode_dir}/results")
            if os.path.isdir(result_dir):
                shutil.rmtree(result_dir)
                print(f"삭제: {result_dir}", flush=True)

    if args.mode in ("all", "helmet_off"):
        run_helmet_off(args, BASE, MODEL_PATH, CAP_DIR)

    if args.mode in ("all", "hard_neg"):
        run_hard_neg(args, BASE, MODEL_PATH, CAP_DIR)

    print("\n" + "=" * 60, flush=True)
    print("  완료!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
