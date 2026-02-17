#!/usr/bin/env python3
"""v17 데이터 수집: helmet_off 추출 + hard negative 마이닝 (통합 스크립트)

Mode 1 (helmet_off): captures에서 helmet_off 탐지 이미지 수집
  - v17 SAHI 추론 → helmet_off 있는 이미지 + 라벨 저장
  - 3k train/val 제외, burst 스킵, 3분 간격 필터

Mode 2 (hard_neg): 사람 없는 배경에서 오탐된 이미지 수집
  - Full-image 추론: 탐지 없음 → SAHI 추론: 탐지 있음 = gate 미스매치
  - 빈 라벨(.txt)과 함께 저장 (학습 시 배경 FP 제거 효과)

사용법:
  python extract_data_v17.py                    # 둘 다 순차 실행
  python extract_data_v17.py --mode helmet_off  # helmet_off만
  python extract_data_v17.py --mode hard_neg    # hard negative만
  python extract_data_v17.py --server           # 서버에서 실행
"""
import os
import sys
import zipfile
import time
import logging
import argparse
import shutil
from collections import defaultdict

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


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


def load_exclusion_set(base, extra_dirs=None):
    """3k train/val + 기타 제외 목록 생성"""
    used = set()
    dirs = [
        os.path.join(base, "home/lay/hoban/datasets/3k_finetune/train/images"),
        os.path.join(base, "home/lay/hoban/datasets/3k_finetune/val/images"),
        os.path.join(base, "home/lay/hoban/datasets_go3k_v16/train/images"),
        os.path.join(base, "home/lay/hoban/datasets_go3k_v16/val/images"),
    ]
    if extra_dirs:
        dirs.extend(extra_dirs)
    for d in dirs:
        if os.path.isdir(d):
            used.update(os.listdir(d))
    return used


def collect_candidates(cap_dir, cameras, used, already_found, interval=5):
    """캡처 디렉터리에서 후보 이미지 수집"""
    candidates = []
    for cam in cameras:
        cam_dir = os.path.join(cap_dir, cam)
        if not os.path.isdir(cam_dir):
            print(f"  {cam}: 디렉터리 없음 ({cam_dir})")
            continue
        all_files = sorted(f for f in os.listdir(cam_dir)
                          if f.endswith(".jpg") and f not in used and f not in already_found)
        sampled = all_files[::interval]
        candidates.extend([(cam, f) for f in sampled])
        print(f"  {cam}: 전체 {len(all_files)} -> 샘플 {len(sampled)}")
    return candidates


def time_gap_filter(result_dir, images, min_gap=180, target=None):
    """시간 간격 기반 필터링"""
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

    # helmet_off가 많은 순으로 정렬 후 target 제한
    filtered.sort(key=lambda x: -x[3])
    if target and len(filtered) > target:
        filtered = filtered[:target]

    return filtered


def package_cvat(out_dir, filtered, class_names, prefix=""):
    """CVAT 업로드용 zip 패키징"""
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
#  Mode 1: helmet_off 추출
# ============================================================================
def run_helmet_off(args, base, model_path, cap_dir):
    """captures에서 v17 SAHI로 helmet_off 탐지 이미지 추출"""
    print("\n" + "=" * 60)
    print("  MODE 1: helmet_off 추출")
    print("=" * 60)

    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
    CAMERAS = ["cam1", "cam2"]

    OUT_DIR = os.path.join(base, "home/lay/hoban/datasets/helmet_off_v17")
    RESULT_DIR = os.path.join(OUT_DIR, "results")
    os.makedirs(os.path.join(RESULT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "labels"), exist_ok=True)

    # 제외 목록
    used = load_exclusion_set(base)
    already_found = set(os.listdir(os.path.join(RESULT_DIR, "images")))
    print(f"3k 제외: {len(used)}장, 이전 결과: {len(already_found)}장")

    # 후보 수집
    candidates = collect_candidates(cap_dir, CAMERAS, used, already_found, args.interval)
    print(f"스캔 대상: {len(candidates)}장")

    if not candidates:
        print("스캔할 이미지 없음")
        return

    # SAHI 모델 로드
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from PIL import Image

    print(f"v17 SAHI 모델 로드 (device={args.device})")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.15, device=args.device, image_size=1280)

    # Cross-class NMS (후처리)
    def cross_class_nms(preds, iou_thresh=0.3):
        if len(preds) <= 1:
            return preds
        sorted_p = sorted(preds, key=lambda x: -x[1])
        keep, suppressed = [], set()
        for i in range(len(sorted_p)):
            if i in suppressed:
                continue
            keep.append(sorted_p[i])
            for j in range(i + 1, len(sorted_p)):
                if j in suppressed:
                    continue
                if sorted_p[i][0] != sorted_p[j][0]:
                    b1 = sorted_p[i][2:]
                    b2 = sorted_p[j][2:]
                    inter_x1, inter_y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                    inter_x2, inter_y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
                    if iou >= iou_thresh:
                        suppressed.add(j)
        return keep

    # 추론 루프
    BURST_WINDOW, BURST_THRESHOLD, BURST_SKIP = 100, 10, 200
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

        try:
            result = get_sliced_prediction(
                img_path, sahi_model,
                slice_height=720, slice_width=1280,
                overlap_height_ratio=0.15, overlap_width_ratio=0.15,
                perform_standard_pred=True,
                postprocess_type="NMS", postprocess_match_threshold=0.4,
                postprocess_match_metric="IOS", verbose=0)
        except Exception:
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size
        img_area = img_w * img_h

        # raw predictions
        raw_preds = [(p.category.id, p.score.value,
                      p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                     for p in result.object_prediction_list]

        # 후처리: cross-class NMS + min area + per-class conf
        preds = cross_class_nms(raw_preds, 0.3)
        preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in preds
                 if ((x2 - x1) * (y2 - y1)) / img_area >= 5e-05]
        preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in preds
                 if s >= (0.40 if c == 0 else 0.15)]

        has_off = any(c == 1 for c, *_ in preds)
        if not has_off:
            continue

        new_found += 1
        window_hits += 1

        # YOLO format 라벨 생성
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

    # 시간 간격 필터
    all_images = list(os.listdir(os.path.join(RESULT_DIR, "images")))
    filtered = time_gap_filter(RESULT_DIR, all_images, args.min_gap, args.target_off)
    print(f"3분 간격 필터 후: {len(filtered)}장")

    # CVAT 패키징
    package_cvat(OUT_DIR, filtered, CLASS_NAMES)
    print(f"출력: {OUT_DIR}")

    return new_found


# ============================================================================
#  Mode 2: hard negative 마이닝
# ============================================================================
def run_hard_neg(args, base, model_path, cap_dir):
    """사람 없는 배경에서 v17이 오탐하는 이미지 수집 (빈 라벨)

    Gate 미스매치 방식:
      1. Full-image 추론 (conf=0.20) → 탐지 없음
      2. SAHI 추론 → 탐지 있음
      → 이런 이미지는 SAHI가 배경을 사람으로 오인하는 케이스
      → 빈 라벨로 학습하면 배경 FP 제거 효과
    """
    print("\n" + "=" * 60)
    print("  MODE 2: hard negative 마이닝")
    print("=" * 60)

    CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
    CAMERAS = ["cam1", "cam2"]

    OUT_DIR = os.path.join(base, "home/lay/hoban/datasets/hard_neg_v17")
    RESULT_DIR = os.path.join(OUT_DIR, "results")
    os.makedirs(os.path.join(RESULT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "labels"), exist_ok=True)

    # 제외 목록
    used = load_exclusion_set(base)
    already_found = set(os.listdir(os.path.join(RESULT_DIR, "images")))
    print(f"3k 제외: {len(used)}장, 이전 결과: {len(already_found)}장")

    # 후보 수집 (더 넓게 샘플링 - hard neg은 대부분 탐지 없으므로)
    candidates = collect_candidates(cap_dir, CAMERAS, used, already_found, args.interval * 2)
    print(f"스캔 대상: {len(candidates)}장")

    if not candidates:
        print("스캔할 이미지 없음")
        return

    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    # Full-image 모델 (Gate용)
    print(f"v17 Full-image 모델 로드 (device={args.device})")
    yolo_model = YOLO(model_path)

    # SAHI 모델
    print(f"v17 SAHI 모델 로드 (device={args.device})")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.10, device=args.device, image_size=1280)

    GATE_CONF = 0.20      # full-image에서 이 conf 이상이면 사람 있음
    SAHI_MIN_CONF = 0.30  # SAHI에서 이 conf 이상 탐지가 있으면 오탐 후보

    new_found, processed, skipped_has_person = 0, 0, 0
    t_start = time.time()

    for idx, (cam, fname) in enumerate(candidates):
        img_path = os.path.join(cap_dir, cam, fname)
        processed += 1

        # Step 1: Full-image 추론
        try:
            results = yolo_model.predict(img_path, conf=0.01, imgsz=1280,
                                         device=args.device, verbose=False)
        except Exception:
            continue

        boxes = results[0].boxes
        # Gate 확인: full-image에서 conf >= GATE_CONF 탐지가 있으면 사람 있는 이미지 → 스킵
        has_person = any(float(boxes.conf[j]) >= GATE_CONF for j in range(len(boxes)))
        if has_person:
            skipped_has_person += 1
            continue

        # Step 2: Full-image에서 탐지 없음 → SAHI 추론
        try:
            result = get_sliced_prediction(
                img_path, sahi_model,
                slice_height=720, slice_width=1280,
                overlap_height_ratio=0.15, overlap_width_ratio=0.15,
                perform_standard_pred=False,  # full-image는 이미 했으므로 불필요
                postprocess_type="NMS", postprocess_match_threshold=0.4,
                postprocess_match_metric="IOS", verbose=0)
        except Exception:
            continue

        # SAHI에서 conf >= SAHI_MIN_CONF 탐지가 있으면 = gate 미스매치 = hard negative
        sahi_dets = [p for p in result.object_prediction_list
                     if p.score.value >= SAHI_MIN_CONF]

        if not sahi_dets:
            continue

        # Hard negative 발견!
        new_found += 1
        shutil.copy2(img_path, os.path.join(RESULT_DIR, "images", fname))
        # 빈 라벨 파일 생성
        with open(os.path.join(RESULT_DIR, "labels", fname.replace(".jpg", ".txt")), "w") as f:
            f.write("")  # 빈 라벨 = 이 이미지에는 객체 없음

        if processed % 100 == 0 or new_found % 10 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            total = len(already_found) + new_found
            print(f"  [{processed}/{len(candidates)}] "
                  f"hard_neg: {new_found} | 사람있음: {skipped_has_person} "
                  f"({rate:.1f} img/s)")
            if total >= args.target_neg:
                print(f"  목표 달성! ({total}장)")
                break

        if len(already_found) + new_found >= args.target_neg:
            break

    elapsed = time.time() - t_start
    total = len(already_found) + new_found
    print(f"\nhard negative 수집 완료: {total}장 (신규 {new_found}, {elapsed:.0f}s)")
    print(f"  사람 있는 이미지 스킵: {skipped_has_person}장")

    # 시간 간격 필터 (hard neg도 동일 시간대 중복 방지)
    all_images = list(os.listdir(os.path.join(RESULT_DIR, "images")))
    filtered = time_gap_filter(RESULT_DIR, all_images, args.min_gap, args.target_neg)
    print(f"3분 간격 필터 후: {len(filtered)}장")

    # CVAT 패키징
    package_cvat(OUT_DIR, filtered, CLASS_NAMES)
    print(f"출력: {OUT_DIR}")

    return new_found


# ============================================================================
#  Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="v17 데이터 수집: helmet_off + hard negative")
    parser.add_argument("--server", action="store_true",
                        help="서버에서 실행 (경로 / 기준)")
    parser.add_argument("--mode", choices=["all", "helmet_off", "hard_neg"],
                        default="all", help="실행 모드 (default: all)")
    parser.add_argument("--interval", type=int, default=5,
                        help="캡처 샘플링 간격 (default: 5)")
    parser.add_argument("--target-off", type=int, default=400,
                        help="helmet_off 목표 수 (default: 400)")
    parser.add_argument("--target-neg", type=int, default=300,
                        help="hard negative 목표 수 (default: 300)")
    parser.add_argument("--min-gap", type=int, default=180,
                        help="최소 시간 간격-초 (default: 180)")
    parser.add_argument("--device", type=str, default=None,
                        help="GPU device (default: auto)")
    args = parser.parse_args()

    # 경로 설정
    BASE = "/" if args.server else "Z:/"
    MODEL_PATH = os.path.join(BASE, "home/lay/hoban/hoban_go3k_v17/weights/best.pt")
    CAP_DIR = os.path.join(BASE, "home/lay/video_indoor/static/captures")

    # device 자동 선택
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

    # 실행
    if args.mode in ("all", "helmet_off"):
        run_helmet_off(args, BASE, MODEL_PATH, CAP_DIR)

    if args.mode in ("all", "hard_neg"):
        run_hard_neg(args, BASE, MODEL_PATH, CAP_DIR)

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
