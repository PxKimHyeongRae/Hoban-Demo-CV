#!/usr/bin/env python3
"""min_area 임계값 시뮬레이션 + SAHI 타일 크기 실험

기존 v19 모델로 다양한 후처리 파라미터 테스트.
"""
import os, sys, time, logging, json
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

MODEL_PATH = "/home/lay/hoban/hoban_go3k_v19/weights/best.pt"
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def cross_class_nms(dets, iou_thr=0.3):
    if len(dets) <= 1:
        return dets
    sorted_d = sorted(dets, key=lambda x: -x[1])
    keep, supp = [], set()
    for i in range(len(sorted_d)):
        if i in supp:
            continue
        keep.append(sorted_d[i])
        for j in range(i + 1, len(sorted_d)):
            if j in supp:
                continue
            if sorted_d[i][0] != sorted_d[j][0]:
                if compute_iou(sorted_d[i][2:], sorted_d[j][2:]) >= iou_thr:
                    supp.add(j)
    return keep


def load_gt(lbl_path, img_w, img_h):
    gts = []
    if not os.path.exists(lbl_path):
        return gts
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            gts.append((cls, x1, y1, x2, y2))
    return gts


def evaluate_with_params(all_sahi_dets, all_full_dets, all_gts, all_sizes,
                          min_area, gate_conf, gate_radius, c0, c1):
    tp, fp, fn = 0, 0, 0
    for i in range(len(all_gts)):
        img_w, img_h = all_sizes[i]
        img_area = img_w * img_h
        gts = all_gts[i]
        sahi_dets = all_sahi_dets[i]
        full_dets = all_full_dets[i]

        # pipeline
        dets = cross_class_nms(sahi_dets, 0.3)
        dets = [d for d in dets if (d[4] - d[2]) * (d[5] - d[3]) / img_area >= min_area]

        # gate
        gated = []
        for d in dets:
            cx = (d[2] + d[4]) / 2
            cy = (d[3] + d[5]) / 2
            for fd in full_dets:
                fcx = (fd[2] + fd[4]) / 2
                fcy = (fd[3] + fd[5]) / 2
                if abs(cx - fcx) < gate_radius and abs(cy - fcy) < gate_radius and fd[1] >= gate_conf:
                    gated.append(d)
                    break
        dets = gated

        # per-class conf
        dets = [d for d in dets if d[1] >= (c0 if d[0] == 0 else c1)]

        # matching
        matched_gt = set()
        for pred in sorted(dets, key=lambda x: -x[1]):
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gts):
                if gt[0] != pred[0] or gi in matched_gt:
                    continue
                iou = compute_iou(pred[2:], gt[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched_gt.add(best_gi)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)

    P = tp / (tp + fp) if (tp + fp) > 0 else 0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return P, R, F1, tp, fp, fn


def run_sahi_inference(images, slice_h, slice_w, overlap=0.15):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=MODEL_PATH,
        confidence_threshold=0.05, device="cuda:0")

    all_dets = []
    for idx, (img_path, _, fname) in enumerate(images):
        if idx % 100 == 0:
            print(f"  SAHI ({slice_w}x{slice_h}): {idx}/{len(images)}", flush=True)
        pil_img = Image.open(img_path)
        result = get_sliced_prediction(
            pil_img, det_model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5)
        dets = [(p.category.id, p.score.value, p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                for p in result.object_prediction_list]
        all_dets.append(dets)
    return all_dets


def run_full_inference(images):
    from ultralytics import YOLO
    import torch
    from extract_data_v17 import _letterbox, _parse_preds

    model = YOLO(MODEL_PATH)
    model.fuse()
    model.model.to("cuda:0")
    model.model.half()

    all_dets = []
    for idx, (img_path, _, fname) in enumerate(images):
        if idx % 100 == 0:
            print(f"  Full: {idx}/{len(images)}", flush=True)
        img = cv2.imread(img_path)
        if img is None:
            all_dets.append([])
            continue
        img_h, img_w = img.shape[:2]
        lb, scale, dw, dh = _letterbox(img, 1280)
        t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(t[None]).half().to("cuda:0")
        with torch.no_grad():
            preds = model.model(tensor)[0]
        dets = _parse_preds(preds[0], 0.01, img_h, img_w, scale, dw, dh)
        all_dets.append(dets)
    return all_dets


def main():
    images = []
    for f in sorted(os.listdir(VAL_IMG)):
        if f.endswith(".jpg"):
            images.append((os.path.join(VAL_IMG, f),
                          os.path.join(VAL_LBL, f.replace(".jpg", ".txt")), f))
    if os.path.isdir(EXTRA_IMG):
        for f in sorted(os.listdir(EXTRA_IMG)):
            if f.endswith(".jpg"):
                images.append((os.path.join(EXTRA_IMG, f),
                              os.path.join(EXTRA_LBL, f.replace(".jpg", ".txt")), f))
    print(f"평가 이미지: {len(images)}장")

    # GT 로드
    all_gts, all_sizes = [], []
    for img_path, lbl_path, fname in images:
        img = cv2.imread(img_path)
        if img is None:
            all_gts.append([])
            all_sizes.append((0, 0))
            continue
        h, w = img.shape[:2]
        all_sizes.append((w, h))
        all_gts.append(load_gt(lbl_path, w, h))

    # ========================================
    # 실험 1: min_area sweep (기본 SAHI 1280x720)
    # ========================================
    print("\n=== 실험 1: SAHI 1280x720 추론 ===")
    sahi_1280 = run_sahi_inference(images, 720, 1280)
    full_dets = run_full_inference(images)

    print("\n" + "=" * 70)
    print("  실험 1: min_area 임계값 sweep")
    print("=" * 70)
    min_areas = [0, 1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
    for ma in min_areas:
        P, R, F1, tp, fp, fn = evaluate_with_params(
            sahi_1280, full_dets, all_gts, all_sizes,
            min_area=ma, gate_conf=0.20, gate_radius=30, c0=0.40, c1=0.15)
        print(f"  min_area={ma:.0e}: F1={F1:.4f} P={P:.3f} R={R:.3f} (TP={tp} FP={fp} FN={fn})")

    # per-class conf sweep (min_area 최적값 근처)
    print("\n" + "=" * 70)
    print("  실험 1b: per-class conf sweep (min_area=1e-4)")
    print("=" * 70)
    for c0 in [0.35, 0.40, 0.45, 0.50]:
        for c1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
            P, R, F1, tp, fp, fn = evaluate_with_params(
                sahi_1280, full_dets, all_gts, all_sizes,
                min_area=1e-4, gate_conf=0.20, gate_radius=30, c0=c0, c1=c1)
            if F1 > 0.925:
                print(f"  c0={c0:.2f} c1={c1:.2f}: F1={F1:.4f} P={P:.3f} R={R:.3f} (FP={fp} FN={fn})")

    # ========================================
    # 실험 2: SAHI 타일 크기
    # ========================================
    tile_configs = [
        (540, 960, "960x540"),
        (480, 640, "640x480"),
    ]
    for slice_h, slice_w, label in tile_configs:
        print(f"\n=== 실험 2: SAHI {label} 추론 ===")
        sahi_dets = run_sahi_inference(images, slice_h, slice_w)

        print(f"\n{'='*70}")
        print(f"  실험 2: SAHI {label} + min_area sweep")
        print(f"{'='*70}")
        for ma in [0, 5e-5, 1e-4, 2e-4]:
            for c0 in [0.35, 0.40, 0.45]:
                for c1 in [0.10, 0.15, 0.20, 0.30]:
                    P, R, F1, tp, fp, fn = evaluate_with_params(
                        sahi_dets, full_dets, all_gts, all_sizes,
                        min_area=ma, gate_conf=0.20, gate_radius=30, c0=c0, c1=c1)
                    if F1 > 0.925:
                        print(f"  ma={ma:.0e} c0={c0:.2f} c1={c1:.2f}: F1={F1:.4f} P={P:.3f} R={R:.3f} (FP={fp} FN={fn})")

    print("\n완료!")


if __name__ == "__main__":
    main()
