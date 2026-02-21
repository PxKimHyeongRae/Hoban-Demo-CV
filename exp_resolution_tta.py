#!/usr/bin/env python3
"""추론 해상도 + TTA 실험

학습 없이 추론 시점에서 F1을 올릴 수 있는 2가지 방법 테스트:
  1. 추론 해상도 증가 (1280 → 1536 → 1920)
  2. TTA (좌우 반전 + 병합)
  3. 1+2 결합

v19 모델 사용, 729장 combined eval set.
"""
import os, sys, time, logging, argparse
import numpy as np
from collections import defaultdict
from PIL import Image, ImageOps

sys.stdout.reconfigure(line_buffering=True)
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

MODEL_PATH = "/home/lay/hoban/hoban_go3k_v19/weights/best.pt"
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}


def load_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls, (cx-w/2)*img_w, (cy-h/2)*img_h,
                          (cx+w/2)*img_w, (cy+h/2)*img_h))
    return boxes


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def cross_class_nms(preds, iou_thresh=0.3):
    if len(preds) <= 1:
        return preds
    sorted_preds = sorted(preds, key=lambda x: -x[1])
    keep, suppressed = [], set()
    for i in range(len(sorted_preds)):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            if sorted_preds[i][0] != sorted_preds[j][0]:
                if compute_iou(sorted_preds[i][2:], sorted_preds[j][2:]) >= iou_thresh:
                    suppressed.add(j)
    return keep


def nms_merge(preds, iou_thresh=0.5):
    """동일 클래스 NMS + cross-class NMS로 TTA 결과 병합"""
    if len(preds) <= 1:
        return preds
    # 동일 클래스 NMS
    by_class = defaultdict(list)
    for p in preds:
        by_class[p[0]].append(p)

    merged = []
    for cls, dets in by_class.items():
        sorted_d = sorted(dets, key=lambda x: -x[1])
        keep, supp = [], set()
        for i in range(len(sorted_d)):
            if i in supp:
                continue
            keep.append(sorted_d[i])
            for j in range(i+1, len(sorted_d)):
                if j in supp:
                    continue
                if compute_iou(sorted_d[i][2:], sorted_d[j][2:]) >= iou_thresh:
                    supp.add(j)
        merged.extend(keep)

    # cross-class NMS
    return cross_class_nms(merged, 0.3)


def apply_pipeline(preds, full_raw, img_w, img_h):
    """v17 최적 후처리 파이프라인"""
    filtered = cross_class_nms(preds, 0.3)
    img_area = img_w * img_h
    filtered = [d for d in filtered if ((d[4]-d[2])*(d[5]-d[3])) / img_area >= 5e-05]
    gates = [(x1, y1, x2, y2) for conf, x1, y1, x2, y2 in full_raw if conf >= 0.20]
    if gates:
        gated = []
        for d in filtered:
            cx, cy = (d[2]+d[4])/2, (d[3]+d[5])/2
            for gx1, gy1, gx2, gy2 in gates:
                gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
                if abs(cx-gcx) <= 30 and abs(cy-gcy) <= 30:
                    gated.append(d)
                    break
        filtered = gated
    return filtered


def evaluate(all_gt, all_preds, fnames, per_class_conf):
    tp = fp = fn = 0
    ctp, cfp, cfn = defaultdict(int), defaultdict(int), defaultdict(int)
    for fname in fnames:
        gts = all_gt.get(fname, [])
        preds = [d for d in all_preds.get(fname, [])
                 if d[1] >= per_class_conf.get(d[0], 0.5)]
        matched = set()
        for pred in sorted(preds, key=lambda x: -x[1]):
            bi, bv = -1, 0
            for gi, gt in enumerate(gts):
                if gi in matched or gt[0] != pred[0]:
                    continue
                iou = compute_iou(pred[2:], gt[1:])
                if iou > bv:
                    bv, bi = iou, gi
            if bv >= 0.5 and bi >= 0:
                tp += 1; ctp[gts[bi][0]] += 1; matched.add(bi)
            else:
                fp += 1; cfp[pred[0]] += 1
        for gi in range(len(gts)):
            if gi not in matched:
                fn += 1; cfn[gts[gi][0]] += 1
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn


def run_sahi(images, det_model, slice_h=720, slice_w=1280, overlap=0.15):
    from sahi.predict import get_sliced_prediction
    all_preds = {}
    for i, (fname, path) in enumerate(images):
        if i % 50 == 0:
            print(f"    SAHI: {i}/{len(images)}...", flush=True)
        r = get_sliced_prediction(
            path, det_model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", verbose=0)
        all_preds[fname] = [(p.category.id, p.score.value,
                             p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                            for p in r.object_prediction_list]
    return all_preds


def run_sahi_tta(images, det_model, slice_h=720, slice_w=1280, overlap=0.15):
    """TTA: 원본 + 좌우반전, 병합"""
    from sahi.predict import get_sliced_prediction
    all_preds = {}
    for i, (fname, path) in enumerate(images):
        if i % 50 == 0:
            print(f"    TTA: {i}/{len(images)}...", flush=True)
        pil_img = Image.open(path)
        img_w, img_h = pil_img.size

        # 원본
        r_orig = get_sliced_prediction(
            pil_img, det_model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", verbose=0)
        preds_orig = [(p.category.id, p.score.value,
                       p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                      for p in r_orig.object_prediction_list]

        # 좌우 반전
        flipped = ImageOps.mirror(pil_img)
        r_flip = get_sliced_prediction(
            flipped, det_model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", verbose=0)
        # 좌표 반전 복원
        preds_flip = [(p.category.id, p.score.value,
                       img_w - p.bbox.maxx, p.bbox.miny,
                       img_w - p.bbox.minx, p.bbox.maxy)
                      for p in r_flip.object_prediction_list]

        # 병합 (NMS)
        all_preds[fname] = nms_merge(preds_orig + preds_flip, iou_thresh=0.5)
    return all_preds


def run_full_inference(images, model_path):
    from ultralytics import YOLO
    print("  Full-image 추론 (Gate용)...")
    model = YOLO(model_path)
    full_preds = {}
    for i, (fname, path) in enumerate(images):
        if i % 100 == 0:
            print(f"    Full: {i}/{len(images)}...", flush=True)
        results = model.predict(path, conf=0.01, imgsz=1280, device="0", verbose=False)
        boxes = results[0].boxes
        full_preds[fname] = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                             for j in range(len(boxes))]
    return full_preds


def eval_with_sweep(all_gt, all_preds, full_preds, images, img_sizes, label):
    """후처리 파이프라인 + per-class conf sweep"""
    fnames = [f for f, _ in images]

    # 파이프라인 적용
    processed = {}
    for fname in fnames:
        img_w, img_h = img_sizes[fname]
        processed[fname] = apply_pipeline(
            all_preds.get(fname, []), full_preds.get(fname, []), img_w, img_h)

    best_f1, best_conf, best_result = 0, {}, None
    for c0 in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for c1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
            pc = {0: c0, 1: c1}
            tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, processed, fnames, pc)
            if f1 > best_f1:
                best_f1, best_conf = f1, pc
                best_result = (tp, fp, fn, p, r, f1, ctp, cfp, cfn)

    tp, fp, fn, p, r, f1, ctp, cfp, cfn = best_result
    print(f"\n  [{label}] F1={f1:.4f} P={p:.3f} R={r:.3f} "
          f"(TP={tp} FP={fp} FN={fn}) @c0={best_conf[0]:.2f} c1={best_conf[1]:.2f}")
    for cls_id, name in CLASS_NAMES.items():
        ct, cf, cm = ctp[cls_id], cfp[cls_id], cfn[cls_id]
        cp = ct/(ct+cf) if ct+cf else 0
        cr = ct/(ct+cm) if ct+cm else 0
        cf1 = 2*cp*cr/(cp+cr) if cp+cr else 0
        print(f"    {name}: P={cp:.3f} R={cr:.3f} F1={cf1:.3f} (TP={ct} FP={cf} FN={cm})")
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    t_start = time.time()

    # 이미지 로드
    images = []
    all_gt, img_sizes = {}, {}
    for f in sorted(os.listdir(VAL_IMG)):
        if f.endswith(".jpg"):
            path = os.path.join(VAL_IMG, f)
            img = Image.open(path)
            img_sizes[f] = img.size
            all_gt[f] = load_gt(os.path.join(VAL_LBL, f.replace(".jpg", ".txt")), *img.size)
            images.append((f, path))

    if os.path.isdir(EXTRA_IMG):
        for f in sorted(os.listdir(EXTRA_IMG)):
            if f.endswith(".jpg") and f not in all_gt:
                path = os.path.join(EXTRA_IMG, f)
                img = Image.open(path)
                img_sizes[f] = img.size
                all_gt[f] = load_gt(os.path.join(EXTRA_LBL, f.replace(".jpg", ".txt")), *img.size)
                images.append((f, path))

    print(f"모델: {args.model}")
    print(f"평가: {len(images)}장")

    # Full-image (Gate용, 1회만)
    full_preds = run_full_inference(images, args.model)

    from sahi import AutoDetectionModel

    results = {}

    # ========================================
    # 실험 1: 기본 (imgsz=1280, baseline)
    # ========================================
    print(f"\n{'='*70}")
    print(f"  실험 1: 기본 (imgsz=1280)")
    print(f"{'='*70}")
    det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=args.model,
        confidence_threshold=0.05, device="0", image_size=1280)
    preds = run_sahi(images, det_model)
    results["1280"] = eval_with_sweep(all_gt, preds, full_preds, images, img_sizes, "imgsz=1280")

    # ========================================
    # 실험 2: 고해상도 (imgsz=1536)
    # ========================================
    print(f"\n{'='*70}")
    print(f"  실험 2: 고해상도 (imgsz=1536)")
    print(f"{'='*70}")
    det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=args.model,
        confidence_threshold=0.05, device="0", image_size=1536)
    preds = run_sahi(images, det_model)
    results["1536"] = eval_with_sweep(all_gt, preds, full_preds, images, img_sizes, "imgsz=1536")

    # ========================================
    # 실험 3: 고해상도 (imgsz=1920)
    # ========================================
    print(f"\n{'='*70}")
    print(f"  실험 3: 고해상도 (imgsz=1920)")
    print(f"{'='*70}")
    det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=args.model,
        confidence_threshold=0.05, device="0", image_size=1920)
    preds = run_sahi(images, det_model)
    results["1920"] = eval_with_sweep(all_gt, preds, full_preds, images, img_sizes, "imgsz=1920")

    # ========================================
    # 실험 4: TTA (imgsz=1280)
    # ========================================
    print(f"\n{'='*70}")
    print(f"  실험 4: TTA + imgsz=1280")
    print(f"{'='*70}")
    det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=args.model,
        confidence_threshold=0.05, device="0", image_size=1280)
    preds = run_sahi_tta(images, det_model)
    results["TTA+1280"] = eval_with_sweep(all_gt, preds, full_preds, images, img_sizes, "TTA+1280")

    # ========================================
    # 실험 5: TTA + 고해상도 (best resolution)
    # ========================================
    # 1536과 1920 중 더 좋은 해상도로 TTA
    best_res = "1536" if results.get("1536", 0) >= results.get("1920", 0) else "1920"
    best_imgsz = int(best_res)
    print(f"\n{'='*70}")
    print(f"  실험 5: TTA + imgsz={best_imgsz}")
    print(f"{'='*70}")
    det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=args.model,
        confidence_threshold=0.05, device="0", image_size=best_imgsz)
    preds = run_sahi_tta(images, det_model)
    results[f"TTA+{best_res}"] = eval_with_sweep(
        all_gt, preds, full_preds, images, img_sizes, f"TTA+{best_res}")

    # ========================================
    # 최종 요약
    # ========================================
    print(f"\n{'='*70}")
    print(f"  최종 요약")
    print(f"{'='*70}")
    for label, f1 in sorted(results.items(), key=lambda x: -x[1]):
        diff = f1 - results["1280"]
        sign = "+" if diff >= 0 else ""
        print(f"  {label:>12s}: F1={f1:.4f} ({sign}{diff:.4f})")

    print(f"\n  v19 기준: F1=0.928")
    print(f"  소요: {(time.time()-t_start)/60:.1f}분")


if __name__ == "__main__":
    main()
