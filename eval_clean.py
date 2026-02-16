#!/usr/bin/env python3
"""
Clean Evaluation: data leakage 제거 후 진짜 일반화 성능 측정
- eval-only 125장 (train과 겹치지 않는 이미지)
- leaked 479장 (train과 동일한 이미지)
- 전체 604장 (기존 baseline)
세 그룹을 모두 평가하여 leakage 영향 정량화
"""
import os
import time
import numpy as np
from collections import defaultdict
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion

# === Config ===
CLASS_NAMES = {0: "person_with_helmet", 1: "person_without_helmet"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"
TRAIN_DIR = "/home/lay/hoban/datasets_go2k_v2/train/images"

MODELS = {
    "v2": "/home/lay/hoban/hoban_go2k_v2/weights/best.pt",
    "v3": "/home/lay/hoban/hoban_go2k_v3/weights/best.pt",
    "v5": "/home/lay/hoban/hoban_go2k_v5/weights/best.pt",
}


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
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate(all_gt, all_preds, image_set, conf_thresh=None, per_class_conf=None, iou_thresh=0.5):
    """Evaluate on a specific image set."""
    sum_tp = sum_fp = sum_fn = 0
    cls_tp = defaultdict(int)
    cls_fp = defaultdict(int)
    cls_fn = defaultdict(int)

    for fname in image_set:
        gt_boxes = all_gt.get(fname, [])
        raw_preds = all_preds.get(fname, [])

        if per_class_conf:
            pred_boxes = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in raw_preds
                          if s >= per_class_conf.get(c, 0.5)]
        elif conf_thresh is not None:
            pred_boxes = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in raw_preds
                          if s >= conf_thresh]
        else:
            pred_boxes = raw_preds

        matched_gt = set()
        pred_sorted = sorted(enumerate(pred_boxes), key=lambda x: -x[1][1])
        for pi, (p_cls, p_conf, p_x1, p_y1, p_x2, p_y2) in pred_sorted:
            best_iou = 0
            best_gi = -1
            for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if gi in matched_gt or g_cls != p_cls:
                    continue
                iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thresh and best_gi >= 0:
                sum_tp += 1
                cls_tp[gt_boxes[best_gi][0]] += 1
                matched_gt.add(best_gi)
            else:
                sum_fp += 1
                cls_fp[p_cls] += 1

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                sum_fn += 1
                cls_fn[gt_boxes[gi][0]] += 1

    prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return sum_tp, sum_fp, sum_fn, prec, rec, f1, cls_tp, cls_fp, cls_fn


def run_sahi_inference(model_path, images, img_sizes, device="0",
                       tile_w=1280, tile_h=720, overlap=0.15, image_size=None):
    kwargs = {}
    if image_size:
        kwargs["image_size"] = image_size

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.05,
        device=device,
        **kwargs,
    )

    all_preds = {}
    for i, fname in enumerate(images):
        if i % 100 == 0:
            print(f"  {i}/{len(images)}...", end="\r")
        img_path = os.path.join(IMG_DIR, fname)
        result = get_sliced_prediction(
            img_path, model,
            slice_height=tile_h, slice_width=tile_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS",
            postprocess_class_agnostic=False,
        )
        preds = []
        for p in result.object_prediction_list:
            bbox = p.bbox
            preds.append((p.category.id, p.score.value, bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))
        all_preds[fname] = preds

    print(f"  Done: {sum(len(v) for v in all_preds.values())} raw predictions")
    return all_preds


def wbf_ensemble(preds_list, img_sizes, iou_thr=0.4, weights=None):
    all_merged = {}
    for fname in preds_list[0]:
        img_w, img_h = img_sizes[fname]
        boxes_list, scores_list, labels_list = [], [], []

        for preds in preds_list:
            raw = preds.get(fname, [])
            if not raw:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue
            boxes, scores, labels = [], [], []
            for cls, score, x1, y1, x2, y2 in raw:
                boxes.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
                scores.append(score)
                labels.append(cls)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        if all(len(b) == 0 for b in boxes_list):
            all_merged[fname] = []
            continue

        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=0.0001,
        )
        result = []
        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            result.append((int(label), float(score),
                           box[0]*img_w, box[1]*img_h, box[2]*img_w, box[3]*img_h))
        all_merged[fname] = result

    return all_merged


def print_eval(title, tp, fp, fn, prec, rec, f1, cls_tp, cls_fp, cls_fn):
    print(f"\n  {title}")
    print(f"  {'Overall':>20}: TP={tp:>4} FP={fp:>4} FN={fn:>4} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
    for c in [0, 1]:
        ctp = cls_tp.get(c, 0)
        cfp = cls_fp.get(c, 0)
        cfn = cls_fn.get(c, 0)
        cp = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0
        cr = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0
        print(f"  {CLASS_NAMES[c]:>20}: TP={ctp:>4} FP={cfp:>4} FN={cfn:>4} P={cp:.3f} R={cr:.3f} F1={cf:.3f}")


# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Clean Evaluation: Data Leakage 영향 정량화")
    print("=" * 70)

    # === Identify image groups ===
    train_originals = set()
    for f in os.listdir(TRAIN_DIR):
        if f.startswith("cam") and "_x" not in f and f.endswith(".jpg"):
            train_originals.add(f)

    all_images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
    leaked = sorted([f for f in all_images if f in train_originals])
    clean = sorted([f for f in all_images if f not in train_originals])

    print(f"\n전체: {len(all_images)}장")
    print(f"  Leaked (학습 포함): {len(leaked)}장")
    print(f"  Clean (미학습): {len(clean)}장")

    # === Load GT ===
    all_gt = {}
    img_sizes = {}
    for fname in all_images:
        img = Image.open(os.path.join(IMG_DIR, fname))
        img_w, img_h = img.size
        img_sizes[fname] = (img_w, img_h)
        all_gt[fname] = load_gt(os.path.join(LBL_DIR, fname.replace(".jpg", ".txt")), img_w, img_h)

    # GT stats per group
    for name, group in [("전체", all_images), ("Leaked", leaked), ("Clean", clean)]:
        gt_count = sum(len(all_gt.get(f, [])) for f in group)
        cls0 = sum(1 for f in group for b in all_gt.get(f, []) if b[0] == 0)
        cls1 = sum(1 for f in group for b in all_gt.get(f, []) if b[0] == 1)
        print(f"\n  {name} GT: {gt_count} bbox (cls0={cls0}, cls1={cls1})")

    # === SAHI Inference (cached for all images) ===
    print("\n" + "=" * 70)
    print("SAHI 추론 (3개 모델)")
    print("=" * 70)

    pred_cache = {}

    print("\n[v2] 1280x720 tiles, overlap=0.15")
    t0 = time.time()
    pred_cache["v2"] = run_sahi_inference(MODELS["v2"], all_images, img_sizes)
    print(f"  Time: {time.time()-t0:.0f}s")

    print("\n[v3] 1280x720 tiles, overlap=0.15, image_size=1280")
    t0 = time.time()
    pred_cache["v3"] = run_sahi_inference(MODELS["v3"], all_images, img_sizes, image_size=1280)
    print(f"  Time: {time.time()-t0:.0f}s")

    print("\n[v5] 1280x720 tiles, overlap=0.15")
    t0 = time.time()
    pred_cache["v5"] = run_sahi_inference(MODELS["v5"], all_images, img_sizes)
    print(f"  Time: {time.time()-t0:.0f}s")

    # === WBF Ensemble ===
    print("\nWBF 앙상블 생성...")
    wbf_preds = wbf_ensemble(
        [pred_cache["v2"], pred_cache["v3"], pred_cache["v5"]],
        img_sizes, iou_thr=0.4, weights=[1, 1, 0.5]
    )

    # === Evaluation ===
    print("\n" + "=" * 70)
    print("결과 비교: Leaked vs Clean vs 전체")
    print("=" * 70)

    configs = [
        ("v2 단독 conf=0.55", pred_cache["v2"], {"conf_thresh": 0.55}),
        ("v2 per-class c0=0.65 c1=0.35", pred_cache["v2"], {"per_class_conf": {0: 0.65, 1: 0.35}}),
        ("v2+v3+v5 WBF/0.4 c0=0.50 c1=0.30", wbf_preds, {"per_class_conf": {0: 0.50, 1: 0.30}}),
    ]

    for config_name, preds, kwargs in configs:
        print(f"\n{'='*70}")
        print(f"설정: {config_name}")
        print(f"{'='*70}")

        for group_name, group in [("전체 (604장)", all_images),
                                   ("Leaked (479장)", leaked),
                                   ("Clean (125장)", clean)]:
            tp, fp, fn, prec, rec, f1, cls_tp, cls_fp, cls_fn = evaluate(
                all_gt, preds, group, **kwargs
            )
            print_eval(group_name, tp, fp, fn, prec, rec, f1, cls_tp, cls_fp, cls_fn)

    # === IoU sensitivity ===
    print(f"\n{'='*70}")
    print("IoU 임계값 민감도 분석 (WBF best, Clean 125장)")
    print(f"{'='*70}")

    for iou in [0.3, 0.4, 0.5, 0.6]:
        tp, fp, fn, prec, rec, f1, cls_tp, cls_fp, cls_fn = evaluate(
            all_gt, wbf_preds, clean,
            per_class_conf={0: 0.50, 1: 0.30},
            iou_thresh=iou,
        )
        print(f"\n  IoU={iou}: TP={tp} FP={fp} FN={fn} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # === FN analysis by bbox size ===
    print(f"\n{'='*70}")
    print("FN 분석: bbox 크기별 (WBF best, Clean 125장)")
    print(f"{'='*70}")

    per_class_conf = {0: 0.50, 1: 0.30}
    size_bins = {"tiny(<15px)": 0, "small(15-30px)": 0, "medium(30-50px)": 0, "large(>50px)": 0}
    size_total = {"tiny(<15px)": 0, "small(15-30px)": 0, "medium(30-50px)": 0, "large(>50px)": 0}

    for fname in clean:
        gt_boxes = all_gt.get(fname, [])
        raw_preds = wbf_preds.get(fname, [])
        pred_boxes = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in raw_preds
                      if s >= per_class_conf.get(c, 0.5)]

        matched_gt = set()
        pred_sorted = sorted(enumerate(pred_boxes), key=lambda x: -x[1][1])
        for pi, (p_cls, p_conf, p_x1, p_y1, p_x2, p_y2) in pred_sorted:
            best_iou = 0
            best_gi = -1
            for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if gi in matched_gt or g_cls != p_cls:
                    continue
                iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                matched_gt.add(best_gi)

        for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
            w = g_x2 - g_x1
            h = g_y2 - g_y1
            min_dim = min(w, h)

            if min_dim < 15:
                key = "tiny(<15px)"
            elif min_dim < 30:
                key = "small(15-30px)"
            elif min_dim < 50:
                key = "medium(30-50px)"
            else:
                key = "large(>50px)"

            size_total[key] += 1
            if gi not in matched_gt:
                size_bins[key] += 1

    print(f"\n  {'크기 범주':<20} {'GT 수':>8} {'FN 수':>8} {'Recall':>8}")
    print(f"  {'-'*48}")
    for key in ["tiny(<15px)", "small(15-30px)", "medium(30-50px)", "large(>50px)"]:
        total = size_total[key]
        fn = size_bins[key]
        rec = (total - fn) / total if total > 0 else 0
        print(f"  {key:<20} {total:>8} {fn:>8} {rec:>8.3f}")

    print("\nDone.")
