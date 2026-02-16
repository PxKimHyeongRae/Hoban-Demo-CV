#!/usr/bin/env python3
"""
Phase B: 추론 최적화 실험 (재학습 없음)
B1: WBF vs NMS 앙상블 (v2+v3)
B2: SAHI overlap sweep
B3: Multi-model ensemble (v2+v3+v5)
B4: Per-class conf + WBF 조합
"""
import os
import time
import numpy as np
from collections import defaultdict
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion, nms as ens_nms

# === Config ===
CLASS_NAMES = {0: "person_with_helmet", 1: "person_without_helmet"}
IMG_DIR = "/home/lay/hoban/datasets/go2k_manual/images"
LBL_DIR = "/home/lay/hoban/datasets/go2k_manual/labels"

MODELS = {
    "v2": "/home/lay/hoban/hoban_go2k_v2/weights/best.pt",
    "v3": "/home/lay/hoban/hoban_go2k_v3/weights/best.pt",
    "v5": "/home/lay/hoban/hoban_go2k_v5/weights/best.pt",
}

# === GT Loading ===
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

def evaluate(all_gt, all_preds, conf_thresh=None, per_class_conf=None, iou_thresh=0.5):
    """Evaluate predictions against GT. Supports uniform or per-class conf."""
    sum_tp = sum_fp = sum_fn = 0
    for fname in all_gt:
        gt_boxes = all_gt[fname]
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
                matched_gt.add(best_gi)
            else:
                sum_fp += 1
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                sum_fn += 1

    prec = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return sum_tp, sum_fp, sum_fn, prec, rec, f1


def run_sahi_inference(model_path, images, img_sizes, device="0",
                       tile_w=1280, tile_h=720, overlap=0.15,
                       pp_type="NMS", pp_thresh=0.4, pp_metric="IOS",
                       image_size=None):
    """Run SAHI inference and return raw predictions (conf=0.05)."""
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
            postprocess_type=pp_type,
            postprocess_match_threshold=pp_thresh,
            postprocess_match_metric=pp_metric,
            postprocess_class_agnostic=False,
        )
        preds = []
        for p in result.object_prediction_list:
            bbox = p.bbox
            preds.append((p.category.id, p.score.value, bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))
        all_preds[fname] = preds

    print(f"  Done: {sum(len(v) for v in all_preds.values())} raw predictions")
    return all_preds


def wbf_ensemble(preds_list, img_sizes, iou_thr=0.5, skip_box_thr=0.0001, weights=None):
    """Merge predictions from multiple models using WBF."""
    all_merged = {}
    for fname in preds_list[0]:
        img_w, img_h = img_sizes[fname]

        boxes_list = []
        scores_list = []
        labels_list = []

        for preds in preds_list:
            raw = preds.get(fname, [])
            if not raw:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
                continue

            boxes = []
            scores = []
            labels = []
            for cls, score, x1, y1, x2, y2 in raw:
                # Normalize to [0, 1]
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
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        result = []
        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            x1 = box[0] * img_w
            y1 = box[1] * img_h
            x2 = box[2] * img_w
            y2 = box[3] * img_h
            result.append((int(label), float(score), x1, y1, x2, y2))
        all_merged[fname] = result

    return all_merged


def nms_ensemble(preds_list, img_sizes, iou_thr=0.5):
    """Merge predictions from multiple models using NMS."""
    all_merged = {}
    for fname in preds_list[0]:
        img_w, img_h = img_sizes[fname]

        boxes_list = []
        scores_list = []
        labels_list = []

        for preds in preds_list:
            raw = preds.get(fname, [])
            for cls, score, x1, y1, x2, y2 in raw:
                boxes_list.append([x1/img_w, y1/img_h, x2/img_w, y2/img_h])
                scores_list.append(score)
                labels_list.append(cls)

        if not boxes_list:
            all_merged[fname] = []
            continue

        # Use ensemble_boxes NMS
        boxes_arr = [boxes_list]
        scores_arr = [scores_list]
        labels_arr = [labels_list]

        merged_boxes, merged_scores, merged_labels = ens_nms(
            boxes_arr, scores_arr, labels_arr, iou_thr=iou_thr,
        )

        result = []
        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            x1 = box[0] * img_w
            y1 = box[1] * img_h
            x2 = box[2] * img_w
            y2 = box[3] * img_h
            result.append((int(label), float(score), x1, y1, x2, y2))
        all_merged[fname] = result

    return all_merged


def print_results(title, results):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    print(f"{'설정':<40} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-"*80)
    best_f1 = max(r[-1] for r in results)
    for row in results:
        label = row[0]
        tp, fp, fn, prec, rec, f1 = row[1:]
        marker = " <<<" if f1 == best_f1 else ""
        print(f"{label:<40} {tp:>5} {fp:>5} {fn:>5} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}{marker}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Phase B: 추론 최적화 실험\n")

    # Load GT
    images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
    all_gt = {}
    img_sizes = {}
    for fname in images:
        img = Image.open(os.path.join(IMG_DIR, fname))
        img_w, img_h = img.size
        img_sizes[fname] = (img_w, img_h)
        all_gt[fname] = load_gt(os.path.join(LBL_DIR, fname.replace(".jpg", ".txt")), img_w, img_h)

    total_gt = sum(len(v) for v in all_gt.values())
    print(f"GT: {len(images)}장, {total_gt} bbox\n")

    # ========================================
    # Phase B1: Model inference (cache all)
    # ========================================
    print("=" * 60)
    print("Phase B1: 모델별 SAHI 추론 (캐시)")
    print("=" * 60)

    pred_cache = {}

    # v2: 640px model, 1280x720 tiles
    print("\n[v2] 1280x720 tiles, overlap=0.15")
    t0 = time.time()
    pred_cache["v2"] = run_sahi_inference(
        MODELS["v2"], images, img_sizes,
        tile_w=1280, tile_h=720, overlap=0.15,
    )
    print(f"  Time: {time.time()-t0:.0f}s")

    # v3: 1280px model, 1280x720 tiles, image_size=1280
    print("\n[v3] 1280x720 tiles, overlap=0.15, image_size=1280")
    t0 = time.time()
    pred_cache["v3"] = run_sahi_inference(
        MODELS["v3"], images, img_sizes,
        tile_w=1280, tile_h=720, overlap=0.15,
        image_size=1280,
    )
    print(f"  Time: {time.time()-t0:.0f}s")

    # v5
    print("\n[v5] 1280x720 tiles, overlap=0.15")
    t0 = time.time()
    pred_cache["v5"] = run_sahi_inference(
        MODELS["v5"], images, img_sizes,
        tile_w=1280, tile_h=720, overlap=0.15,
    )
    print(f"  Time: {time.time()-t0:.0f}s")

    # ========================================
    # Baseline: single model results
    # ========================================
    print("\n\n" + "=" * 60)
    print("Baseline: 단일 모델 결과")
    print("=" * 60)

    baseline_results = []
    for model_name in ["v2", "v3", "v5"]:
        for conf in [0.40, 0.45, 0.50, 0.55, 0.60]:
            tp, fp, fn, prec, rec, f1 = evaluate(all_gt, pred_cache[model_name], conf_thresh=conf)
            baseline_results.append((f"{model_name} conf={conf:.2f}", tp, fp, fn, prec, rec, f1))

    print_results("단일 모델 + uniform conf", baseline_results)

    # Per-class conf for single models
    perclass_results = []
    for model_name in ["v2", "v3", "v5"]:
        for c0 in [0.50, 0.55, 0.60, 0.65, 0.70]:
            for c1 in [0.25, 0.30, 0.35, 0.40]:
                tp, fp, fn, prec, rec, f1 = evaluate(
                    all_gt, pred_cache[model_name],
                    per_class_conf={0: c0, 1: c1}
                )
                perclass_results.append((f"{model_name} c0={c0:.2f} c1={c1:.2f}", tp, fp, fn, prec, rec, f1))

    # Sort by F1, show top 10
    perclass_results.sort(key=lambda x: -x[-1])
    print_results("단일 모델 + per-class conf (Top 10)", perclass_results[:10])

    # ========================================
    # Phase B2: WBF vs NMS ensemble
    # ========================================
    print("\n\n" + "=" * 60)
    print("Phase B2: WBF vs NMS 앙상블")
    print("=" * 60)

    ensemble_results = []

    # v2+v3 combinations
    for iou_thr in [0.4, 0.5, 0.6]:
        wbf_preds = wbf_ensemble(
            [pred_cache["v2"], pred_cache["v3"]], img_sizes,
            iou_thr=iou_thr, weights=[1, 1]
        )
        nms_preds = nms_ensemble(
            [pred_cache["v2"], pred_cache["v3"]], img_sizes,
            iou_thr=iou_thr
        )

        for conf in [0.40, 0.45, 0.50, 0.55]:
            tp, fp, fn, prec, rec, f1 = evaluate(all_gt, wbf_preds, conf_thresh=conf)
            ensemble_results.append((f"v2+v3 WBF iou={iou_thr} conf={conf:.2f}", tp, fp, fn, prec, rec, f1))

            tp, fp, fn, prec, rec, f1 = evaluate(all_gt, nms_preds, conf_thresh=conf)
            ensemble_results.append((f"v2+v3 NMS iou={iou_thr} conf={conf:.2f}", tp, fp, fn, prec, rec, f1))

    ensemble_results.sort(key=lambda x: -x[-1])
    print_results("v2+v3 WBF vs NMS (Top 15)", ensemble_results[:15])

    # ========================================
    # Phase B3: WBF + per-class conf
    # ========================================
    print("\n\n" + "=" * 60)
    print("Phase B3: WBF 앙상블 + per-class conf")
    print("=" * 60)

    # Best WBF iou from B2
    wbf_perclass_results = []

    for iou_thr in [0.4, 0.5, 0.6]:
        # v2+v3
        wbf_preds_23 = wbf_ensemble(
            [pred_cache["v2"], pred_cache["v3"]], img_sizes,
            iou_thr=iou_thr, weights=[1, 1]
        )
        # v2+v3+v5
        wbf_preds_235 = wbf_ensemble(
            [pred_cache["v2"], pred_cache["v3"], pred_cache["v5"]], img_sizes,
            iou_thr=iou_thr, weights=[1, 1, 0.5]
        )

        for c0 in [0.50, 0.55, 0.60, 0.65, 0.70]:
            for c1 in [0.25, 0.30, 0.35, 0.40]:
                tp, fp, fn, prec, rec, f1 = evaluate(
                    all_gt, wbf_preds_23,
                    per_class_conf={0: c0, 1: c1}
                )
                wbf_perclass_results.append((
                    f"v2+v3 WBF/{iou_thr} c0={c0:.2f} c1={c1:.2f}",
                    tp, fp, fn, prec, rec, f1
                ))

                tp, fp, fn, prec, rec, f1 = evaluate(
                    all_gt, wbf_preds_235,
                    per_class_conf={0: c0, 1: c1}
                )
                wbf_perclass_results.append((
                    f"v2+v3+v5 WBF/{iou_thr} c0={c0:.2f} c1={c1:.2f}",
                    tp, fp, fn, prec, rec, f1
                ))

    wbf_perclass_results.sort(key=lambda x: -x[-1])
    print_results("WBF 앙상블 + per-class conf (Top 20)", wbf_perclass_results[:20])

    # ========================================
    # Phase B4: Overlap sweep (v2 model)
    # ========================================
    print("\n\n" + "=" * 60)
    print("Phase B4: SAHI overlap sweep (v2 모델)")
    print("=" * 60)

    overlap_results = []
    for overlap in [0.05, 0.10, 0.20, 0.25, 0.30]:
        print(f"\n  overlap={overlap}...")
        t0 = time.time()
        preds = run_sahi_inference(
            MODELS["v2"], images, img_sizes,
            tile_w=1280, tile_h=720, overlap=overlap,
        )
        print(f"  Time: {time.time()-t0:.0f}s")

        for conf in [0.45, 0.50, 0.55]:
            tp, fp, fn, prec, rec, f1 = evaluate(all_gt, preds, conf_thresh=conf)
            overlap_results.append((f"overlap={overlap:.2f} conf={conf:.2f}", tp, fp, fn, prec, rec, f1))

    # Add overlap=0.15 from cache
    for conf in [0.45, 0.50, 0.55]:
        tp, fp, fn, prec, rec, f1 = evaluate(all_gt, pred_cache["v2"], conf_thresh=conf)
        overlap_results.append((f"overlap=0.15 conf={conf:.2f}", tp, fp, fn, prec, rec, f1))

    overlap_results.sort(key=lambda x: -x[-1])
    print_results("SAHI Overlap Sweep (v2)", overlap_results)

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n\n" + "=" * 80)
    print(" FINAL: 전체 Top 10")
    print("=" * 80)

    all_results = baseline_results + perclass_results + ensemble_results + wbf_perclass_results + overlap_results
    all_results.sort(key=lambda x: -x[-1])
    print_results("전체 실험 Top 10", all_results[:10])

    # Also show best by Precision (FP minimization priority)
    high_prec = [r for r in all_results if r[4] >= 0.93]  # Prec >= 0.93
    if high_prec:
        high_prec.sort(key=lambda x: -x[-1])
        print_results("Precision >= 0.93 중 Top 10 (오탐 최소화)", high_prec[:10])
