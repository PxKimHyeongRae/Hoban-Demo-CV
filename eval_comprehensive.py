#!/usr/bin/env python3
"""
SAHI 정확도 극한 최적화: 6 Phase 종합 평가
  Phase 1: 모델 교체 (v2 vs v3) + conf sweep
  Phase 2: Overlap 비율 최적화
  Phase 3: Per-Class Confidence
  Phase 4: TTA (perform_standard_pred)
  Phase 5: 모델 앙상블
  Phase 6: Gate + 1280x720

실행: python eval_comprehensive.py
"""
import os
import time
import numpy as np
from collections import defaultdict
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from PIL import Image

# ── 설정 ──
IMG_DIR = "/home/lay/hoban/datasets/3k_finetune/val/images"
LBL_DIR = "/home/lay/hoban/datasets/3k_finetune/val/labels"

MODELS = {
    "v2": "/home/lay/hoban/hoban_go2k_v2/weights/best.pt",      # 주의: 3k val에 82장 누출
    "v3": "/home/lay/hoban/hoban_go2k_v3/weights/best.pt",      # 깨끗 (0 overlap)
    "v16": "/home/lay/hoban/hoban_go3k_v16_640/weights/best.pt", # 깨끗
    "v17": "/home/lay/hoban/hoban_go3k_v17/weights/best.pt",     # 깨끗 (학습 완료 후)
    "v13": "/home/lay/hoban/hoban_v13_stage2/weights/best.pt",   # 깨끗
}

CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}


# ── 유틸 함수 ──
def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


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
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            boxes.append((cls, (cx-w/2)*img_w, (cy-h/2)*img_h, (cx+w/2)*img_w, (cy+h/2)*img_h))
    return boxes


def evaluate(all_gt, all_preds, conf_thresh=0.0):
    tp = fp = fn = 0
    for fname in all_gt:
        gt_boxes = all_gt[fname]
        pred_boxes = [(c, s, x1, y1, x2, y2)
                      for c, s, x1, y1, x2, y2 in all_preds.get(fname, [])
                      if s >= conf_thresh]
        matched_gt = set()
        for p_cls, p_conf, p_x1, p_y1, p_x2, p_y2 in sorted(pred_boxes, key=lambda x: -x[1]):
            best_iou, best_gi = 0, -1
            for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                if gi in matched_gt or g_cls != p_cls:
                    continue
                iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= 0.5 and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp + fp, tp, fp, fn, prec, rec, f1


def evaluate_perclass(all_gt, all_preds, conf0=0.5, conf1=0.5):
    """클래스별 차등 conf로 평가"""
    results = {}
    for cls_id in [0, 1]:
        conf_t = conf0 if cls_id == 0 else conf1
        tp = fp = fn = 0
        for fname in all_gt:
            gt_boxes = [(c, x1, y1, x2, y2) for c, x1, y1, x2, y2 in all_gt[fname] if c == cls_id]
            pred_boxes = [(c, s, x1, y1, x2, y2)
                          for c, s, x1, y1, x2, y2 in all_preds.get(fname, [])
                          if c == cls_id and s >= conf_t]
            matched_gt = set()
            for p_cls, p_conf, p_x1, p_y1, p_x2, p_y2 in sorted(pred_boxes, key=lambda x: -x[1]):
                best_iou, best_gi = 0, -1
                for gi, (g_cls, g_x1, g_y1, g_x2, g_y2) in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou((p_x1, p_y1, p_x2, p_y2), (g_x1, g_y1, g_x2, g_y2))
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= 0.5 and best_gi >= 0:
                    tp += 1
                    matched_gt.add(best_gi)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[cls_id] = (tp + fp, tp, fp, fn, prec, rec, f1)

    # 전체 합산
    total_tp = sum(results[c][1] for c in results)
    total_fp = sum(results[c][2] for c in results)
    total_fn = sum(results[c][3] for c in results)
    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0
    return results, (total_tp + total_fp, total_tp, total_fp, total_fn, total_prec, total_rec, total_f1)


def run_sahi(model_path, images, tile_w=1280, tile_h=720, overlap=0.15,
             conf=0.05, image_size=None, perform_standard_pred=False):
    """SAHI 추론, 낮은 conf로 1회 추론 후 캐시 반환"""
    kwargs = dict(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=conf, device="0",
    )
    if image_size:
        kwargs["image_size"] = image_size
    sahi_model = AutoDetectionModel.from_pretrained(**kwargs)

    all_preds = {}
    for fname in images:
        img_path = os.path.join(IMG_DIR, fname)
        result = get_sliced_prediction(
            img_path, sahi_model,
            slice_height=tile_h, slice_width=tile_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            postprocess_type="NMS",
            postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS",
            perform_standard_pred=perform_standard_pred,
        )
        preds = []
        for p in result.object_prediction_list:
            preds.append((
                p.category.id, p.score.value,
                p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
            ))
        all_preds[fname] = preds
    return all_preds


def run_fullimage(model_path, images, conf=0.05, imgsz=640):
    """풀이미지 YOLO 추론"""
    model = YOLO(model_path)
    model.to("cuda:0")
    all_preds = {}
    for fname in images:
        img_path = os.path.join(IMG_DIR, fname)
        results = model.predict(img_path, imgsz=imgsz, conf=conf, device="0", verbose=False)
        preds = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                preds.append((int(box.cls[0]), float(box.conf[0]),
                              float(x1), float(y1), float(x2), float(y2)))
        all_preds[fname] = preds
    return all_preds


def ensemble_nms(preds_list, iou_thresh=0.5):
    """다중 모델 predictions를 NMS로 병합"""
    merged = {}
    all_fnames = set()
    for preds in preds_list:
        all_fnames.update(preds.keys())

    for fname in all_fnames:
        all_boxes = []
        for preds in preds_list:
            all_boxes.extend(preds.get(fname, []))
        # conf 기준 정렬 후 NMS
        all_boxes.sort(key=lambda x: -x[1])
        keep = []
        for box in all_boxes:
            cls_id, conf, x1, y1, x2, y2 = box
            suppressed = False
            for k_cls, k_conf, k_x1, k_y1, k_x2, k_y2 in keep:
                if cls_id == k_cls and compute_iou((x1, y1, x2, y2), (k_x1, k_y1, k_x2, k_y2)) > iou_thresh:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(box)
        merged[fname] = keep
    return merged


def point_near_any(px, py, gates, radius):
    for gx1, gy1, gx2, gy2 in gates:
        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        if abs(px - gcx) <= radius and abs(py - gcy) <= radius:
            return True
    return False


def apply_gate(sahi_preds, full_preds, gate_conf=0.20, radius=40):
    """풀이미지 게이트 적용"""
    gated = {}
    for fname in sahi_preds:
        gates = [(x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in full_preds.get(fname, [])
                 if s >= gate_conf]
        filtered = []
        for cls_id, conf, x1, y1, x2, y2 in sahi_preds[fname]:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not gates or point_near_any(cx, cy, gates, radius):
                filtered.append((cls_id, conf, x1, y1, x2, y2))
        gated[fname] = filtered
    return gated


def print_header(title):
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    print(f"{'설정':<40} {'예측':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 95)


def print_row(label, n_pred, tp, fp, fn, prec, rec, f1, best_f1=None):
    marker = " <<<" if best_f1 and f1 >= best_f1 - 0.0001 else ""
    print(f"{label:<40} {n_pred:>6} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}{marker}")


# ══════════════════════════════════════════════════════════════════
# GT 로드
# ══════════════════════════════════════════════════════════════════
images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
all_gt = {}
for fname in images:
    img = Image.open(os.path.join(IMG_DIR, fname))
    img_w, img_h = img.size
    all_gt[fname] = load_gt(os.path.join(LBL_DIR, fname.replace(".jpg", ".txt")), img_w, img_h)

total_gt = sum(len(v) for v in all_gt.values())
gt_cls0 = sum(1 for v in all_gt.values() for b in v if b[0] == 0)
gt_cls1 = sum(1 for v in all_gt.values() for b in v if b[0] == 1)
print(f"이미지: {len(images)}장, GT: {total_gt}개 (helmet_on={gt_cls0}, helmet_off={gt_cls1})")

# 결과 수집
all_results = []  # (phase, label, n_pred, tp, fp, fn, prec, rec, f1)

# ══════════════════════════════════════════════════════════════════
# Phase 1: 모델 교체 + conf sweep
# ══════════════════════════════════════════════════════════════════
print_header("Phase 1: 모델 교체 + conf sweep (1280x720, overlap=0.15)")

CONF_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
model_preds = {}  # 캐시

MODEL_IMGSZ = {"v2": None, "v3": 1280, "v16": None, "v17": 1280, "v13": None}
PHASE1_MODELS = ["v16", "v3", "v17", "v2"]  # v2는 82장 누출 주의

for model_name in PHASE1_MODELS:
    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        print(f"  [{model_name}] 모델 없음 (스킵): {model_path}")
        continue

    img_size = MODEL_IMGSZ.get(model_name)
    print(f"\n  [{model_name}] 추론 중 (conf=0.05, image_size={img_size})...", end=" ", flush=True)
    t0 = time.time()
    preds = run_sahi(model_path, images, tile_w=1280, tile_h=720, overlap=0.15,
                     conf=0.05, image_size=img_size)
    elapsed = time.time() - t0
    model_preds[model_name] = preds
    raw = sum(len(v) for v in preds.values())
    print(f"완료 ({elapsed:.0f}s, raw={raw}개)")

    for conf in CONF_SWEEP:
        n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, preds, conf)
        label = f"{model_name} conf={conf:.2f}"
        print_row(label, n_pred, tp, fp, fn, prec, rec, f1)
        all_results.append(("P1", label, n_pred, tp, fp, fn, prec, rec, f1))

# Phase 1 최적 설정 결정
p1_best = max([r for r in all_results if r[0] == "P1"], key=lambda x: x[8], default=None)
if p1_best:
    best_model = "v3" if "v3" in p1_best[1] else "v2"
    best_conf_str = p1_best[1].split("conf=")[1]
    best_conf = float(best_conf_str)
    print(f"\n  >>> Phase 1 최적: {p1_best[1]} (F1={p1_best[8]:.3f})")
else:
    best_model = "v2"
    best_conf = 0.50

# ══════════════════════════════════════════════════════════════════
# Phase 2: Overlap 비율 최적화
# ══════════════════════════════════════════════════════════════════
print_header(f"Phase 2: Overlap 최적화 ({best_model} 모델)")

OVERLAP_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
best_model_path = MODELS[best_model]
img_size = 1280 if best_model == "v3" else None
overlap_preds = {}

for overlap in OVERLAP_SWEEP:
    print(f"\n  overlap={overlap:.2f} 추론 중...", end=" ", flush=True)
    t0 = time.time()
    preds = run_sahi(best_model_path, images, tile_w=1280, tile_h=720,
                     overlap=overlap, conf=0.05, image_size=img_size)
    elapsed = time.time() - t0
    overlap_preds[overlap] = preds
    print(f"완료 ({elapsed:.0f}s)")

    # best_conf 근처 sweep
    for conf in [best_conf - 0.10, best_conf - 0.05, best_conf, best_conf + 0.05, best_conf + 0.10]:
        if conf < 0.20 or conf > 0.70:
            continue
        n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, preds, conf)
        label = f"overlap={overlap:.2f} conf={conf:.2f}"
        print_row(label, n_pred, tp, fp, fn, prec, rec, f1)
        all_results.append(("P2", label, n_pred, tp, fp, fn, prec, rec, f1))

p2_best = max([r for r in all_results if r[0] == "P2"], key=lambda x: x[8], default=None)
if p2_best:
    best_overlap = float(p2_best[1].split("overlap=")[1].split(" ")[0])
    best_conf2 = float(p2_best[1].split("conf=")[1])
    print(f"\n  >>> Phase 2 최적: {p2_best[1]} (F1={p2_best[8]:.3f})")
else:
    best_overlap = 0.15
    best_conf2 = best_conf

# Phase 2 최적 predictions 캐시
best_preds = overlap_preds.get(best_overlap, model_preds.get(best_model, {}))

# ══════════════════════════════════════════════════════════════════
# Phase 3: Per-Class Confidence
# ══════════════════════════════════════════════════════════════════
print_header("Phase 3: Per-Class Confidence (post-hoc)")
print(f"  GT 분포: helmet_on={gt_cls0}, helmet_off={gt_cls1}")

CONF0_SWEEP = [0.40, 0.45, 0.50, 0.55, 0.60]  # helmet_on
CONF1_SWEEP = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # helmet_off

best_perclass_f1 = 0
best_perclass_label = ""

for c0 in CONF0_SWEEP:
    for c1 in CONF1_SWEEP:
        cls_results, total = evaluate_perclass(all_gt, best_preds, conf0=c0, conf1=c1)
        n_pred, tp, fp, fn, prec, rec, f1 = total
        label = f"cls0={c0:.2f} cls1={c1:.2f}"
        all_results.append(("P3", label, n_pred, tp, fp, fn, prec, rec, f1))
        if f1 > best_perclass_f1:
            best_perclass_f1 = f1
            best_perclass_label = label
            best_perclass_detail = cls_results

# Top 5만 출력
p3_sorted = sorted([r for r in all_results if r[0] == "P3"], key=lambda x: -x[8])
for r in p3_sorted[:10]:
    print_row(r[1], *r[2:])

if best_perclass_detail:
    print(f"\n  >>> Phase 3 최적: {best_perclass_label} (F1={best_perclass_f1:.3f})")
    for cls_id in [0, 1]:
        n, tp, fp, fn, p, r, f = best_perclass_detail[cls_id]
        print(f"      {CLASS_NAMES[cls_id]}: P={p:.3f} R={r:.3f} F1={f:.3f} (TP={tp} FP={fp} FN={fn})")

# ══════════════════════════════════════════════════════════════════
# Phase 4: TTA (perform_standard_pred=True)
# ══════════════════════════════════════════════════════════════════
print_header(f"Phase 4: TTA - perform_standard_pred ({best_model} 모델)")

print(f"\n  perform_standard_pred=True 추론 중...", end=" ", flush=True)
t0 = time.time()
tta_preds = run_sahi(best_model_path, images, tile_w=1280, tile_h=720,
                     overlap=best_overlap, conf=0.05, image_size=img_size,
                     perform_standard_pred=True)
elapsed = time.time() - t0
print(f"완료 ({elapsed:.0f}s)")

for conf in [best_conf2 - 0.10, best_conf2 - 0.05, best_conf2, best_conf2 + 0.05, best_conf2 + 0.10]:
    if conf < 0.20 or conf > 0.70:
        continue
    n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, tta_preds, conf)
    label = f"standard_pred conf={conf:.2f}"
    print_row(label, n_pred, tp, fp, fn, prec, rec, f1)
    all_results.append(("P4", label, n_pred, tp, fp, fn, prec, rec, f1))

p4_best = max([r for r in all_results if r[0] == "P4"], key=lambda x: x[8], default=None)
if p4_best:
    print(f"\n  >>> Phase 4 최적: {p4_best[1]} (F1={p4_best[8]:.3f})")

# ══════════════════════════════════════════════════════════════════
# Phase 5: 모델 앙상블
# ══════════════════════════════════════════════════════════════════
print_header("Phase 5: 모델 앙상블")

# v13 추론 (앙상블용)
if os.path.exists(MODELS["v13"]) and "v13" not in model_preds:
    print(f"\n  v13 추론 중...", end=" ", flush=True)
    t0 = time.time()
    v13_preds = run_sahi(MODELS["v13"], images, tile_w=1280, tile_h=720,
                         overlap=best_overlap, conf=0.05)
    elapsed = time.time() - t0
    model_preds["v13"] = v13_preds
    print(f"완료 ({elapsed:.0f}s)")

# 가용 모델로 2-모델 앙상블 조합 테스트
from itertools import combinations
available = [k for k in model_preds]
print(f"\n  가용 모델: {available}")
print(f"  (주의: v2는 3k val에 82장 누출)")

for combo in combinations(available, 2):
    combo_name = "+".join(combo)
    print(f"\n  {combo_name} 앙상블 (NMS)...")
    ens = ensemble_nms([model_preds[m] for m in combo], iou_thresh=0.5)
    for conf in CONF_SWEEP:
        n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, ens, conf)
        label = f"{combo_name} NMS conf={conf:.2f}"
        print_row(label, n_pred, tp, fp, fn, prec, rec, f1)
        all_results.append(("P5", label, n_pred, tp, fp, fn, prec, rec, f1))

# 3-모델 앙상블 (가용 모델 3개 이상일 때)
if len(available) >= 3:
    for combo in combinations(available, 3):
        combo_name = "+".join(combo)
        print(f"\n  {combo_name} 앙상블 (NMS)...")
        ens = ensemble_nms([model_preds[m] for m in combo], iou_thresh=0.5)
        for conf in CONF_SWEEP:
            n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, ens, conf)
            label = f"{combo_name} NMS conf={conf:.2f}"
            print_row(label, n_pred, tp, fp, fn, prec, rec, f1)
            all_results.append(("P5", label, n_pred, tp, fp, fn, prec, rec, f1))

p5_best = max([r for r in all_results if r[0] == "P5"], key=lambda x: x[8], default=None)
if p5_best:
    print(f"\n  >>> Phase 5 최적: {p5_best[1]} (F1={p5_best[8]:.3f})")

# ══════════════════════════════════════════════════════════════════
# Phase 6: Gate + 1280x720
# ══════════════════════════════════════════════════════════════════
print_header(f"Phase 6: Gate + 1280x720 ({best_model} 모델)")

print(f"\n  풀이미지 추론 중 (gate용)...", end=" ", flush=True)
t0 = time.time()
full_preds = run_fullimage(best_model_path, images, conf=0.05, imgsz=640)
elapsed = time.time() - t0
print(f"완료 ({elapsed:.0f}s)")

GATE_CONFIGS = [
    (0.15, 40), (0.20, 40), (0.25, 40),
    (0.15, 60), (0.20, 60), (0.25, 60),
    (0.20, 30), (0.25, 30),
]

for gate_conf, radius in GATE_CONFIGS:
    gated = apply_gate(best_preds, full_preds, gate_conf=gate_conf, radius=radius)
    n_pred, tp, fp, fn, prec, rec, f1 = evaluate(all_gt, gated, best_conf2)
    label = f"gate c={gate_conf} r={radius} conf={best_conf2:.2f}"
    print_row(label, n_pred, tp, fp, fn, prec, rec, f1)
    all_results.append(("P6", label, n_pred, tp, fp, fn, prec, rec, f1))

p6_best = max([r for r in all_results if r[0] == "P6"], key=lambda x: x[8], default=None)
if p6_best:
    print(f"\n  >>> Phase 6 최적: {p6_best[1]} (F1={p6_best[8]:.3f})")

# ══════════════════════════════════════════════════════════════════
# 최종 Top 10 요약
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*95}")
print(f"  최종 Top 15 설정 (전체 Phase 통합)")
print(f"{'='*95}")
print(f"{'Phase':<6} {'설정':<40} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("-" * 95)

top_results = sorted(all_results, key=lambda x: -x[8])[:15]
for phase, label, n_pred, tp, fp, fn, prec, rec, f1 in top_results:
    print(f"{phase:<6} {label:<40} {tp:>6} {fp:>6} {fn:>6} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

print(f"\n{'='*95}")
best = top_results[0]
print(f"  최고 설정: [{best[0]}] {best[1]}")
print(f"  F1={best[8]:.3f}, P={best[6]:.3f}, R={best[7]:.3f}")
print(f"  TP={best[3]}, FP={best[4]}, FN={best[5]}")
print(f"{'='*95}")
