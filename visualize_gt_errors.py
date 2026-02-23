#!/usr/bin/env python3
"""GT 오류 시각화: 이미지에 GT(초록) + 모델예측(빨강) + 오류유형 표시

prepare_gt_fix_cvat.py의 추론 캐시를 재사용하여 빠르게 시각화.

색상 규칙:
  - 초록(GREEN): 기존 GT (정상)
  - 빨강(RED): 모델이 탐지했으나 GT에 없음 → 미라벨링 후보
  - 노랑(YELLOW): 클래스 혼동 (GT와 모델 클래스 다름)
  - 파랑(BLUE): FN (GT에 있으나 모델 미탐지)
"""
import json
import os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

HOBAN = "/home/lay/hoban"
CACHE_PATH = f"{HOBAN}/cvat_gt_fix/inference_cache.json"
OUT_DIR = f"{HOBAN}/cvat_gt_fix/visualized"

VAL_IMG = f"{HOBAN}/datasets/3k_finetune/val/images"
VAL_LBL = f"{HOBAN}/datasets/3k_finetune/val/labels"
EXTRA_IMG = f"{HOBAN}/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = f"{HOBAN}/datasets/cvat_helmet_off/valid_helmet_off_137/labels"

CLASS_NAMES = {0: "ON", 1: "OFF"}
IGNORE_THRESH = 0.00020
PER_CLASS_CONF = {0: 0.30, 1: 0.15}
CROSS_NMS_IOU = 0.3
MIN_AREA = 5e-05
GATE_CONF = 0.20
GATE_RADIUS = 30

# 색상
GREEN = (0, 200, 0)
RED = (255, 40, 40)
YELLOW = (255, 220, 0)
BLUE = (60, 120, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()
    font_small = font
    font_title = font


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
            boxes.append({
                "cls": cls,
                "x1": (cx - w / 2) * img_w, "y1": (cy - h / 2) * img_h,
                "x2": (cx + w / 2) * img_w, "y2": (cy + h / 2) * img_h,
                "source": "gt",
            })
    return boxes


def compute_iou(b1, b2):
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def bbox_area(b, img_w, img_h):
    return ((b["x2"] - b["x1"]) * (b["y2"] - b["y1"])) / (img_w * img_h)


def cross_class_nms(preds, iou_thresh=0.3):
    if len(preds) <= 1:
        return preds
    sorted_p = sorted(preds, key=lambda x: -x["conf"])
    keep, suppressed = [], set()
    for i in range(len(sorted_p)):
        if i in suppressed:
            continue
        keep.append(sorted_p[i])
        for j in range(i + 1, len(sorted_p)):
            if j in suppressed:
                continue
            if sorted_p[i]["cls"] != sorted_p[j]["cls"]:
                if compute_iou(sorted_p[i], sorted_p[j]) >= iou_thresh:
                    suppressed.add(j)
    return keep


def apply_pipeline(sahi_preds, full_preds, img_w, img_h):
    filtered = cross_class_nms(sahi_preds, CROSS_NMS_IOU)
    img_area = img_w * img_h
    filtered = [p for p in filtered
                if ((p["x2"] - p["x1"]) * (p["y2"] - p["y1"])) / img_area >= MIN_AREA]
    gates = [p for p in full_preds if p["conf"] >= GATE_CONF]
    if gates:
        gated = []
        for p in filtered:
            cx = (p["x1"] + p["x2"]) / 2
            cy = (p["y1"] + p["y2"]) / 2
            for g in gates:
                gcx = (g["x1"] + g["x2"]) / 2
                gcy = (g["y1"] + g["y2"]) / 2
                if abs(cx - gcx) <= GATE_RADIUS and abs(cy - gcy) <= GATE_RADIUS:
                    gated.append(p)
                    break
        filtered = gated
    filtered = [p for p in filtered if p["conf"] >= PER_CLASS_CONF.get(p["cls"], 0.5)]
    return filtered


def draw_box(draw, b, color, label, width=3):
    x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    # label background
    bbox = draw.textbbox((0, 0), label, font=font_small)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ly = max(0, y1 - th - 4)
    draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
    draw.text((x1 + 3, ly + 1), label, fill=WHITE, font=font_small)


def draw_legend(draw, img_w, errors_summary):
    """좌상단 범례"""
    y = 8
    items = [
        (GREEN, "GT (Normal)"),
        (RED, "Missing Label (Add GT)"),
        (YELLOW, "Class Confusion"),
        (BLUE, "FN (Model Missed)"),
    ]
    for color, text in items:
        draw.rectangle([8, y, 24, y + 16], fill=color)
        draw.text((30, y), text, fill=WHITE, font=font_small)
        y += 22

    # 오류 요약
    y += 6
    for line in errors_summary:
        draw.text((10, y), line, fill=WHITE, font=font_small)
        y += 18


def process_image(fname, img_path, lbl_path, sahi_preds, full_preds, out_dir):
    img = Image.open(img_path)
    img_w, img_h = img.size
    gts = load_gt(lbl_path, img_w, img_h)
    preds = apply_pipeline(sahi_preds, full_preds, img_w, img_h)

    active_gts = [g for g in gts if bbox_area(g, img_w, img_h) >= IGNORE_THRESH]
    ignore_gts = [g for g in gts if bbox_area(g, img_w, img_h) < IGNORE_THRESH]

    # 매칭
    matched_gt = set()
    matched_pred = set()
    errors = []

    sorted_preds = sorted(enumerate(preds), key=lambda x: -x[1]["conf"])
    for pi, pred in sorted_preds:
        pred_a = bbox_area(pred, img_w, img_h)

        best_iou, best_gi = 0, -1
        for gi, gt in enumerate(active_gts):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_iou >= 0.5 and best_gi >= 0:
            if pred["cls"] != active_gts[best_gi]["cls"]:
                errors.append(("confusion", pred, active_gts[best_gi]))
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            continue

        # ignore
        ignore_match = False
        for gt in ignore_gts:
            if compute_iou(pred, gt) >= 0.5:
                ignore_match = True
                break
        if ignore_match or pred_a < IGNORE_THRESH:
            matched_pred.add(pi)
            continue

        # FP → missing GT
        errors.append(("missing", pred, None))
        matched_pred.add(pi)

    # FN
    for gi, gt in enumerate(active_gts):
        if gi not in matched_gt:
            errors.append(("fn", None, gt))

    if not errors:
        return None

    # 시각화
    draw = ImageDraw.Draw(img)

    # 1. 정상 매칭된 GT: 초록
    for gi, gt in enumerate(active_gts):
        if gi in matched_gt:
            # confusion 체크
            is_confused = any(e[0] == "confusion" and e[2] is gt for e in errors)
            if not is_confused:
                label = f"GT:{CLASS_NAMES[gt['cls']]}"
                draw_box(draw, gt, GREEN, label, width=2)

    # 2. 오류 표시
    error_summary = []
    n_missing, n_confusion, n_fn = 0, 0, 0

    for etype, pred, gt in errors:
        if etype == "missing":
            n_missing += 1
            conf = pred["conf"]
            label = f"ADD:{CLASS_NAMES[pred['cls']]} {conf:.0%}"
            draw_box(draw, pred, RED, label, width=4)
        elif etype == "confusion":
            n_confusion += 1
            gt_cls = CLASS_NAMES[gt["cls"]]
            pred_cls = CLASS_NAMES[pred["cls"]]
            label = f"FIX:GT={gt_cls}>Pred={pred_cls}"
            draw_box(draw, gt, YELLOW, label, width=4)
        elif etype == "fn":
            n_fn += 1
            label = f"FN:{CLASS_NAMES[gt['cls']]}"
            draw_box(draw, gt, BLUE, label, width=3)

    if n_missing:
        error_summary.append(f"Missing: {n_missing}")
    if n_confusion:
        error_summary.append(f"Confusion: {n_confusion}")
    if n_fn:
        error_summary.append(f"FN: {n_fn}")

    draw_legend(draw, img_w, error_summary)

    # 파일명에 오류 유형 표시
    tag = []
    if n_missing:
        tag.append(f"add{n_missing}")
    if n_confusion:
        tag.append(f"fix{n_confusion}")
    if n_fn:
        tag.append(f"fn{n_fn}")
    tag_str = "_".join(tag)
    out_name = f"{tag_str}__{fname}"
    img.save(os.path.join(out_dir, out_name), quality=95)
    return {"missing": n_missing, "confusion": n_confusion, "fn": n_fn}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  GT 오류 시각화")
    print("=" * 60)

    # 캐시 로드
    print(f"\n캐시 로드: {CACHE_PATH}")
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    all_sahi = cache["sahi"]
    all_full = cache["full"]

    # 평가셋
    eval_images = []
    seen = set()
    for img_dir, lbl_dir in [(VAL_IMG, VAL_LBL), (EXTRA_IMG, EXTRA_LBL)]:
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith(".jpg") and fname not in seen:
                eval_images.append((
                    fname,
                    os.path.join(img_dir, fname),
                    os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
                ))
                seen.add(fname)

    print(f"평가셋: {len(eval_images)}장")
    print(f"출력: {OUT_DIR}")

    totals = defaultdict(int)
    error_count = 0

    for i, (fname, img_path, lbl_path) in enumerate(eval_images):
        if fname not in all_sahi:
            continue

        sahi_preds = [
            {"cls": p["cls"], "conf": p["conf"],
             "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]}
            for p in all_sahi[fname]
        ]
        full_preds = [
            {"cls": p["cls"], "conf": p["conf"],
             "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]}
            for p in all_full[fname]
        ]

        result = process_image(fname, img_path, lbl_path, sahi_preds, full_preds, OUT_DIR)
        if result:
            error_count += 1
            for k, v in result.items():
                totals[k] += v

    print(f"\n{'='*60}")
    print(f"  완료!")
    print(f"{'='*60}")
    print(f"  오류 이미지: {error_count}장")
    print(f"  미라벨링 (빨강): {totals['missing']}개")
    print(f"  클래스혼동 (노랑): {totals['confusion']}개")
    print(f"  FN (파랑): {totals['fn']}개")
    print(f"\n  출력: {OUT_DIR}/")
    print(f"  파일명: add3__cam1_xxx.jpg (미라벨링 3개)")
    print(f"         fix1__cam2_xxx.jpg (클래스혼동 1개)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
