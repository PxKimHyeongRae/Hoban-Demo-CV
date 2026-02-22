#!/usr/bin/env python3
"""Ignore-aware 평가 후 남은 FP/FN 크롭 분석

area < 0.00020 ignore 적용 후 남은 오류만 크롭으로 저장.
v19 모델 기준, best conf (c0=0.30, c1=0.25)
"""
import os, time, logging
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

MODEL = "/home/lay/hoban/hoban_go3k_v19/weights/best.pt"
OUT_DIR = "/home/lay/hoban/analysis/remaining_errors"
os.makedirs(OUT_DIR, exist_ok=True)

VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"

IGNORE_THRESH = 0.00020
CONF = {0: 0.30, 1: 0.25}  # best conf from ignore-aware eval
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except:
    font = ImageFont.load_default()


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


def bbox_area_ratio(x1, y1, x2, y2, img_w, img_h):
    return ((x2-x1) * (y2-y1)) / (img_w * img_h)


def cross_class_nms(preds, iou_thresh=0.3):
    if len(preds) <= 1:
        return preds
    sorted_preds = sorted(preds, key=lambda x: -x[1])
    keep, suppressed = [], set()
    for i in range(len(sorted_preds)):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        c1, b1 = sorted_preds[i][0], sorted_preds[i][2:]
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            if c1 != sorted_preds[j][0]:
                if compute_iou(b1, sorted_preds[j][2:]) >= iou_thresh:
                    suppressed.add(j)
    return keep


def apply_pipeline(preds, full_raw, img_w, img_h):
    filtered = cross_class_nms(preds, 0.3)
    img_area = img_w * img_h
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if ((x2-x1)*(y2-y1)) / img_area >= 5e-05]
    gates = [(x1,y1,x2,y2) for conf,x1,y1,x2,y2 in full_raw if conf >= 0.20]
    if gates:
        gated = []
        for c,s,x1,y1,x2,y2 in filtered:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            for gx1,gy1,gx2,gy2 in gates:
                gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
                if abs(cx-gcx) <= 30 and abs(cy-gcy) <= 30:
                    gated.append((c,s,x1,y1,x2,y2))
                    break
        filtered = gated
    return filtered


def save_crop(img, bbox, label, out_path, pad=60):
    """bbox 주변 크롭 저장 (패딩 포함)"""
    x1, y1, x2, y2 = bbox
    img_w, img_h = img.size
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(img_w, int(x2) + pad)
    cy2 = min(img_h, int(y2) + pad)

    crop = img.crop((cx1, cy1, cx2, cy2))
    # 최소 크기 보장 (작으면 리사이즈)
    min_size = 200
    w, h = crop.size
    if w < min_size or h < min_size:
        scale = max(min_size / w, min_size / h)
        crop = crop.resize((int(w*scale), int(h*scale)), Image.NEAREST)

    draw = ImageDraw.Draw(crop)
    # bbox 위치 (크롭 좌표계)
    scale_x = crop.size[0] / (cx2 - cx1)
    scale_y = crop.size[1] / (cy2 - cy1)
    bx1 = (x1 - cx1) * scale_x
    by1 = (y1 - cy1) * scale_y
    bx2 = (x2 - cx1) * scale_x
    by2 = (y2 - cy1) * scale_y
    draw.rectangle([bx1, by1, bx2, by2], outline="red", width=2)
    draw.text((bx1, by1-16), label, fill="red", font=font)

    crop.save(out_path, quality=95)


# 이미지 수집
eval_images = []
for img_dir, lbl_dir in [(VAL_IMG, VAL_LBL), (EXTRA_IMG, EXTRA_LBL)]:
    if not os.path.isdir(img_dir):
        continue
    seen = set(e[0] for e in eval_images)
    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith(".jpg") and fname not in seen:
            eval_images.append((
                fname,
                os.path.join(img_dir, fname),
                os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
            ))

print(f"평가셋: {len(eval_images)}장")
print(f"모델: {MODEL}")
print(f"Ignore threshold: area < {IGNORE_THRESH}")
print(f"Per-class conf: ON>={CONF[0]}, OFF>={CONF[1]}")

# 모델 로드
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

print("\nSAHI 모델 로드...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=MODEL,
    confidence_threshold=0.05, device="0", image_size=1280)

print("Full-image 모델 로드...")
yolo_model = YOLO(MODEL)

# 분석
fp_list = []  # (fname, cls, conf, bbox, area, error_type)
fn_list = []  # (fname, cls, bbox, area)

t0 = time.time()
for i, (fname, img_path, lbl_path) in enumerate(eval_images):
    if i % 50 == 0:
        print(f"  {i}/{len(eval_images)}...", end="\r")

    img = Image.open(img_path)
    img_w, img_h = img.size

    # SAHI 추론
    r = get_sliced_prediction(
        img_path, sahi_model,
        slice_height=720, slice_width=1280,
        overlap_height_ratio=0.15, overlap_width_ratio=0.15,
        perform_standard_pred=True,
        postprocess_type="NMS", postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS", verbose=0)
    raw_preds = [(p.category.id, p.score.value,
                  p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                 for p in r.object_prediction_list]

    # Full-image (Gate)
    results = yolo_model.predict(img_path, conf=0.01, imgsz=1280,
                                 device="0", verbose=False)
    boxes = results[0].boxes
    full_raw = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                for j in range(len(boxes))]

    # 후처리
    preds = apply_pipeline(raw_preds, full_raw, img_w, img_h)

    # conf 필터
    preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in preds
             if s >= CONF.get(c, 0.5)]

    # GT 로드 + active/ignore 분리
    gts = load_gt(lbl_path, img_w, img_h)
    active_gts = []
    ignore_gts = []
    for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
        area = bbox_area_ratio(gx1, gy1, gx2, gy2, img_w, img_h)
        if area >= IGNORE_THRESH:
            active_gts.append((gi, gc, gx1, gy1, gx2, gy2))
        else:
            ignore_gts.append((gi, gc, gx1, gy1, gx2, gy2))

    # 매칭
    matched_active = set()
    matched_ignore = set()

    sorted_p = sorted(enumerate(preds), key=lambda x: -x[1][1])
    for pi, (pc, ps, px1, py1, px2, py2) in sorted_p:
        pred_area = bbox_area_ratio(px1, py1, px2, py2, img_w, img_h)

        # active GT 매칭
        best_iou, best_idx = 0, -1
        for ai, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(active_gts):
            if ai in matched_active or gc != pc:
                continue
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou, best_idx = iou, ai

        if best_iou >= 0.5 and best_idx >= 0:
            matched_active.add(best_idx)
            continue

        # ignore GT 매칭
        ignore_match = False
        for ii, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(ignore_gts):
            if ii in matched_ignore or gc != pc:
                continue
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou >= 0.5:
                ignore_match = True
                matched_ignore.add(ii)
                break

        if ignore_match or pred_area < IGNORE_THRESH:
            continue

        # FP — 분류
        # GT에 다른 클래스로 매칭되는지 확인 (class confusion)
        best_iou_any, best_gt_cls = 0, -1
        for ai, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(active_gts):
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou_any:
                best_iou_any, best_gt_cls = iou, gc

        if best_iou_any >= 0.5 and best_gt_cls != pc:
            if pc == 1 and best_gt_cls == 0:
                error_type = "ON→OFF"  # helmet_on을 helmet_off로 잘못 탐지 (false alarm)
            else:
                error_type = "OFF→ON"  # helmet_off를 helmet_on으로 (놓침)
        else:
            if pc == 1:
                error_type = "BG→OFF"  # 배경을 helmet_off로 (false alarm)
            else:
                error_type = "BG→ON"   # 배경을 helmet_on으로 (무해)

        fp_list.append((fname, pc, ps, (px1, py1, px2, py2), pred_area, error_type))

    # FN — 미매칭 active GT
    for ai, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(active_gts):
        if ai not in matched_active:
            area = bbox_area_ratio(gx1, gy1, gx2, gy2, img_w, img_h)
            fn_list.append((fname, gc, (gx1, gy1, gx2, gy2), area))

elapsed = time.time() - t0
print(f"\n추론 완료 ({elapsed:.0f}s)")

# ── 통계 ──
print(f"\n{'='*70}")
print(f"  Ignore-aware 오류 분석 (area < {IGNORE_THRESH} 제외)")
print(f"{'='*70}")

print(f"\n  FP: {len(fp_list)}개")
fp_types = defaultdict(list)
for item in fp_list:
    fp_types[item[5]].append(item)
for etype in ["ON→OFF", "OFF→ON", "BG→OFF", "BG→ON"]:
    items = fp_types.get(etype, [])
    danger = "위험" if etype in ["ON→OFF", "BG→OFF"] else ("놓침" if etype == "OFF→ON" else "무해")
    print(f"    {etype}: {len(items)}개 ({danger})")

print(f"\n  FN: {len(fn_list)}개")
fn_on = [x for x in fn_list if x[1] == 0]
fn_off = [x for x in fn_list if x[1] == 1]
print(f"    helmet_on FN: {len(fn_on)}개 (무해)")
print(f"    helmet_off FN: {len(fn_off)}개 (위험)")

# 크기 분포
print(f"\n  FP area 분포:")
for label, thresh in [("<0.0005", 0.0005), ("<0.001", 0.001), ("<0.005", 0.005), (">=0.005", 999)]:
    if thresh == 999:
        count = sum(1 for x in fp_list if x[4] >= 0.005)
    elif thresh == 0.0005:
        count = sum(1 for x in fp_list if x[4] < 0.0005)
    elif thresh == 0.001:
        count = sum(1 for x in fp_list if 0.0005 <= x[4] < 0.001)
    else:
        count = sum(1 for x in fp_list if 0.001 <= x[4] < 0.005)
    print(f"    {label}: {count}개")

print(f"\n  FN area 분포:")
for label, thresh in [("<0.0005", 0.0005), ("<0.001", 0.001), ("<0.005", 0.005), (">=0.005", 999)]:
    if thresh == 999:
        count = sum(1 for x in fn_list if x[3] >= 0.005)
    elif thresh == 0.0005:
        count = sum(1 for x in fn_list if x[3] < 0.0005)
    elif thresh == 0.001:
        count = sum(1 for x in fn_list if 0.0005 <= x[3] < 0.001)
    else:
        count = sum(1 for x in fn_list if 0.001 <= x[3] < 0.005)
    print(f"    {label}: {count}개")

# ── 크롭 저장 ──
print(f"\n크롭 저장 중...")
for subdir in ["fp_on2off", "fp_off2on", "fp_bg_off", "fp_bg_on", "fn_on", "fn_off"]:
    os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)

# FP 크롭
for idx, (fname, cls, conf, bbox, area, etype) in enumerate(
        sorted(fp_list, key=lambda x: -x[2])):  # conf 내림차순
    img = Image.open(next(
        ip for fn, ip, lp in eval_images if fn == fname))

    if etype == "ON→OFF":
        subdir = "fp_on2off"
    elif etype == "OFF→ON":
        subdir = "fp_off2on"
    elif etype == "BG→OFF":
        subdir = "fp_bg_off"
    else:
        subdir = "fp_bg_on"

    label = f"{etype} c={conf:.2f} a={area:.5f}"
    out_name = f"{idx:03d}_{etype.replace('→','2')}_{conf:.2f}_{fname}"
    save_crop(img, bbox, label, os.path.join(OUT_DIR, subdir, out_name))

# FN 크롭
for idx, (fname, cls, bbox, area) in enumerate(
        sorted(fn_list, key=lambda x: -x[3])):  # area 내림차순
    img = Image.open(next(
        ip for fn, ip, lp in eval_images if fn == fname))

    subdir = "fn_off" if cls == 1 else "fn_on"
    cls_name = CLASS_NAMES[cls]
    label = f"FN {cls_name} a={area:.5f}"
    out_name = f"{idx:03d}_fn_{cls_name}_{area:.5f}_{fname}"
    save_crop(img, bbox, label, os.path.join(OUT_DIR, subdir, out_name))

# 크롭 수 보고
for subdir in ["fp_on2off", "fp_off2on", "fp_bg_off", "fp_bg_on", "fn_on", "fn_off"]:
    path = os.path.join(OUT_DIR, subdir)
    count = len([f for f in os.listdir(path) if f.endswith(".jpg")])
    print(f"  {subdir}: {count}개")

print(f"\n출력: {OUT_DIR}")

# 위험 오류 요약
critical_fp = len(fp_types.get("ON→OFF", [])) + len(fp_types.get("BG→OFF", []))
critical_fn = len(fn_off)
print(f"\n{'='*70}")
print(f"  위험 오류 요약 (실제 대응 필요)")
print(f"{'='*70}")
print(f"  False Alarm (없는데 OFF 알림): {critical_fp}개")
print(f"    - ON→OFF 혼동: {len(fp_types.get('ON→OFF', []))}개")
print(f"    - 배경→OFF: {len(fp_types.get('BG→OFF', []))}개")
print(f"  Missed Danger (OFF인데 놓침): {critical_fn + len(fp_types.get('OFF→ON', []))}개")
print(f"    - OFF→ON 혼동: {len(fp_types.get('OFF→ON', []))}개")
print(f"    - FN OFF: {critical_fn}개")
print(f"  무해 오류: {len(fp_types.get('BG→ON', []))}개 + FN ON {len(fn_on)}개")
print(f"{'='*70}")
