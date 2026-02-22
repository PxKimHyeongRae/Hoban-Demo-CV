#!/usr/bin/env python3
"""v23 모델 평가 결과 시각화 (GT + Pred bbox)

729장 평가셋에 대해 GT(점선)와 Pred(실선) bbox를 함께 그려서 저장.
후처리 파이프라인 적용: cross_nms → min_area → gate → per_class_conf
"""
import os, time, logging
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

MODEL = "/home/lay/hoban/hoban_go3k_v23/weights/best.pt"
OUT_DIR = "/home/lay/hoban/analysis/v23_vis"
os.makedirs(OUT_DIR, exist_ok=True)

# 평가셋 (729장)
EVAL_SETS = [
    ("/home/lay/hoban/datasets/3k_finetune/val/images",
     "/home/lay/hoban/datasets/3k_finetune/val/labels"),
    ("/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images",
     "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"),
]

# 후처리 파라미터 (최적)
CROSS_NMS_IOU = 0.3
MIN_AREA = 5e-05
GATE_CONF = 0.20
GATE_RADIUS = 30
CONF_ON = 0.45   # v23 eval best: c0=0.45
CONF_OFF = 0.10  # v23 eval best: c1=0.10

COLORS = {0: (0, 200, 0), 1: (255, 40, 40)}       # pred
GT_COLORS = {0: (0, 255, 200), 1: (255, 180, 0)}   # GT (cyan, orange)
LABELS = {0: "ON", 1: "OFF"}

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except:
    font = ImageFont.load_default()
    font_sm = font


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def cross_class_nms(preds, iou_thresh):
    if len(preds) <= 1:
        return preds
    sorted_preds = sorted(preds, key=lambda x: -x[1])
    keep, suppressed = [], set()
    for i, (c1, s1, x1a, y1a, x2a, y2a) in enumerate(sorted_preds):
        if i in suppressed:
            continue
        keep.append(sorted_preds[i])
        for j in range(i+1, len(sorted_preds)):
            if j in suppressed:
                continue
            c2 = sorted_preds[j][0]
            if c1 != c2:
                iou = compute_iou((x1a, y1a, x2a, y2a),
                                  (sorted_preds[j][2], sorted_preds[j][3],
                                   sorted_preds[j][4], sorted_preds[j][5]))
                if iou >= iou_thresh:
                    suppressed.add(j)
    return keep


def apply_pipeline(preds, full_raw, img_w, img_h):
    filtered = cross_class_nms(preds, CROSS_NMS_IOU)
    img_area = img_w * img_h
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if ((x2-x1)*(y2-y1)) / img_area >= MIN_AREA]
    gates = [(x1,y1,x2,y2) for conf,x1,y1,x2,y2 in full_raw if conf >= GATE_CONF]
    if gates:
        gated = []
        for c,s,x1,y1,x2,y2 in filtered:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            for gx1,gy1,gx2,gy2 in gates:
                gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
                if abs(cx-gcx) <= GATE_RADIUS and abs(cy-gcy) <= GATE_RADIUS:
                    gated.append((c,s,x1,y1,x2,y2))
                    break
        filtered = gated
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if s >= (CONF_ON if c == 0 else CONF_OFF)]
    return filtered


def load_gt(lbl_path, img_w, img_h):
    """YOLO format GT 로드 → (cls, x1, y1, x2, y2)"""
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
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            gts.append((cls, x1, y1, x2, y2))
    return gts


def draw_dashed_rect(draw, bbox, color, dash_len=8, gap_len=4, width=2):
    """점선 사각형"""
    x1, y1, x2, y2 = bbox
    for side in [
        [(x1, y1), (x2, y1)],  # top
        [(x2, y1), (x2, y2)],  # right
        [(x2, y2), (x1, y2)],  # bottom
        [(x1, y2), (x1, y1)],  # left
    ]:
        sx, sy = side[0]
        ex, ey = side[1]
        dx = ex - sx
        dy = ey - sy
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            continue
        ux, uy = dx/length, dy/length
        pos = 0
        while pos < length:
            seg_end = min(pos + dash_len, length)
            draw.line([(sx + ux*pos, sy + uy*pos),
                       (sx + ux*seg_end, sy + uy*seg_end)],
                      fill=color, width=width)
            pos = seg_end + gap_len


# 평가셋 이미지 수집
eval_images = []
for img_dir, lbl_dir in EVAL_SETS:
    if not os.path.isdir(img_dir):
        print(f"경고: {img_dir} 없음")
        continue
    for fname in sorted(os.listdir(img_dir)):
        if fname.endswith(".jpg"):
            eval_images.append((
                os.path.join(img_dir, fname),
                os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
            ))

print(f"평가셋: {len(eval_images)}장")
print(f"모델: {MODEL}")
print(f"출력: {OUT_DIR}")
print(f"후처리: cross_nms({CROSS_NMS_IOU}) + area(>{MIN_AREA}) + gate({GATE_CONF},{GATE_RADIUS}) + per_class(ON≥{CONF_ON},OFF≥{CONF_OFF})")

# 모델 로드
print("\nSAHI 모델 로드...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8", model_path=MODEL,
    confidence_threshold=0.05, device="0", image_size=1280)

print("Full-image 모델 로드 (Gate용)...")
yolo_model = YOLO(MODEL)

t0 = time.time()
tp, fp, fn = 0, 0, 0
saved = 0

for i, (img_path, lbl_path) in enumerate(eval_images):
    if i % 50 == 0:
        print(f"  {i}/{len(eval_images)}...", end="\r")

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

    # Full-image 추론 (Gate)
    results = yolo_model.predict(img_path, conf=0.01, imgsz=1280,
                                 device="0", verbose=False)
    boxes = results[0].boxes
    full_raw = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                for j in range(len(boxes))]

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    # 후처리 적용
    preds = apply_pipeline(raw_preds, full_raw, img_w, img_h)

    # GT 로드
    gts = load_gt(lbl_path, img_w, img_h)

    # 탐지가 없고 GT도 없으면 스킵
    if not preds and not gts:
        continue

    draw = ImageDraw.Draw(img)

    # GT 그리기 (점선)
    for cls_id, x1, y1, x2, y2 in gts:
        color = GT_COLORS[cls_id]
        draw_dashed_rect(draw, (x1, y1, x2, y2), color, width=2)
        label = f"GT:{LABELS[cls_id]}"
        bbox = draw.textbbox((0, 0), label, font=font_sm)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        ty = y2 + 2
        if ty + th > img_h:
            ty = y1 - th - 4
        draw.rectangle([x1, ty-1, x1+tw+4, ty+th+2], fill=color)
        draw.text((x1+2, ty), label, fill="black", font=font_sm)

    # Pred 그리기 (실선, ON 먼저 OFF 나중)
    preds.sort(key=lambda x: x[0])
    for cls_id, conf, x1, y1, x2, y2 in preds:
        color = COLORS[cls_id]
        lw = 3 if cls_id == 1 else 2
        for offset in range(lw):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset], outline=color)
        label = f"{LABELS[cls_id]} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        ty = y1 - th - 6
        if ty < 0:
            ty = y1 + 2
        bg = (0, 140, 0) if cls_id == 0 else (180, 0, 0)
        draw.rectangle([x1-1, ty-1, x1+tw+4, ty+th+3], fill=bg)
        draw.text((x1+2, ty), label, fill="white", font=font)

    fname = os.path.basename(img_path)
    img.save(os.path.join(OUT_DIR, fname), quality=90)
    saved += 1

elapsed = time.time() - t0
print(f"\n완료: {saved}/{len(eval_images)}장 시각화 ({elapsed:.0f}s)")
print(f"출력: {OUT_DIR}")
print(f"\n범례: 실선=Pred, 점선=GT")
print(f"  초록 실선: Pred helmet_on  |  빨강 실선: Pred helmet_off")
print(f"  시안 점선: GT helmet_on    |  주황 점선: GT helmet_off")
