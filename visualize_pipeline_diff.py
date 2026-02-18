#!/usr/bin/env python3
"""실서비스 vs SAHI 탐지 차이 시각화

각 이미지에 3패널 표시:
  [GT (정답)] | [Pipeline A: 실서비스] | [Pipeline B: SAHI]

오탐(FP)은 빨간색, 정탐(TP)은 초록색, 미탐(FN)은 노란색 점선.
차이가 큰 이미지만 자동 선별하여 출력.

사용법:
  python visualize_pipeline_diff.py              # 로컬 (Z:/)
  python visualize_pipeline_diff.py --server     # 서버
  python visualize_pipeline_diff.py --max 30     # 최대 30장
"""
import os, sys, time, argparse, logging
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}
PC = {0: 0.40, 1: 0.15}

# 색상 (BGR)
COLOR_TP = (0, 200, 0)      # 초록 = 정탐
COLOR_FP = (0, 0, 255)      # 빨강 = 오탐
COLOR_FN = (0, 200, 255)    # 노랑 = 미탐 (GT에 있지만 탐지 못함)
COLOR_GT = (255, 180, 0)    # 파랑계열 = GT


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


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


def match_preds_to_gt(preds, gts):
    """탐지를 GT와 매칭 → TP/FP/FN 분류 반환"""
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in preds
                if s >= PC.get(c, 0.5)]
    filtered.sort(key=lambda x: -x[1])

    matched_gt = set()
    tp_preds = []  # (cls, conf, x1,y1,x2,y2)
    fp_preds = []

    for pc, ps, px1, py1, px2, py2 in filtered:
        best_iou, best_gi = 0, -1
        for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
            if gi in matched_gt or gc != pc:
                continue
            iou = compute_iou((px1,py1,px2,py2), (gx1,gy1,gx2,gy2))
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= 0.5 and best_gi >= 0:
            tp_preds.append((pc, ps, px1, py1, px2, py2))
            matched_gt.add(best_gi)
        else:
            fp_preds.append((pc, ps, px1, py1, px2, py2))

    fn_gts = [(gts[gi][0], gts[gi][1], gts[gi][2], gts[gi][3], gts[gi][4])
              for gi in range(len(gts)) if gi not in matched_gt]

    return tp_preds, fp_preds, fn_gts


def draw_panel(img, title, tp_preds, fp_preds, fn_gts):
    """이미지에 bbox 그리기"""
    panel = img.copy()

    # FN (미탐 - 노란 점선)
    for gc, gx1, gy1, gx2, gy2 in fn_gts:
        x1, y1, x2, y2 = int(gx1), int(gy1), int(gx2), int(gy2)
        # 점선 효과
        for i in range(x1, x2, 8):
            cv2.line(panel, (i, y1), (min(i+4, x2), y1), COLOR_FN, 2)
            cv2.line(panel, (i, y2), (min(i+4, x2), y2), COLOR_FN, 2)
        for i in range(y1, y2, 8):
            cv2.line(panel, (x1, i), (x1, min(i+4, y2)), COLOR_FN, 2)
            cv2.line(panel, (x2, i), (x2, min(i+4, y2)), COLOR_FN, 2)
        label = f"FN:{CLASS_NAMES[gc]}"
        cv2.putText(panel, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLOR_FN, 1)

    # TP (정탐 - 초록)
    for pc, ps, px1, py1, px2, py2 in tp_preds:
        x1, y1, x2, y2 = int(px1), int(py1), int(px2), int(py2)
        cv2.rectangle(panel, (x1, y1), (x2, y2), COLOR_TP, 2)
        label = f"{CLASS_NAMES[pc]} {ps:.2f}"
        cv2.putText(panel, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLOR_TP, 1)

    # FP (오탐 - 빨강)
    for pc, ps, px1, py1, px2, py2 in fp_preds:
        x1, y1, x2, y2 = int(px1), int(py1), int(px2), int(py2)
        cv2.rectangle(panel, (x1, y1), (x2, y2), COLOR_FP, 2)
        label = f"FP:{CLASS_NAMES[pc]} {ps:.2f}"
        cv2.putText(panel, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLOR_FP, 1)

    # 타이틀
    cv2.rectangle(panel, (0, 0), (len(title)*10+10, 22), (0,0,0), -1)
    cv2.putText(panel, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return panel


def draw_gt_panel(img, gts):
    """GT 패널"""
    panel = img.copy()
    for gc, gx1, gy1, gx2, gy2 in gts:
        x1, y1, x2, y2 = int(gx1), int(gy1), int(gx2), int(gy2)
        cv2.rectangle(panel, (x1, y1), (x2, y2), COLOR_GT, 2)
        label = CLASS_NAMES[gc]
        cv2.putText(panel, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, COLOR_GT, 1)
    cv2.rectangle(panel, (0, 0), (50, 22), (0,0,0), -1)
    cv2.putText(panel, "GT", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return panel


# ============================================================================
#  배치 슬라이스 추론 (video_indoor 재현)
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
                    if compute_iou(best[2:], d[2:]) < iou_threshold]
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
                iou = compute_iou(sorted_dets[i][2:], sorted_dets[j][2:])
                if iou >= iou_threshold:
                    suppressed.add(j)
    return keep


def batch_sliced_predict(frame, yolo_model, slice_size, overlap, conf, device):
    import torch
    img_h, img_w = frame.shape[:2]
    slices = _calc_slices(img_h, img_w, slice_size, slice_size, overlap, overlap)

    batch_list, metas = [], []
    for (sx, sy, ex, ey) in slices:
        crop = frame[sy:ey, sx:ex]
        lb, scale, dw, dh = _letterbox(crop, slice_size)
        t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        batch_list.append(t)
        metas.append((sx, sy, ex-sx, ey-sy, scale, dw, dh))

    batch_np = np.ascontiguousarray(np.stack(batch_list))
    batch_tensor = torch.from_numpy(batch_np)
    if "cuda" in device:
        batch_tensor = batch_tensor.half().cuda()

    with torch.no_grad():
        raw_preds = yolo_model.model(batch_tensor)
    preds_tensor = raw_preds[0]

    all_dets = []
    for i, (sx, sy, sw, sh, scale, dw, dh) in enumerate(metas):
        preds = preds_tensor[i]
        valid = preds[preds[:, 4] >= conf]
        if len(valid) == 0:
            continue
        for det in valid:
            x1, y1, x2, y2 = det[:4].cpu().numpy()
            conf_val = float(det[4])
            cls_id = int(det[5])
            x1 = (x1 - dw) / scale + sx
            y1 = (y1 - dh) / scale + sy
            x2 = (x2 - dw) / scale + sx
            y2 = (y2 - dh) / scale + sy
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            if x2 > x1 and y2 > y1:
                all_dets.append((cls_id, conf_val, x1, y1, x2, y2))

    all_dets = _cross_slice_nms(all_dets, 0.5)
    all_dets = _cross_class_nms(all_dets, 0.3)
    return all_dets


def sahi_predict(img_path, sahi_model):
    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(
        img_path, sahi_model,
        slice_height=720, slice_width=1280,
        overlap_height_ratio=0.15, overlap_width_ratio=0.15,
        perform_standard_pred=True,
        postprocess_type="NMS", postprocess_match_threshold=0.4,
        postprocess_match_metric="IOS", verbose=0)
    preds = [(p.category.id, p.score.value,
              p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
             for p in result.object_prediction_list]
    return _cross_class_nms(preds, 0.3)


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max", type=int, default=50, help="최대 출력 이미지 수")
    parser.add_argument("--out", default=None, help="출력 디렉터리")
    args = parser.parse_args()

    BASE = "/" if args.server else "Z:/"
    VAL_IMG = os.path.join(BASE, "home/lay/hoban/datasets/3k_finetune/val/images")
    VAL_LBL = os.path.join(BASE, "home/lay/hoban/datasets/3k_finetune/val/labels")
    EXTRA_IMG = os.path.join(BASE, "home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images")
    EXTRA_LBL = os.path.join(BASE, "home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels")
    MODEL_PATH = args.model or os.path.join(BASE, "home/lay/hoban/hoban_go3k_v17/weights/best.pt")
    OUT_DIR = args.out or os.path.join(BASE, "home/lay/hoban/pipeline_diff_viz")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUT_DIR}")

    # GT 로드
    val_imgs = sorted(f for f in os.listdir(VAL_IMG) if f.endswith(".jpg"))
    all_gt, img_paths = {}, {}
    for f in val_imgs:
        path = os.path.join(VAL_IMG, f)
        img = Image.open(path)
        img_paths[f] = path
        all_gt[f] = load_gt(os.path.join(VAL_LBL, f.replace(".jpg", ".txt")), *img.size)

    extra_imgs = []
    if os.path.isdir(EXTRA_IMG):
        for f in sorted(os.listdir(EXTRA_IMG)):
            if f.endswith(".jpg") and f not in all_gt:
                path = os.path.join(EXTRA_IMG, f)
                img = Image.open(path)
                img_paths[f] = path
                all_gt[f] = load_gt(os.path.join(EXTRA_LBL, f.replace(".jpg", ".txt")), *img.size)
                extra_imgs.append(f)

    combined = val_imgs + extra_imgs
    print(f"평가: {len(combined)}장")

    # 모델 로드
    from ultralytics import YOLO
    yolo_model = YOLO(MODEL_PATH)
    yolo_model.fuse()
    if "cuda" in args.device:
        yolo_model.model.to(args.device)
        yolo_model.model.half()

    from sahi import AutoDetectionModel
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=MODEL_PATH,
        confidence_threshold=0.05, device=args.device, image_size=1280)

    # 추론 + 차이 분석
    print(f"\n추론 중...")
    diff_images = []  # (fname, score, a_fp, a_fn, b_fp, b_fn)

    for i, f in enumerate(combined):
        if i % 50 == 0:
            print(f"  {i}/{len(combined)}...", end="\r")

        frame = cv2.imread(img_paths[f])
        if frame is None:
            continue
        gts = all_gt.get(f, [])

        # Pipeline A: 실서비스 (L2: 1280, overlap=0.1)
        preds_a = batch_sliced_predict(frame, yolo_model, 1280, 0.1, 0.15, args.device)
        # Pipeline B: SAHI
        preds_b = sahi_predict(img_paths[f], sahi_model)

        # 매칭
        tp_a, fp_a, fn_a = match_preds_to_gt(preds_a, gts)
        tp_b, fp_b, fn_b = match_preds_to_gt(preds_b, gts)

        # 차이가 있는 이미지만 수집
        # 점수 = A의 FP수 + A의 FN수 - B의 FN수 (A가 놓치는데 B는 잡는 것 우선)
        diff_score = len(fp_a) + len(fn_a) + abs(len(fn_a) - len(fn_b))
        if diff_score > 0 or len(fp_a) != len(fp_b) or len(fn_a) != len(fn_b):
            diff_images.append((f, diff_score, tp_a, fp_a, fn_a, tp_b, fp_b, fn_b))

    print(f"\n차이 있는 이미지: {len(diff_images)}장")

    # 차이 큰 순서로 정렬
    diff_images.sort(key=lambda x: -x[1])
    to_save = diff_images[:args.max]

    # 시각화 저장
    print(f"시각화 저장: {len(to_save)}장 → {OUT_DIR}")
    for idx, (f, score, tp_a, fp_a, fn_a, tp_b, fp_b, fn_b) in enumerate(to_save):
        frame = cv2.imread(img_paths[f])
        gts = all_gt.get(f, [])

        # 3패널: GT | Pipeline A | Pipeline B
        p_gt = draw_gt_panel(frame, gts)
        p_a = draw_panel(frame,
                          f"A:Production (TP={len(tp_a)} FP={len(fp_a)} FN={len(fn_a)})",
                          tp_a, fp_a, fn_a)
        p_b = draw_panel(frame,
                          f"B:SAHI (TP={len(tp_b)} FP={len(fp_b)} FN={len(fn_b)})",
                          tp_b, fp_b, fn_b)

        # 가로 연결
        combined_img = np.hstack([p_gt, p_a, p_b])

        # 파일명에 정보 포함
        out_name = f"{idx:03d}_fp{len(fp_a)}_fn{len(fn_a)}_{f}"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), combined_img)

    # 요약
    total_a_fp = sum(len(x[3]) for x in diff_images)
    total_a_fn = sum(len(x[4]) for x in diff_images)
    total_b_fp = sum(len(x[6]) for x in diff_images)
    total_b_fn = sum(len(x[7]) for x in diff_images)

    print(f"\n{'='*60}")
    print(f"  차이 요약 ({len(diff_images)}장)")
    print(f"{'='*60}")
    print(f"  Pipeline A (실서비스): FP={total_a_fp}, FN={total_a_fn}")
    print(f"  Pipeline B (SAHI):    FP={total_b_fp}, FN={total_b_fn}")
    print(f"\n  범례:")
    print(f"    초록 실선 = TP (정탐)")
    print(f"    빨강 실선 = FP (오탐)")
    print(f"    노랑 점선 = FN (미탐)")
    print(f"    파랑 실선 = GT (정답)")
    print(f"\n  출력: {OUT_DIR}")
    print(f"{'='*60}")
