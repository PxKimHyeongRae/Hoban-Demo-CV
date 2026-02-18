#!/usr/bin/env python3
"""실서비스 파이프라인 전체 레벨 비교 평가

SLICE_LEVEL 2/4/6/8 + full-image 조합 + SAHI 비교.
속도와 정확도의 최적 조합 탐색.

사용법:
  python eval_pipeline_compare.py              # 로컬 (Z:/)
  python eval_pipeline_compare.py --server     # 서버 (/)
"""
import os, sys, time, argparse, logging
import numpy as np
import cv2
from collections import defaultdict
from PIL import Image

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}
PC = {0: 0.40, 1: 0.15}  # per-class confidence

# 테스트 설정
SLICE_CONFIGS = {
    2: (1280, 0.1),   # 1280x1280, overlap 0.1 → 2 slices (현재 실서비스)
    4: (1024, 0.1),   # 1024x1024, overlap 0.1 → 4 slices
    6: (640, 0.05),   #  640x640,  overlap 0.05 → 6 slices
    8: (640, 0.2),    #  640x640,  overlap 0.2  → 8 slices
}


# ============================================================================
#  GT / 평가 유틸
# ============================================================================

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


def evaluate(all_gt, all_preds, image_set):
    tp = fp = fn = 0
    ctp, cfp, cfn = defaultdict(int), defaultdict(int), defaultdict(int)
    for fname in image_set:
        gts = all_gt.get(fname, [])
        preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in all_preds.get(fname, [])
                 if s >= PC.get(c, 0.5)]
        matched = set()
        for _, (pc, ps, px1, py1, px2, py2) in sorted(enumerate(preds), key=lambda x: -x[1][1]):
            bi, bv = -1, 0
            for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
                if gi in matched or gc != pc:
                    continue
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > bv:
                    bv, bi = iou, gi
            if bv >= 0.5 and bi >= 0:
                tp += 1; ctp[gts[bi][0]] += 1; matched.add(bi)
            else:
                fp += 1; cfp[pc] += 1
        for gi in range(len(gts)):
            if gi not in matched:
                fn += 1; cfn[gts[gi][0]] += 1
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn


# ============================================================================
#  배치 슬라이스 추론 (video_indoor 방식)
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


def batch_sliced_predict(frame, yolo_model, slice_size=1280, overlap=0.1,
                          conf=0.15, device="cuda:0"):
    """video_indoor _batch_sliced_predict 재현"""
    import torch

    img_h, img_w = frame.shape[:2]
    slices = _calc_slices(img_h, img_w, slice_size, slice_size, overlap, overlap)

    batch_list = []
    metas = []
    for (sx, sy, ex, ey) in slices:
        crop = frame[sy:ey, sx:ex]
        lb, scale, dw, dh = _letterbox(crop, slice_size)
        t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        batch_list.append(t)
        metas.append((sx, sy, ex - sx, ey - sy, scale, dw, dh))

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

    all_dets = _cross_slice_nms(all_dets, iou_threshold=0.5)
    return all_dets


def full_image_predict(frame, yolo_model, conf=0.15, device="cuda:0"):
    """전체 이미지 추론 (1280px letterbox)"""
    import torch

    img_h, img_w = frame.shape[:2]
    lb, scale, dw, dh = _letterbox(frame, 1280)
    t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    batch_tensor = torch.from_numpy(t[np.newaxis])
    if "cuda" in device:
        batch_tensor = batch_tensor.half().cuda()

    with torch.no_grad():
        raw_preds = yolo_model.model(batch_tensor)

    preds_tensor = raw_preds[0][0]
    valid = preds_tensor[preds_tensor[:, 4] >= conf]

    dets = []
    for det in valid:
        x1, y1, x2, y2 = det[:4].cpu().numpy()
        conf_val = float(det[4])
        cls_id = int(det[5])

        x1 = (x1 - dw) / scale
        y1 = (y1 - dh) / scale
        x2 = (x2 - dw) / scale
        y2 = (y2 - dh) / scale

        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 > x1 and y2 > y1:
            dets.append((cls_id, conf_val, x1, y1, x2, y2))
    return dets


def run_pipeline(combined, img_paths, yolo_model, slice_size, overlap,
                  add_full=False, conf=0.15, device="cuda:0"):
    """특정 설정으로 전체 이미지 세트 추론"""
    all_preds = {}
    for i, f in enumerate(combined):
        if i % 100 == 0:
            print(f"    {i}/{len(combined)}...", end="\r")
        frame = cv2.imread(img_paths[f])
        if frame is None:
            all_preds[f] = []
            continue

        dets = batch_sliced_predict(frame, yolo_model, slice_size, overlap,
                                     conf, device)
        if add_full:
            full_dets = full_image_predict(frame, yolo_model, conf, device)
            dets = dets + full_dets
            dets = _cross_slice_nms(dets, iou_threshold=0.5)

        dets = _cross_class_nms(dets, 0.3)
        all_preds[f] = dets

    return all_preds


# ============================================================================
#  SAHI
# ============================================================================

def run_sahi(combined, img_paths, model_path, device):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.05, device=device, image_size=1280)

    all_preds = {}
    for i, f in enumerate(combined):
        if i % 50 == 0:
            print(f"    {i}/{len(combined)}...", end="\r")
        result = get_sliced_prediction(
            img_paths[f], sahi_model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", verbose=0)
        preds = [(p.category.id, p.score.value,
                  p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                 for p in result.object_prediction_list]
        preds = _cross_class_nms(preds, 0.3)
        all_preds[f] = preds
    return all_preds


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-sahi", action="store_true", help="SAHI 생략 (슬라이스만)")
    args = parser.parse_args()

    BASE = "/" if args.server else "Z:/"
    VAL_IMG = os.path.join(BASE, "home/lay/hoban/datasets/3k_finetune/val/images")
    VAL_LBL = os.path.join(BASE, "home/lay/hoban/datasets/3k_finetune/val/labels")
    EXTRA_IMG = os.path.join(BASE, "home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images")
    EXTRA_LBL = os.path.join(BASE, "home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels")
    MODEL_PATH = args.model or os.path.join(BASE, "home/lay/hoban/hoban_go3k_v17/weights/best.pt")

    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Device: {args.device}")
    print(f"Model: {MODEL_PATH}")

    # ── GT 로드 ──
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
    gt_total = sum(len(all_gt[f]) for f in combined)
    gt_off = sum(1 for f in combined for g in all_gt[f] if g[0] == 1)
    print(f"평가 세트: {len(combined)}장 ({len(val_imgs)} val + {len(extra_imgs)} extra)")
    print(f"GT bbox: {gt_total} (helmet_on: {gt_total-gt_off}, helmet_off: {gt_off})")

    # ── 모델 로드 (1회) ──
    from ultralytics import YOLO
    yolo_model = YOLO(MODEL_PATH)
    yolo_model.fuse()
    if "cuda" in args.device:
        yolo_model.model.to(args.device)
        yolo_model.model.half()
    print(f"모델 로드 완료 (FP16 + fuse)\n")

    # ── 실험 설정 ──
    experiments = []
    for level, (sz, ovl) in sorted(SLICE_CONFIGS.items()):
        n_slices = len(_calc_slices(1080, 1920, sz, sz, ovl, ovl))
        experiments.append({
            'name': f'L{level} ({sz}px, {n_slices}tiles)',
            'slice_size': sz, 'overlap': ovl, 'full': False
        })
        experiments.append({
            'name': f'L{level}+full ({sz}px, {n_slices}t+full)',
            'slice_size': sz, 'overlap': ovl, 'full': True
        })

    # ── 실행 ──
    results = []

    for exp in experiments:
        print(f"{'='*70}")
        print(f"  {exp['name']}")
        print(f"{'='*70}")
        t0 = time.time()
        preds = run_pipeline(combined, img_paths, yolo_model,
                              exp['slice_size'], exp['overlap'],
                              add_full=exp['full'], conf=0.15,
                              device=args.device)
        elapsed = time.time() - t0
        speed = len(combined) / elapsed
        total_dets = sum(len(v) for v in preds.values())

        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, preds, combined)

        # per-class
        on_p = ctp[0]/(ctp[0]+cfp[0]) if ctp[0]+cfp[0] else 0
        on_r = ctp[0]/(ctp[0]+cfn[0]) if ctp[0]+cfn[0] else 0
        on_f1 = 2*on_p*on_r/(on_p+on_r) if on_p+on_r else 0
        off_p = ctp[1]/(ctp[1]+cfp[1]) if ctp[1]+cfp[1] else 0
        off_r = ctp[1]/(ctp[1]+cfn[1]) if ctp[1]+cfn[1] else 0
        off_f1 = 2*off_p*off_r/(off_p+off_r) if off_p+off_r else 0

        print(f"  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  "
              f"(TP={tp} FP={fp} FN={fn})  "
              f"[{total_dets} dets, {speed:.1f} img/s, {elapsed:.0f}s]")
        print(f"    ON:  P={on_p:.3f} R={on_r:.3f} F1={on_f1:.3f}")
        print(f"    OFF: P={off_p:.3f} R={off_r:.3f} F1={off_f1:.3f}")

        results.append({
            'name': exp['name'], 'f1': f1, 'p': p, 'r': r,
            'tp': tp, 'fp': fp, 'fn': fn,
            'on_f1': on_f1, 'on_r': on_r,
            'off_f1': off_f1, 'off_r': off_r,
            'dets': total_dets, 'speed': speed, 'time': elapsed
        })

    # ── SAHI ──
    if not args.skip_sahi:
        print(f"\n{'='*70}")
        print(f"  SAHI (1280x720, overlap=0.15, +full-image)")
        print(f"{'='*70}")
        t0 = time.time()
        preds = run_sahi(combined, img_paths, MODEL_PATH, args.device)
        elapsed = time.time() - t0
        speed = len(combined) / elapsed
        total_dets = sum(len(v) for v in preds.values())

        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(all_gt, preds, combined)
        on_p = ctp[0]/(ctp[0]+cfp[0]) if ctp[0]+cfp[0] else 0
        on_r = ctp[0]/(ctp[0]+cfn[0]) if ctp[0]+cfn[0] else 0
        on_f1 = 2*on_p*on_r/(on_p+on_r) if on_p+on_r else 0
        off_p = ctp[1]/(ctp[1]+cfp[1]) if ctp[1]+cfp[1] else 0
        off_r = ctp[1]/(ctp[1]+cfn[1]) if ctp[1]+cfn[1] else 0
        off_f1 = 2*off_p*off_r/(off_p+off_r) if off_p+off_r else 0

        print(f"  F1={f1:.3f}  P={p:.3f}  R={r:.3f}  "
              f"(TP={tp} FP={fp} FN={fn})  "
              f"[{total_dets} dets, {speed:.1f} img/s, {elapsed:.0f}s]")
        print(f"    ON:  P={on_p:.3f} R={on_r:.3f} F1={on_f1:.3f}")
        print(f"    OFF: P={off_p:.3f} R={off_r:.3f} F1={off_f1:.3f}")

        results.append({
            'name': 'SAHI', 'f1': f1, 'p': p, 'r': r,
            'tp': tp, 'fp': fp, 'fn': fn,
            'on_f1': on_f1, 'on_r': on_r,
            'off_f1': off_f1, 'off_r': off_r,
            'dets': total_dets, 'speed': speed, 'time': elapsed
        })

    # ── 요약 테이블 ──
    print(f"\n\n{'='*90}")
    print(f"  요약 (per-class conf: ON≥0.40, OFF≥0.15)")
    print(f"{'='*90}")
    print(f"  {'Config':<30} {'F1':>5} {'P':>5} {'R':>5} "
          f"{'ON_F1':>6} {'OFF_F1':>6} {'OFF_R':>5} "
          f"{'FP':>4} {'Dets':>5} {'Speed':>7}")
    print(f"  {'-'*85}")

    best_f1 = max(r['f1'] for r in results)
    for r in results:
        marker = " *" if r['f1'] == best_f1 else "  "
        print(f"  {r['name']:<30} {r['f1']:.3f} {r['p']:.3f} {r['r']:.3f} "
              f"{r['on_f1']:.3f}  {r['off_f1']:.3f} {r['off_r']:.3f} "
              f"{r['fp']:>4} {r['dets']:>5} {r['speed']:>5.1f}/s{marker}")

    print(f"\n  * = 최고 F1")
    print(f"  현재 실서비스: L2 (1280px, 2tiles)")
    print(f"  v17 SAHI 기준: F1=0.918")
    print(f"{'='*90}")
