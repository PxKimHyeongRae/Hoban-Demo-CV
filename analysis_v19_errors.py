#!/usr/bin/env python3
"""v19 FP/FN 오류 분석 + 시각화

SAHI 평가 후 모든 FP/FN을 시각화하고 유형별 통계 생성.
출력: analysis_v19_errors/ 디렉터리
"""
import os, sys, time, logging, json
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)
logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ── 설정 ──
MODEL_PATH = "/home/lay/hoban/hoban_go3k_v19/weights/best.pt"
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
OUT_DIR = "/home/lay/hoban/analysis_v19_errors"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}

# 후처리 파이프라인 (v19 최적)
PIPELINE_CONF = {"c0": 0.40, "c1": 0.15}
CROSS_NMS_IOU = 0.3
MIN_AREA = 5e-05
GATE_CONF = 0.20
GATE_RADIUS = 30
IOU_MATCH = 0.5


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


def apply_pipeline(sahi_dets, full_dets, img_w, img_h):
    """후처리 파이프라인 적용"""
    # 1. cross-class NMS
    dets = cross_class_nms(sahi_dets, CROSS_NMS_IOU)

    # 2. min area
    img_area = img_w * img_h
    dets = [d for d in dets if (d[4] - d[2]) * (d[5] - d[3]) / img_area >= MIN_AREA]

    # 3. gate (SAHI only → full-image에서도 보이는지)
    gated = []
    for d in dets:
        cx = (d[2] + d[4]) / 2
        cy = (d[3] + d[5]) / 2
        found = False
        for fd in full_dets:
            fcx = (fd[2] + fd[4]) / 2
            fcy = (fd[3] + fd[5]) / 2
            if abs(cx - fcx) < GATE_RADIUS and abs(cy - fcy) < GATE_RADIUS:
                if fd[1] >= GATE_CONF:
                    found = True
                    break
        if found:
            gated.append(d)
    dets = gated

    # 4. per-class conf
    dets = [d for d in dets if d[1] >= (PIPELINE_CONF["c0"] if d[0] == 0 else PIPELINE_CONF["c1"])]
    return dets


def load_gt(lbl_path, img_w, img_h):
    """YOLO 라벨 → (cls, x1, y1, x2, y2)"""
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


def bbox_size_category(x1, y1, x2, y2, img_w, img_h):
    area_ratio = (x2 - x1) * (y2 - y1) / (img_w * img_h)
    if area_ratio < 0.001:
        return "tiny(<0.1%)"
    elif area_ratio < 0.005:
        return "small(0.1-0.5%)"
    elif area_ratio < 0.02:
        return "medium(0.5-2%)"
    else:
        return "large(>2%)"


def draw_results(img, gts, preds, fp_list, fn_list, fname):
    """FP=빨강, FN=파랑, TP=초록으로 시각화"""
    vis = img.copy()
    h, w = vis.shape[:2]

    # TP (매칭된 GT) - 초록
    matched_gt = set()
    matched_pred = set()
    for pi, pred in enumerate(preds):
        best_iou, best_gi = 0, -1
        for gi, gt in enumerate(gts):
            if gt[0] != pred[0] or gi in matched_gt:
                continue
            iou = compute_iou(pred[2:], gt[1:])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= IOU_MATCH and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            # TP - 초록 (얇게)
            x1, y1, x2, y2 = [int(v) for v in pred[2:]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 1)

    # FP - 빨강 (굵게)
    for pi in range(len(preds)):
        if pi not in matched_pred:
            pred = preds[pi]
            x1, y1, x2, y2 = [int(v) for v in pred[2:]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"FP:{CLASS_NAMES[pred[0]]} {pred[1]:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # FN - 파랑 (굵게)
    for gi in range(len(gts)):
        if gi not in matched_gt:
            gt = gts[gi]
            x1, y1, x2, y2 = [int(v) for v in gt[1:]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"FN:{CLASS_NAMES[gt[0]]}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return vis


def main():
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO

    os.makedirs(f"{OUT_DIR}/fp_images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/fn_images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/all_errors", exist_ok=True)

    # 평가 이미지 수집
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

    # SAHI 모델
    det_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=0.05,
        device="cuda:0")

    # Full-image 모델 (gate용)
    full_model = YOLO(MODEL_PATH)
    full_model.fuse()
    full_model.model.to("cuda:0")
    full_model.model.half()

    # 통계 수집
    stats = {
        "total_gt": 0, "total_pred": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "fp_by_class": defaultdict(int),
        "fn_by_class": defaultdict(int),
        "fp_by_size": defaultdict(int),
        "fn_by_size": defaultdict(int),
        "fp_images": [],  # (fname, fp_count, fp_details)
        "fn_images": [],  # (fname, fn_count, fn_details)
    }

    import torch
    from extract_data_v17 import _letterbox, _parse_preds

    print("추론 시작...")
    for idx, (img_path, lbl_path, fname) in enumerate(images):
        if idx % 50 == 0:
            print(f"  {idx}/{len(images)}...", flush=True)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # GT
        gts = load_gt(lbl_path, img_w, img_h)
        stats["total_gt"] += len(gts)

        # SAHI 추론
        pil_img = Image.open(img_path)
        result = get_sliced_prediction(
            pil_img, det_model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5)

        sahi_dets = []
        for pred in result.object_prediction_list:
            bb = pred.bbox
            sahi_dets.append((pred.category.id, pred.score.value,
                            bb.minx, bb.miny, bb.maxx, bb.maxy))

        # Full-image 추론 (gate용)
        lb, scale, dw, dh = _letterbox(img, 1280)
        t = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(t[None]).half().to("cuda:0")
        with torch.no_grad():
            preds_raw = full_model.model(tensor)[0]
        full_dets = _parse_preds(preds_raw[0], 0.01, img_h, img_w, scale, dw, dh)

        # 후처리
        final_preds = apply_pipeline(sahi_dets, full_dets, img_w, img_h)
        stats["total_pred"] += len(final_preds)

        # 매칭 (TP/FP/FN)
        matched_gt = set()
        matched_pred = set()
        fp_details, fn_details = [], []

        for pi, pred in enumerate(final_preds):
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gts):
                if gt[0] != pred[0] or gi in matched_gt:
                    continue
                iou = compute_iou(pred[2:], gt[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= IOU_MATCH and best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                stats["tp"] += 1
            else:
                stats["fp"] += 1
                stats["fp_by_class"][pred[0]] += 1
                size = bbox_size_category(pred[2], pred[3], pred[4], pred[5], img_w, img_h)
                stats["fp_by_size"][size] += 1
                fp_details.append({
                    "class": CLASS_NAMES[pred[0]], "conf": pred[1],
                    "bbox": [pred[2], pred[3], pred[4], pred[5]], "size": size
                })

        for gi, gt in enumerate(gts):
            if gi not in matched_gt:
                stats["fn"] += 1
                stats["fn_by_class"][gt[0]] += 1
                size = bbox_size_category(gt[1], gt[2], gt[3], gt[4], img_w, img_h)
                stats["fn_by_size"][size] += 1
                fn_details.append({
                    "class": CLASS_NAMES[gt[0]],
                    "bbox": [gt[1], gt[2], gt[3], gt[4]], "size": size
                })

        # 에러 있는 이미지 시각화
        if fp_details or fn_details:
            vis = draw_results(img, gts, final_preds, fp_details, fn_details, fname)
            cv2.imwrite(f"{OUT_DIR}/all_errors/{fname}", vis)

        if fp_details:
            stats["fp_images"].append((fname, len(fp_details), fp_details))
            vis = draw_results(img, gts, final_preds, fp_details, fn_details, fname)
            cv2.imwrite(f"{OUT_DIR}/fp_images/{fname}", vis)

        if fn_details:
            stats["fn_images"].append((fname, len(fn_details), fn_details))
            if not fp_details:  # 이미 all_errors에 저장 안된 경우
                vis = draw_results(img, gts, final_preds, fp_details, fn_details, fname)
            cv2.imwrite(f"{OUT_DIR}/fn_images/{fname}", vis)

    # 리포트 생성
    P = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    R = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

    report = []
    report.append("=" * 60)
    report.append("  v19 오류 분석 리포트")
    report.append("=" * 60)
    report.append(f"\n전체: GT={stats['total_gt']}, Pred={stats['total_pred']}")
    report.append(f"TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}")
    report.append(f"P={P:.3f}, R={R:.3f}, F1={F1:.3f}")

    report.append(f"\n--- FP 클래스별 ---")
    for cls, cnt in sorted(stats["fp_by_class"].items()):
        report.append(f"  {CLASS_NAMES[cls]}: {cnt}")

    report.append(f"\n--- FN 클래스별 ---")
    for cls, cnt in sorted(stats["fn_by_class"].items()):
        report.append(f"  {CLASS_NAMES[cls]}: {cnt}")

    report.append(f"\n--- FP bbox 크기별 ---")
    for size, cnt in sorted(stats["fp_by_size"].items()):
        report.append(f"  {size}: {cnt}")

    report.append(f"\n--- FN bbox 크기별 ---")
    for size, cnt in sorted(stats["fn_by_size"].items()):
        report.append(f"  {size}: {cnt}")

    report.append(f"\n--- FP 많은 이미지 TOP 10 ---")
    for fname, cnt, details in sorted(stats["fp_images"], key=lambda x: -x[1])[:10]:
        classes = [d["class"] for d in details]
        report.append(f"  {fname}: FP={cnt} ({', '.join(classes)})")

    report.append(f"\n--- FN 많은 이미지 TOP 10 ---")
    for fname, cnt, details in sorted(stats["fn_images"], key=lambda x: -x[1])[:10]:
        classes = [d["class"] for d in details]
        report.append(f"  {fname}: FN={cnt} ({', '.join(classes)})")

    report.append(f"\n총 에러 이미지: FP={len(stats['fp_images'])}장, FN={len(stats['fn_images'])}장")
    report.append(f"출력: {OUT_DIR}/")

    report_text = "\n".join(report)
    print(report_text)

    with open(f"{OUT_DIR}/report.txt", "w") as f:
        f.write(report_text)

    # JSON 상세 데이터
    json_stats = {
        "summary": {"tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"],
                     "P": P, "R": R, "F1": F1},
        "fp_by_class": {CLASS_NAMES.get(k, k): v for k, v in stats["fp_by_class"].items()},
        "fn_by_class": {CLASS_NAMES.get(k, k): v for k, v in stats["fn_by_class"].items()},
        "fp_by_size": dict(stats["fp_by_size"]),
        "fn_by_size": dict(stats["fn_by_size"]),
    }
    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)

    print(f"\n완료! 리포트: {OUT_DIR}/report.txt")


if __name__ == "__main__":
    main()
