#!/usr/bin/env python3
"""GT 오류 이미지 + 모델 예측 Pre-annotation → CVAT 업로드 패키지 생성

v19 모델의 SAHI 추론 결과와 기존 GT를 비교하여:
1. FP (모델이 탐지했으나 GT 없음) → GT 미라벨링 가능성 높음
2. FN (GT 있으나 모델 미탐지) → GT 오류 또는 어려운 케이스
3. Class confusion (ON↔OFF) → 클래스 라벨 오류

해당 이미지를 CVAT에 올려서 검수할 수 있도록:
- 기존 GT + 모델 예측을 병합한 Pre-annotation 생성
- 모델 예측은 GT와 겹치지 않는 것만 추가 (likely missing labels)
- YOLO 1.1 형식 (annotations.zip + images.zip)

사용법:
  python prepare_gt_fix_cvat.py                    # v19로 GT 오류 검출 + CVAT 패키지
  python prepare_gt_fix_cvat.py --model path.pt    # 다른 모델 사용
  python prepare_gt_fix_cvat.py --skip-inference    # 이전 추론 결과 재사용
"""
import argparse
import json
import os
import time
import zipfile
from collections import defaultdict
from PIL import Image

HOBAN = "/home/lay/hoban"
MODEL = f"{HOBAN}/hoban_go3k_v19/weights/best.pt"
OUT_DIR = f"{HOBAN}/cvat_gt_fix"

VAL_IMG = f"{HOBAN}/datasets/3k_finetune/val/images"
VAL_LBL = f"{HOBAN}/datasets/3k_finetune/val/labels"
EXTRA_IMG = f"{HOBAN}/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = f"{HOBAN}/datasets/cvat_helmet_off/valid_helmet_off_137/labels"

CLASS_NAMES = ["person_with_helmet", "person_without_helmet"]
IGNORE_THRESH = 0.00020  # tiny objects (annotation uncertainty)

# 후처리 파라미터 (v19 최적)
GATE_CONF = 0.20
GATE_RADIUS = 30
CROSS_NMS_IOU = 0.3
MIN_AREA = 5e-05
PER_CLASS_CONF = {0: 0.30, 1: 0.15}  # 낮은 임계값 (FP 후보 최대한 포착)


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
                "x1": (cx - w / 2) * img_w,
                "y1": (cy - h / 2) * img_h,
                "x2": (cx + w / 2) * img_w,
                "y2": (cy + h / 2) * img_h,
                "cx": cx, "cy": cy, "w": w, "h": h,  # 원본 YOLO 좌표
                "source": "gt",
            })
    return boxes


def compute_iou(b1, b2):
    x1, y1 = max(b1["x1"], b2["x1"]), max(b1["y1"], b2["y1"])
    x2, y2 = min(b1["x2"], b2["x2"]), min(b1["y2"], b2["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
    a2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def bbox_area_ratio(b, img_w, img_h):
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
    """v19 후처리 파이프라인"""
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
    # Per-class conf
    filtered = [p for p in filtered if p["conf"] >= PER_CLASS_CONF.get(p["cls"], 0.5)]
    return filtered


def run_inference(model_path, eval_images):
    """SAHI + Full-image 추론"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from ultralytics import YOLO
    import logging
    logging.getLogger("sahi").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    print("SAHI 모델 로드...")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.05, device="0", image_size=1280)

    print("Full-image 모델 로드...")
    yolo_model = YOLO(model_path)

    all_sahi = {}
    all_full = {}

    t0 = time.time()
    for i, (fname, img_path, _) in enumerate(eval_images):
        if i % 50 == 0:
            print(f"  추론: {i}/{len(eval_images)}...", end="\r")

        # SAHI
        r = get_sliced_prediction(
            img_path, sahi_model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.4,
            postprocess_match_metric="IOS", verbose=0)
        all_sahi[fname] = [
            {"cls": p.category.id, "conf": p.score.value,
             "x1": p.bbox.minx, "y1": p.bbox.miny,
             "x2": p.bbox.maxx, "y2": p.bbox.maxy, "source": "model"}
            for p in r.object_prediction_list
        ]

        # Full-image (Gate)
        results = yolo_model.predict(img_path, conf=0.01, imgsz=1280,
                                     device="0", verbose=False)
        boxes = results[0].boxes
        all_full[fname] = [
            {"cls": int(boxes.cls[j]), "conf": float(boxes.conf[j]),
             "x1": float(boxes.xyxy[j][0]), "y1": float(boxes.xyxy[j][1]),
             "x2": float(boxes.xyxy[j][2]), "y2": float(boxes.xyxy[j][3]),
             "source": "model"}
            for j in range(len(boxes))
        ]

    print(f"\n추론 완료 ({time.time() - t0:.0f}s)")
    return all_sahi, all_full


def find_gt_errors(eval_images, all_sahi, all_full):
    """GT 오류 탐지: FP (GT 미라벨링), class confusion, FN"""
    error_images = {}  # fname -> {gts, preds, errors}

    for fname, img_path, lbl_path in eval_images:
        img = Image.open(img_path)
        img_w, img_h = img.size

        gts = load_gt(lbl_path, img_w, img_h)
        sahi_preds = all_sahi.get(fname, [])
        full_preds = all_full.get(fname, [])
        preds = apply_pipeline(sahi_preds, full_preds, img_w, img_h)

        errors = []

        # active GT만 (ignore 제외)
        active_gts = [g for g in gts if bbox_area_ratio(g, img_w, img_h) >= IGNORE_THRESH]
        ignore_gts = [g for g in gts if bbox_area_ratio(g, img_w, img_h) < IGNORE_THRESH]

        matched_gt = set()
        matched_pred = set()
        unmatched_preds = []

        # Pred→GT 매칭 (conf 내림차순)
        sorted_preds = sorted(enumerate(preds), key=lambda x: -x[1]["conf"])
        for pi, pred in sorted_preds:
            pred_area = bbox_area_ratio(pred, img_w, img_h)

            # active GT 매칭
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(active_gts):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

            if best_iou >= 0.5 and best_gi >= 0:
                # class confusion 체크
                if pred["cls"] != active_gts[best_gi]["cls"]:
                    errors.append({
                        "type": "class_confusion",
                        "pred": pred,
                        "gt": active_gts[best_gi],
                        "iou": best_iou,
                    })
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                continue

            # ignore GT 매칭 → skip
            ignore_match = False
            for gt in ignore_gts:
                if compute_iou(pred, gt) >= 0.5:
                    ignore_match = True
                    break
            if ignore_match or pred_area < IGNORE_THRESH:
                matched_pred.add(pi)
                continue

            # FP → likely missing GT label
            unmatched_preds.append(pred)
            errors.append({
                "type": "missing_gt",
                "pred": pred,
                "conf": pred["conf"],
                "area": pred_area,
            })

        # FN: 미매칭 active GT
        for gi, gt in enumerate(active_gts):
            if gi not in matched_gt:
                errors.append({
                    "type": "fn",
                    "gt": gt,
                    "area": bbox_area_ratio(gt, img_w, img_h),
                })

        if errors:
            error_images[fname] = {
                "img_path": img_path,
                "lbl_path": lbl_path,
                "img_size": (img_w, img_h),
                "gts": gts,
                "active_gts": active_gts,
                "preds": preds,
                "unmatched_preds": unmatched_preds,
                "errors": errors,
            }

    return error_images


def merge_annotations(info):
    """기존 GT + 모델의 미매칭 예측을 병합 → YOLO format"""
    img_w, img_h = info["img_size"]
    lines = []

    # 1. 기존 GT 유지
    for gt in info["gts"]:
        lines.append(f"{gt['cls']} {gt['cx']:.6f} {gt['cy']:.6f} {gt['w']:.6f} {gt['h']:.6f}")

    # 2. 모델의 미매칭 예측 추가 (GT에 없는 것 = likely missing labels)
    for pred in info["unmatched_preds"]:
        cx = ((pred["x1"] + pred["x2"]) / 2) / img_w
        cy = ((pred["y1"] + pred["y2"]) / 2) / img_h
        w = (pred["x2"] - pred["x1"]) / img_w
        h = (pred["y2"] - pred["y1"]) / img_h
        lines.append(f"{pred['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


def package_cvat(out_dir, error_images):
    """CVAT 업로드용 YOLO 1.1 패키지 생성"""
    os.makedirs(out_dir, exist_ok=True)

    # annotations.zip
    ann_path = os.path.join(out_dir, "annotations.zip")
    with zipfile.ZipFile(ann_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.data",
                     f"classes = {len(CLASS_NAMES)}\n"
                     f"train = data/train.txt\n"
                     f"names = data/obj.names\n"
                     f"backup = backup/\n")
        zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")

        train_lines = []
        for fname, info in sorted(error_images.items()):
            train_lines.append(f"data/obj_train_data/{fname}")
            merged = merge_annotations(info)
            zf.writestr(f"obj_train_data/{fname.replace('.jpg', '.txt')}", merged + "\n")

        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    # images.zip
    img_path = os.path.join(out_dir, "images.zip")
    with zipfile.ZipFile(img_path, "w", zipfile.ZIP_STORED) as zf:
        for i, (fname, info) in enumerate(sorted(error_images.items())):
            if i % 20 == 0:
                print(f"  이미지 패키징: {i}/{len(error_images)}...", end="\r")
            zf.write(info["img_path"], fname)

    print(f"\n  annotations.zip: {os.path.getsize(ann_path) / 1024:.0f}KB")
    print(f"  images.zip: {os.path.getsize(img_path) / 1024 / 1024:.1f}MB")
    return ann_path, img_path


def save_report(out_dir, error_images):
    """오류 리포트 저장"""
    report_path = os.path.join(out_dir, "error_report.txt")
    stats = defaultdict(int)

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  GT 오류 리포트 (CVAT 검수용)\n")
        f.write("=" * 70 + "\n\n")

        for fname in sorted(error_images.keys()):
            info = error_images[fname]
            errors = info["errors"]
            f.write(f"\n--- {fname} ---\n")
            f.write(f"  기존 GT: {len(info['gts'])}개, 활성 GT: {len(info['active_gts'])}개\n")
            f.write(f"  모델 예측: {len(info['preds'])}개, 미매칭: {len(info['unmatched_preds'])}개\n")

            for e in errors:
                stats[e["type"]] += 1
                if e["type"] == "missing_gt":
                    cls_name = CLASS_NAMES[e["pred"]["cls"]]
                    f.write(f"  [미라벨링] {cls_name} conf={e['conf']:.2f} area={e['area']:.5f}\n")
                elif e["type"] == "class_confusion":
                    gt_name = CLASS_NAMES[e["gt"]["cls"]]
                    pred_name = CLASS_NAMES[e["pred"]["cls"]]
                    f.write(f"  [클래스혼동] GT={gt_name} → Pred={pred_name} IoU={e['iou']:.2f}\n")
                elif e["type"] == "fn":
                    cls_name = CLASS_NAMES[e["gt"]["cls"]]
                    f.write(f"  [FN] {cls_name} area={e['area']:.5f}\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"  요약\n")
        f.write(f"{'='*70}\n")
        f.write(f"  오류 이미지: {len(error_images)}장\n")
        f.write(f"  미라벨링 (FP→GT 추가 필요): {stats['missing_gt']}개\n")
        f.write(f"  클래스 혼동 (라벨 수정 필요): {stats['class_confusion']}개\n")
        f.write(f"  FN (모델 미탐지, GT 확인): {stats['fn']}개\n")

    return report_path, stats


def main():
    parser = argparse.ArgumentParser(description="GT 오류 → CVAT 업로드 패키지")
    parser.add_argument("--model", default=MODEL, help="모델 경로")
    parser.add_argument("--skip-inference", action="store_true",
                        help="이전 추론 캐시 재사용")
    parser.add_argument("--out-dir", default=OUT_DIR, help="출력 디렉터리")
    args = parser.parse_args()

    print("=" * 60)
    print("  GT 오류 검출 → CVAT 패키지 생성")
    print("=" * 60)

    # 1. 평가셋 수집
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

    print(f"\n평가셋: {len(eval_images)}장")
    print(f"모델: {args.model}")

    # 2. 추론
    cache_path = os.path.join(args.out_dir, "inference_cache.json")
    if args.skip_inference and os.path.exists(cache_path):
        print("\n캐시에서 추론 결과 로드...")
        with open(cache_path) as f:
            cache = json.load(f)
        all_sahi = {k: [dict(p) for p in v] for k, v in cache["sahi"].items()}
        all_full = {k: [dict(p) for p in v] for k, v in cache["full"].items()}
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        all_sahi, all_full = run_inference(args.model, eval_images)
        # 캐시 저장
        cache = {"sahi": all_sahi, "full": all_full}
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"캐시 저장: {cache_path}")

    # 3. GT 오류 탐지
    print("\nGT 오류 탐지 중...")
    error_images = find_gt_errors(eval_images, all_sahi, all_full)

    # 통계
    total_missing = sum(1 for info in error_images.values()
                        for e in info["errors"] if e["type"] == "missing_gt")
    total_confusion = sum(1 for info in error_images.values()
                          for e in info["errors"] if e["type"] == "class_confusion")
    total_fn = sum(1 for info in error_images.values()
                   for e in info["errors"] if e["type"] == "fn")
    total_added = sum(len(info["unmatched_preds"]) for info in error_images.values())

    print(f"\n{'─'*60}")
    print(f"  탐지 결과")
    print(f"{'─'*60}")
    print(f"  오류 이미지: {len(error_images)}장 / {len(eval_images)}장")
    print(f"  미라벨링 (GT에 추가 필요): {total_missing}개")
    print(f"  클래스 혼동 (라벨 수정): {total_confusion}개")
    print(f"  FN (모델 미탐지): {total_fn}개")
    print(f"  Pre-annotation 추가: {total_added}개 bbox")

    # 4. CVAT 패키지 생성
    print(f"\nCVAT 패키지 생성 중...")
    ann_path, img_path = package_cvat(args.out_dir, error_images)

    # 5. 리포트
    report_path, stats = save_report(args.out_dir, error_images)

    print(f"\n{'='*60}")
    print(f"  완료!")
    print(f"{'='*60}")
    print(f"  출력 디렉터리: {args.out_dir}")
    print(f"  annotations.zip: CVAT → Create Task → YOLO 1.1로 업로드")
    print(f"  images.zip:      CVAT → Task에 이미지 추가")
    print(f"  error_report.txt: 오류 상세 리포트")
    print(f"\n  CVAT 업로드 방법:")
    print(f"    1. CVAT에서 새 Task 생성")
    print(f"    2. images.zip 업로드 (이미지)")
    print(f"    3. Annotations → Upload → YOLO 1.1 → annotations.zip")
    print(f"    4. 각 이미지에서 모델 예측(추가된 bbox) 검수/수정")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
