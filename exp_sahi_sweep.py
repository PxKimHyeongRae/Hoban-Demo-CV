#!/usr/bin/env python3
"""SAHI 파라미터 튜닝 sweep (v26 기준)

테스트 파라미터:
  1. 타일 크기: 640x360, 960x540, 1280x720, 1920x1080
  2. overlap: 0.10, 0.15, 0.20, 0.30
  3. postprocess: NMS vs GREEDYNMM, threshold 0.3~0.5
  4. postprocess metric: IOU vs IOS
  5. perform_standard_pred: True/False
  6. gate conf/radius
  7. min_area

사용법:
  python exp_sahi_sweep.py              # 전체 sweep
  python exp_sahi_sweep.py --quick      # 주요 조합만
"""
import os, sys, time, logging, argparse, json
from collections import defaultdict
from PIL import Image

logging.getLogger("sahi").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ── 설정 ──
VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
MODEL = "/home/lay/hoban/hoban_go3k_v26/weights/best.pt"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}
IGNORE_THRESH = 0.00020


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


def apply_pipeline(preds, full_raw, img_w, img_h, min_area=5e-05, gate_conf=0.20, gate_radius=30):
    """후처리 파이프라인 (파라미터화)"""
    filtered = cross_class_nms(preds, 0.3)
    img_area = img_w * img_h
    filtered = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in filtered
                if ((x2-x1)*(y2-y1)) / img_area >= min_area]
    if gate_conf > 0 and gate_radius > 0:
        gates = [(x1,y1,x2,y2) for conf,x1,y1,x2,y2 in full_raw if conf >= gate_conf]
        if gates:
            gated = []
            for c,s,x1,y1,x2,y2 in filtered:
                cx, cy = (x1+x2)/2, (y1+y2)/2
                for gx1,gy1,gx2,gy2 in gates:
                    gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
                    if abs(cx-gcx) <= gate_radius and abs(cy-gcy) <= gate_radius:
                        gated.append((c,s,x1,y1,x2,y2))
                        break
            filtered = gated
    return filtered


def evaluate_ignore(all_gt, all_preds, image_set, img_sizes,
                    per_class_conf=None, ignore_thresh=IGNORE_THRESH):
    tp = fp = fn = 0
    ctp, cfp, cfn = defaultdict(int), defaultdict(int), defaultdict(int)
    ignored_preds = ignored_gts = 0

    for fname in image_set:
        gts = all_gt.get(fname, [])
        raw = all_preds.get(fname, [])
        img_w, img_h = img_sizes[fname]

        if per_class_conf:
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in raw
                     if s >= per_class_conf.get(c, 0.5)]
        else:
            preds = raw

        active_gts, ignore_gts_list = [], []
        for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
            area = bbox_area_ratio(gx1, gy1, gx2, gy2, img_w, img_h)
            if area >= ignore_thresh:
                active_gts.append((gi, gc, gx1, gy1, gx2, gy2))
            else:
                ignore_gts_list.append((gi, gc, gx1, gy1, gx2, gy2))
                ignored_gts += 1

        matched_active = set()
        matched_ignore = set()
        sorted_preds = sorted(enumerate(preds), key=lambda x: -x[1][1])

        for pi, (pc, ps, px1, py1, px2, py2) in sorted_preds:
            best_iou, best_idx = 0, -1
            for ai, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(active_gts):
                if ai in matched_active or gc != pc:
                    continue
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou, best_idx = iou, ai

            if best_iou >= 0.5 and best_idx >= 0:
                tp += 1; ctp[pc] += 1
                matched_active.add(best_idx)
                continue

            ignore_match = False
            for ii, (gi, gc, gx1, gy1, gx2, gy2) in enumerate(ignore_gts_list):
                if ii in matched_ignore or gc != pc:
                    continue
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou >= 0.5:
                    ignore_match = True
                    matched_ignore.add(ii)
                    break

            if ignore_match:
                ignored_preds += 1
                continue

            pred_area = bbox_area_ratio(px1, py1, px2, py2, img_w, img_h)
            if pred_area < ignore_thresh:
                ignored_preds += 1
                continue

            fp += 1; cfp[pc] += 1

        for ai in range(len(active_gts)):
            if ai not in matched_active:
                fn += 1; cfn[active_gts[ai][1]] += 1

    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn


def run_sahi_config(model_path, combined, img_paths,
                    slice_w, slice_h, overlap, pp_type, pp_thresh, pp_metric,
                    standard_pred):
    """특정 SAHI 설정으로 추론"""
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.05, device="0", image_size=1280)

    all_preds = {}
    for i, f in enumerate(combined):
        r = get_sliced_prediction(
            img_paths[f], model,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=standard_pred,
            postprocess_type=pp_type,
            postprocess_match_threshold=pp_thresh,
            postprocess_match_metric=pp_metric,
            verbose=0)
        all_preds[f] = [(p.category.id, p.score.value,
                         p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                        for p in r.object_prediction_list]
    return all_preds


def run_full_inference(model_path, combined, img_paths):
    """Full-image 추론 (Gate용)"""
    from ultralytics import YOLO
    yolo_model = YOLO(model_path)
    full_preds = {}
    for f in combined:
        results = yolo_model.predict(img_paths[f], conf=0.01, imgsz=1280,
                                     device="0", verbose=False)
        boxes = results[0].boxes
        full_preds[f] = [(float(boxes.conf[j]), *[float(v) for v in boxes.xyxy[j]])
                         for j in range(len(boxes))]
    return full_preds


def best_f1_for_config(all_preds, full_preds, all_gt, combined, img_sizes,
                       min_area=5e-05, gate_conf=0.20, gate_radius=30):
    """후처리 적용 후 최적 conf sweep으로 F1 계산"""
    processed = {}
    for f in combined:
        img_w, img_h = img_sizes[f]
        processed[f] = apply_pipeline(all_preds[f], full_preds.get(f, []),
                                      img_w, img_h, min_area, gate_conf, gate_radius)

    best_f1, best_conf, best_detail = 0, {}, None
    for c0 in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for c1 in [0.10, 0.15, 0.20, 0.25, 0.30]:
            pc = {0: c0, 1: c1}
            tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate_ignore(
                all_gt, processed, combined, img_sizes, per_class_conf=pc)
            if f1 > best_f1:
                best_f1, best_conf = f1, pc
                best_detail = (tp, fp, fn, p, r, f1, ctp, cfp, cfn)

    return best_f1, best_conf, best_detail


def main():
    parser = argparse.ArgumentParser(description="SAHI 파라미터 sweep")
    parser.add_argument("--quick", action="store_true", help="주요 조합만 테스트")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    print("=" * 70)
    print("  SAHI 파라미터 튜닝 sweep (v26)")
    print("=" * 70)

    # GT 로드
    val_imgs = sorted(f for f in os.listdir(VAL_IMG) if f.endswith(".jpg"))
    all_gt, img_sizes, img_paths = {}, {}, {}
    for f in val_imgs:
        path = os.path.join(VAL_IMG, f)
        img = Image.open(path)
        img_sizes[f] = img.size
        img_paths[f] = path
        all_gt[f] = load_gt(os.path.join(VAL_LBL, f.replace(".jpg", ".txt")), *img.size)

    extra_imgs = []
    if os.path.isdir(EXTRA_IMG):
        for f in sorted(os.listdir(EXTRA_IMG)):
            if f.endswith(".jpg") and f not in all_gt:
                path = os.path.join(EXTRA_IMG, f)
                img = Image.open(path)
                img_sizes[f] = img.size
                img_paths[f] = path
                all_gt[f] = load_gt(os.path.join(EXTRA_LBL, f.replace(".jpg", ".txt")), *img.size)
                extra_imgs.append(f)

    combined = val_imgs + extra_imgs
    print(f"평가 세트: {len(combined)}장")

    # Full-image 추론 (Gate용, 한 번만)
    print("\n[0] Full-image 추론 (Gate용)...")
    t0 = time.time()
    full_preds = run_full_inference(args.model, combined, img_paths)
    print(f"  완료 ({time.time()-t0:.0f}s)")

    # ── SAHI 설정 조합 ──
    if args.quick:
        configs = [
            # 기본 (현재 설정)
            {"name": "baseline", "slice_w": 1280, "slice_h": 720, "overlap": 0.15,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
            # 타일 크기 변경
            {"name": "tile_960x540", "slice_w": 960, "slice_h": 540, "overlap": 0.15,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
            {"name": "tile_640x360", "slice_w": 640, "slice_h": 360, "overlap": 0.15,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
            # overlap 변경
            {"name": "overlap_0.25", "slice_w": 1280, "slice_h": 720, "overlap": 0.25,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
            {"name": "overlap_0.30", "slice_w": 1280, "slice_h": 720, "overlap": 0.30,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
            # postprocess 변경
            {"name": "pp_GREEDYNMM", "slice_w": 1280, "slice_h": 720, "overlap": 0.15,
             "pp_type": "GREEDYNMM", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
            {"name": "pp_thresh_0.3", "slice_w": 1280, "slice_h": 720, "overlap": 0.15,
             "pp_type": "NMS", "pp_thresh": 0.3, "pp_metric": "IOS", "std_pred": True},
            {"name": "pp_thresh_0.5", "slice_w": 1280, "slice_h": 720, "overlap": 0.15,
             "pp_type": "NMS", "pp_thresh": 0.5, "pp_metric": "IOS", "std_pred": True},
            # metric 변경
            {"name": "metric_IOU", "slice_w": 1280, "slice_h": 720, "overlap": 0.15,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOU", "std_pred": True},
            # standard_pred 끄기
            {"name": "no_std_pred", "slice_w": 1280, "slice_h": 720, "overlap": 0.15,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": False},
            # 소형 타일 + 높은 overlap
            {"name": "tile_640_ov0.25", "slice_w": 640, "slice_h": 360, "overlap": 0.25,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
            {"name": "tile_960_ov0.25", "slice_w": 960, "slice_h": 540, "overlap": 0.25,
             "pp_type": "NMS", "pp_thresh": 0.4, "pp_metric": "IOS", "std_pred": True},
        ]
    else:
        configs = []
        for sw, sh in [(640, 360), (960, 540), (1280, 720)]:
            for ov in [0.15, 0.25]:
                for pp in ["NMS", "GREEDYNMM"]:
                    for pt in [0.3, 0.4, 0.5]:
                        for pm in ["IOS", "IOU"]:
                            for sp in [True, False]:
                                configs.append({
                                    "name": f"t{sw}x{sh}_ov{ov}_pp{pp}_{pt}_{pm}_sp{sp}",
                                    "slice_w": sw, "slice_h": sh, "overlap": ov,
                                    "pp_type": pp, "pp_thresh": pt, "pp_metric": pm,
                                    "std_pred": sp,
                                })

    print(f"\n총 {len(configs)}개 설정 테스트")

    # ── Sweep 실행 ──
    results = []
    for idx, cfg in enumerate(configs):
        print(f"\n[{idx+1}/{len(configs)}] {cfg['name']}...")
        t0 = time.time()

        sahi_preds = run_sahi_config(
            args.model, combined, img_paths,
            cfg["slice_w"], cfg["slice_h"], cfg["overlap"],
            cfg["pp_type"], cfg["pp_thresh"], cfg["pp_metric"],
            cfg["std_pred"])

        n_preds = sum(len(v) for v in sahi_preds.values())
        elapsed = time.time() - t0

        f1, conf, detail = best_f1_for_config(
            sahi_preds, full_preds, all_gt, combined, img_sizes)

        tp, fp, fn, p, r, f1_val, ctp, cfp, cfn = detail

        # per-class F1
        on_tp, on_fp, on_fn = ctp[0], cfp[0], cfn[0]
        on_p = on_tp/(on_tp+on_fp) if on_tp+on_fp else 0
        on_r = on_tp/(on_tp+on_fn) if on_tp+on_fn else 0
        on_f1 = 2*on_p*on_r/(on_p+on_r) if on_p+on_r else 0

        off_tp, off_fp, off_fn = ctp[1], cfp[1], cfn[1]
        off_p = off_tp/(off_tp+off_fp) if off_tp+off_fp else 0
        off_r = off_tp/(off_tp+off_fn) if off_tp+off_fn else 0
        off_f1 = 2*off_p*off_r/(off_p+off_r) if off_p+off_r else 0

        result = {
            "name": cfg["name"],
            "f1": f1_val, "p": p, "r": r,
            "tp": tp, "fp": fp, "fn": fn,
            "on_f1": on_f1, "off_f1": off_f1,
            "conf": conf, "preds": n_preds, "time": elapsed,
            "config": cfg,
        }
        results.append(result)
        print(f"  F1={f1_val:.4f} P={p:.3f} R={r:.3f} "
              f"(TP={tp} FP={fp} FN={fn}) "
              f"ON={on_f1:.3f} OFF={off_f1:.3f} "
              f"@c0={conf[0]},c1={conf[1]} "
              f"({n_preds} preds, {elapsed:.0f}s)")

    # ── 결과 정리 ──
    results.sort(key=lambda x: -x["f1"])

    print(f"\n{'='*90}")
    print(f"  SAHI 파라미터 sweep 결과 (v26, {len(combined)}장)")
    print(f"{'='*90}")
    print(f"  {'#':>2} {'설정':<25} {'F1':>7} {'P':>7} {'R':>7} {'ON_F1':>7} {'OFF_F1':>7} {'FP':>5} {'FN':>5} {'Time':>6}")
    print(f"  {'-'*82}")
    for i, r in enumerate(results):
        marker = " *" if i == 0 else ""
        print(f"  {i+1:>2} {r['name']:<25} {r['f1']:>7.4f} {r['p']:>7.3f} {r['r']:>7.3f} "
              f"{r['on_f1']:>7.3f} {r['off_f1']:>7.3f} {r['fp']:>5} {r['fn']:>5} {r['time']:>5.0f}s{marker}")

    # 최고 vs baseline
    baseline = next((r for r in results if r["name"] == "baseline"), results[-1])
    best = results[0]
    print(f"\n  Baseline: F1={baseline['f1']:.4f} ({baseline['name']})")
    print(f"  Best:     F1={best['f1']:.4f} ({best['name']})")
    diff = best['f1'] - baseline['f1']
    print(f"  차이:     {'+' if diff >= 0 else ''}{diff:.4f}")

    if best['name'] != 'baseline':
        print(f"\n  최적 설정:")
        cfg = best['config']
        print(f"    slice: {cfg['slice_w']}x{cfg['slice_h']}")
        print(f"    overlap: {cfg['overlap']}")
        print(f"    postprocess: {cfg['pp_type']} thresh={cfg['pp_thresh']} metric={cfg['pp_metric']}")
        print(f"    standard_pred: {cfg['std_pred']}")
        print(f"    best conf: c0={best['conf'][0]}, c1={best['conf'][1]}")

    # 결과 저장
    out_path = "/home/lay/hoban/exp_sahi_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
