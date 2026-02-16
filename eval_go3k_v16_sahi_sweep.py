#!/usr/bin/env python3
"""v16 SAHI 파라미터 파인튜닝: 타일 크기, overlap, postprocess 최적화"""
import os, time
from collections import defaultdict
from itertools import product
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

IMG_DIR = "/home/lay/hoban/datasets/3k_finetune/val/images"
LBL_DIR = "/home/lay/hoban/datasets/3k_finetune/val/labels"
MODEL = "/home/lay/hoban/hoban_go3k_v16_640/weights/best.pt"


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
            boxes.append((cls, (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                          (cx + w / 2) * img_w, (cy + h / 2) * img_h))
    return boxes


def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def evaluate(all_gt, all_preds, image_set, per_class_conf=None, conf_thresh=None):
    tp = fp = fn = 0
    ctp, cfp, cfn = defaultdict(int), defaultdict(int), defaultdict(int)
    for fname in image_set:
        gts = all_gt.get(fname, [])
        raw = all_preds.get(fname, [])
        if per_class_conf:
            preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in raw
                     if s >= per_class_conf.get(c, 0.5)]
        elif conf_thresh is not None:
            preds = [(c, s, x1, y1, x2, y2) for c, s, x1, y1, x2, y2 in raw
                     if s >= conf_thresh]
        else:
            preds = raw
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
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return tp, fp, fn, p, r, f1, ctp, cfp, cfn


def run_sahi(model_obj, images, img_dir, slice_h, slice_w, overlap_h, overlap_w,
             std_pred=True, pp_type="NMS", pp_thresh=0.4, pp_metric="IOS"):
    preds = {}
    for i, f in enumerate(images):
        if i % 100 == 0:
            print(f"    {i}/{len(images)}...", end="\r")
        r = get_sliced_prediction(
            os.path.join(img_dir, f), model_obj,
            slice_height=slice_h, slice_width=slice_w,
            overlap_height_ratio=overlap_h, overlap_width_ratio=overlap_w,
            perform_standard_pred=std_pred,
            postprocess_type=pp_type, postprocess_match_threshold=pp_thresh,
            postprocess_match_metric=pp_metric, postprocess_class_agnostic=False)
        preds[f] = [(p.category.id, p.score.value,
                     p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                    for p in r.object_prediction_list]
    print(f"    Done: {sum(len(v) for v in preds.values())} preds     ")
    return preds


def best_conf(all_gt, preds, imgs):
    """Find best per-class conf for given predictions"""
    c0_range = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    c1_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    best_f1, best_c0, best_c1 = 0, 0, 0
    best_detail = None
    for c0, c1 in product(c0_range, c1_range):
        tp, fp, fn, p, r, f1, ctp, cfp, cfn = evaluate(
            all_gt, preds, imgs, per_class_conf={0: c0, 1: c1})
        if f1 > best_f1:
            best_f1, best_c0, best_c1 = f1, c0, c1
            best_detail = (tp, fp, fn, p, r, f1, ctp, cfp, cfn)
    return best_f1, best_c0, best_c1, best_detail


if __name__ == "__main__":
    t_start = time.time()

    # Load images & GT
    all_imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    print(f"Eval images: {len(all_imgs)}")

    all_gt, img_sizes = {}, {}
    total_bbox = 0
    for f in all_imgs:
        img = Image.open(os.path.join(IMG_DIR, f))
        img_sizes[f] = img.size
        gt = load_gt(os.path.join(LBL_DIR, f.replace(".jpg", ".txt")), *img.size)
        all_gt[f] = gt
        total_bbox += len(gt)
    print(f"Total GT bbox: {total_bbox}\n")

    # Load model once
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=MODEL,
        confidence_threshold=0.05, device="0")

    results = []

    # ── Phase 1: 타일 크기 sweep ──
    print("=" * 70)
    print("Phase 1: 타일 크기 (Tile Size) Sweep")
    print("=" * 70)

    tile_configs = [
        (720, 1280, "1280x720 (기본)"),
        (640, 640,  "640x640"),
        (480, 640,  "640x480"),
        (360, 640,  "640x360"),
        (540, 960,  "960x540"),
    ]

    for sh, sw, label in tile_configs:
        print(f"\n  [{label}] slice={sw}x{sh}, overlap=0.15")
        t0 = time.time()
        preds = run_sahi(model, all_imgs, IMG_DIR, sh, sw, 0.15, 0.15)
        f1, c0, c1, detail = best_conf(all_gt, preds, all_imgs)
        elapsed = time.time() - t0
        tp, fp, fn, p, r, _, ctp, cfp, cfn = detail
        print(f"    F1={f1:.3f} P={p:.3f} R={r:.3f} @c0={c0},c1={c1} ({elapsed:.0f}s)")
        results.append((f"tile_{label}", f1, c0, c1, detail, preds))

    # Find best tile config
    best_tile = max(results, key=lambda x: x[1])
    print(f"\n  >> Best tile: {best_tile[0]} F1={best_tile[1]:.3f}")

    # ── Phase 2: Overlap sweep (best tile 기준) ──
    print(f"\n{'=' * 70}")
    print("Phase 2: Overlap Sweep (best tile 기준)")
    print("=" * 70)

    # Extract best tile dimensions
    best_tile_idx = results.index(best_tile)
    best_sh, best_sw = tile_configs[best_tile_idx][0], tile_configs[best_tile_idx][1]

    overlap_results = []
    for ovlp in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        print(f"\n  overlap={ovlp}")
        t0 = time.time()
        preds = run_sahi(model, all_imgs, IMG_DIR, best_sh, best_sw, ovlp, ovlp)
        f1, c0, c1, detail = best_conf(all_gt, preds, all_imgs)
        elapsed = time.time() - t0
        tp, fp, fn, p, r, _, _, _, _ = detail
        print(f"    F1={f1:.3f} P={p:.3f} R={r:.3f} @c0={c0},c1={c1} ({elapsed:.0f}s)")
        overlap_results.append((ovlp, f1, c0, c1, detail, preds))

    best_ovlp = max(overlap_results, key=lambda x: x[1])
    print(f"\n  >> Best overlap: {best_ovlp[0]} F1={best_ovlp[1]:.3f}")

    # ── Phase 3: Postprocess sweep (best tile + overlap 기준) ──
    print(f"\n{'=' * 70}")
    print("Phase 3: Postprocess 설정 Sweep")
    print("=" * 70)

    pp_configs = [
        ("NMS", 0.3, "IOS"),
        ("NMS", 0.4, "IOS"),
        ("NMS", 0.5, "IOS"),
        ("NMS", 0.4, "IOU"),
        ("NMS", 0.5, "IOU"),
        ("NMM", 0.4, "IOS"),
        ("NMM", 0.5, "IOS"),
    ]

    pp_results = []
    for pp_type, pp_thresh, pp_metric in pp_configs:
        label = f"{pp_type}/{pp_metric}@{pp_thresh}"
        print(f"\n  [{label}]")
        t0 = time.time()
        preds = run_sahi(model, all_imgs, IMG_DIR, best_sh, best_sw,
                         best_ovlp[0], best_ovlp[0],
                         pp_type=pp_type, pp_thresh=pp_thresh, pp_metric=pp_metric)
        f1, c0, c1, detail = best_conf(all_gt, preds, all_imgs)
        elapsed = time.time() - t0
        tp, fp, fn, p, r, _, _, _, _ = detail
        print(f"    F1={f1:.3f} P={p:.3f} R={r:.3f} @c0={c0},c1={c1} ({elapsed:.0f}s)")
        pp_results.append((label, f1, c0, c1, detail, preds))

    best_pp = max(pp_results, key=lambda x: x[1])
    print(f"\n  >> Best postprocess: {best_pp[0]} F1={best_pp[1]:.3f}")

    # ── Phase 4: perform_standard_pred on/off ──
    print(f"\n{'=' * 70}")
    print("Phase 4: perform_standard_pred (풀이미지 + 타일 병합)")
    print("=" * 70)

    # Parse best pp config
    best_pp_type = best_pp[0].split("/")[0]
    best_pp_metric = best_pp[0].split("/")[1].split("@")[0]
    best_pp_thresh = float(best_pp[0].split("@")[1])

    for std_pred in [True, False]:
        label = f"std_pred={std_pred}"
        print(f"\n  [{label}]")
        t0 = time.time()
        preds = run_sahi(model, all_imgs, IMG_DIR, best_sh, best_sw,
                         best_ovlp[0], best_ovlp[0], std_pred=std_pred,
                         pp_type=best_pp_type, pp_thresh=best_pp_thresh,
                         pp_metric=best_pp_metric)
        f1, c0, c1, detail = best_conf(all_gt, preds, all_imgs)
        elapsed = time.time() - t0
        tp, fp, fn, p, r, _, ctp, cfp, cfn = detail
        print(f"    F1={f1:.3f} P={p:.3f} R={r:.3f} @c0={c0},c1={c1} ({elapsed:.0f}s)")
        # Per-class detail
        for cls_id, cls_name in {0: "helmet_on", 1: "helmet_off"}.items():
            ct, cf, cm = ctp[cls_id], cfp[cls_id], cfn[cls_id]
            cp = ct / (ct + cf) if ct + cf else 0
            cr = ct / (ct + cm) if ct + cm else 0
            cf1 = 2 * cp * cr / (cp + cr) if cp + cr else 0
            print(f"      {cls_name}: P={cp:.3f} R={cr:.3f} F1={cf1:.3f}")

    # ── Summary ──
    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"종합 요약 (소요: {elapsed_total/60:.1f}분)")
    print(f"{'=' * 70}")
    print(f"  Best tile:       {best_tile[0]} F1={best_tile[1]:.3f}")
    print(f"  Best overlap:    {best_ovlp[0]} F1={best_ovlp[1]:.3f}")
    print(f"  Best postprocess: {best_pp[0]} F1={best_pp[1]:.3f}")
    print(f"\n  최적 설정:")
    print(f"    slice: {best_sw}x{best_sh}")
    print(f"    overlap: {best_ovlp[0]}")
    print(f"    postprocess: {best_pp[0]}")
    print(f"    conf: c0={best_pp[2]}, c1={best_pp[3]}")
