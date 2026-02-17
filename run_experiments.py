#!/usr/bin/env python3
"""
v16 실험 러너: Quick Train (20ep) + SAHI Eval 자동화

Phase A: 학습 변수 실험 7개
Phase B: 앙상블 실험 5개

실행: python run_experiments.py [--phase A|B|all] [--exp A1,A2,...]
"""
import os, sys, time, json, argparse, random, shutil
from pathlib import Path
from collections import defaultdict
from itertools import product

# ── Paths ──
BASE = Path("/home/lay/hoban")
V13_S2 = str(BASE / "hoban_v13_stage2/weights/best.pt")
V16_BEST = str(BASE / "hoban_go3k_v16_640/weights/best.pt")
COCO_PT = "yolov8m.pt"
THRK_TRAIN_IMG = BASE / "datasets/3k_finetune/train/images"
THRK_TRAIN_LBL = BASE / "datasets/3k_finetune/train/labels"
VAL_IMG = str(BASE / "datasets/3k_finetune/val/images")
VAL_LBL = str(BASE / "datasets/3k_finetune/val/labels")
V13_IMG = BASE / "datasets_v13/train/images"
V13_LBL = BASE / "datasets_v13/train/labels"
RESULTS_FILE = BASE / "experiment_results.json"

EXISTING_MODELS = {
    "v2": str(BASE / "hoban_go2k_v2/weights/best.pt"),
    "v3": str(BASE / "hoban_go2k_v3/weights/best.pt"),
    "v5": str(BASE / "hoban_go2k_v5/weights/best.pt"),
}


def build_dataset(exp_name, v13_count=8000, v13_filter=False, seed=42):
    """Build experiment dataset, return data.yaml path"""
    random.seed(seed)
    out = BASE / f"datasets_exp_{exp_name}"

    if (out / "data.yaml").exists():
        print(f"  Dataset exists: {out}")
        return str(out / "data.yaml")

    out_ti = out / "train/images"
    out_tl = out / "train/labels"
    out_vi = out / "valid/images"
    out_vl = out / "valid/labels"
    for d in [out_ti, out_tl, out_vi, out_vl]:
        d.mkdir(parents=True, exist_ok=True)

    # 3k train
    cctv = sorted(f for f in os.listdir(THRK_TRAIN_IMG) if f.endswith(".jpg"))
    for name in cctv:
        (out_ti / name).symlink_to(THRK_TRAIN_IMG / name)
        lbl = THRK_TRAIN_LBL / name.replace(".jpg", ".txt")
        if lbl.exists():
            (out_tl / name.replace(".jpg", ".txt")).symlink_to(lbl)

    # v13
    if v13_count > 0:
        v13_all = sorted(f for f in os.listdir(V13_IMG) if f.endswith(".jpg"))
        if v13_filter:
            filtered = []
            for name in v13_all:
                lbl = V13_LBL / name.replace(".jpg", ".txt")
                if not lbl.exists():
                    continue
                areas = []
                with open(lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            areas.append(float(parts[3]) * float(parts[4]))
                if areas and 0.0001 <= sum(areas)/len(areas) <= 0.01:
                    filtered.append(name)
            v13_all = filtered
            print(f"  v13 filtered: {len(v13_all)}")

        n = min(v13_count, len(v13_all))
        v13_sel = sorted(random.sample(v13_all, n))
        for name in v13_sel:
            (out_ti / name).symlink_to(V13_IMG / name)
            lbl = V13_LBL / name.replace(".jpg", ".txt")
            if lbl.exists():
                (out_tl / name.replace(".jpg", ".txt")).symlink_to(lbl)
        print(f"  v13: {n} images")

    # val (symlink to 3k val)
    val_imgs = sorted(f for f in os.listdir(VAL_IMG) if f.endswith(".jpg"))
    for name in val_imgs:
        (out_vi / name).symlink_to(Path(VAL_IMG) / name)
        lbl = Path(VAL_LBL) / name.replace(".jpg", ".txt")
        if lbl.exists():
            (out_vl / name.replace(".jpg", ".txt")).symlink_to(lbl)

    total = len(os.listdir(out_ti))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train/images\nval: valid/images\n\nnc: 2\nnames:\n  0: person_with_helmet\n  1: person_without_helmet\n")
    print(f"  Dataset built: {total} train, {len(val_imgs)} val")
    return str(out / "data.yaml")


def quick_train(exp_name, data_yaml, base_model=V13_S2, imgsz=640, epochs=20,
                batch=None, extra_kwargs=None):
    """Quick training, return best.pt path"""
    from ultralytics import YOLO

    out_dir = BASE / f"exp_{exp_name}"
    best_pt = out_dir / "weights/best.pt"
    if best_pt.exists():
        print(f"  Model exists: {best_pt}")
        return str(best_pt)

    if batch is None:
        batch = 4 if imgsz == 1280 else 24

    model = YOLO(base_model)
    kwargs = dict(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
        device="0", project=str(BASE), name=f"exp_{exp_name}", exist_ok=True,
        optimizer="SGD", lr0=0.005, lrf=0.01, momentum=0.937,
        warmup_epochs=3.0, weight_decay=0.0005, cos_lr=True,
        mosaic=1.0, mixup=0.1, copy_paste=0.15,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        scale=0.5, translate=0.1, degrees=5.0, fliplr=0.5,
        erasing=0.15, close_mosaic=10,
        patience=20, amp=True, workers=4, seed=42,
        plots=False, save=True, val=True,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    model.train(**kwargs)
    return str(best_pt)


def sahi_eval(model_path, exp_name, image_size=None):
    """SAHI eval on full val set, return best F1 and details"""
    from PIL import Image
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    kw = {"image_size": image_size} if image_size else {}
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=model_path,
        confidence_threshold=0.05, device="0", **kw)

    all_imgs = sorted(f for f in os.listdir(VAL_IMG) if f.endswith(".jpg"))
    all_gt, img_sizes = {}, {}
    for f in all_imgs:
        img = Image.open(os.path.join(VAL_IMG, f))
        img_sizes[f] = img.size
        gt = []
        lbl = os.path.join(VAL_LBL, f.replace(".jpg", ".txt"))
        if os.path.exists(lbl):
            with open(lbl) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        iw, ih = img.size
                        gt.append((cls, (cx-w/2)*iw, (cy-h/2)*ih, (cx+w/2)*iw, (cy+h/2)*ih))
        all_gt[f] = gt

    all_preds = {}
    for i, f in enumerate(all_imgs):
        if i % 100 == 0:
            print(f"    SAHI {i}/{len(all_imgs)}...", end="\r")
        r = get_sliced_prediction(
            os.path.join(VAL_IMG, f), model,
            slice_height=720, slice_width=1280,
            overlap_height_ratio=0.15, overlap_width_ratio=0.15,
            perform_standard_pred=True,
            postprocess_type="NMS", postprocess_match_threshold=0.3,
            postprocess_match_metric="IOS", postprocess_class_agnostic=False)
        all_preds[f] = [(p.category.id, p.score.value,
                         p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy)
                        for p in r.object_prediction_list]
    print(f"    SAHI done: {sum(len(v) for v in all_preds.values())} preds     ")

    # Per-class conf sweep
    c0r = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    c1r = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    best_f1, best_c0, best_c1 = 0, 0, 0
    for c0, c1 in product(c0r, c1r):
        tp = fp = fn = 0
        for fname in all_imgs:
            gts = all_gt[fname]
            preds = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in all_preds[fname]
                     if s >= ({0:c0, 1:c1}.get(c, 0.5))]
            matched = set()
            for _, (pc,ps,px1,py1,px2,py2) in sorted(enumerate(preds), key=lambda x:-x[1][1]):
                bi, bv = -1, 0
                for gi, (gc,gx1,gy1,gx2,gy2) in enumerate(gts):
                    if gi in matched or gc != pc: continue
                    ix1,iy1 = max(px1,gx1), max(py1,gy1)
                    ix2,iy2 = min(px2,gx2), min(py2,gy2)
                    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                    a1 = (px2-px1)*(py2-py1)
                    a2 = (gx2-gx1)*(gy2-gy1)
                    iou = inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0
                    if iou > bv: bv, bi = iou, gi
                if bv >= 0.5 and bi >= 0:
                    tp += 1; matched.add(bi)
                else:
                    fp += 1
            fn += len(gts) - len(matched)
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        if f1 > best_f1:
            best_f1, best_c0, best_c1 = f1, c0, c1

    return best_f1, best_c0, best_c1, all_preds, img_sizes


def wbf_ensemble(preds_list, img_sizes, iou_thr=0.4, weights=None):
    """WBF ensemble of multiple model predictions"""
    from ensemble_boxes import weighted_boxes_fusion
    import numpy as np

    merged = {}
    all_files = set()
    for p in preds_list:
        all_files |= set(p.keys())

    for f in all_files:
        w, h = img_sizes[f]
        bl, sl, ll = [], [], []
        for p in preds_list:
            raw = p.get(f, [])
            if not raw:
                bl.append(np.empty((0, 4)))
                sl.append([])
                ll.append([])
                continue
            bl.append([[x1/w,y1/h,x2/w,y2/h] for _,_,x1,y1,x2,y2 in raw])
            sl.append([s for _,s,_,_,_,_ in raw])
            ll.append([c for c,_,_,_,_,_ in raw])
        if all(len(s)==0 for s in sl):
            merged[f] = []
            continue
        mb, ms, ml = weighted_boxes_fusion(bl, sl, ll, weights=weights,
                                           iou_thr=iou_thr, skip_box_thr=0.0001)
        merged[f] = [(int(l), float(s), b[0]*w, b[1]*h, b[2]*w, b[3]*h)
                     for b,s,l in zip(mb,ms,ml)]
    return merged


def eval_preds(all_gt, preds, all_imgs):
    """Evaluate cached predictions with per-class conf sweep"""
    c0r = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    c1r = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    best_f1, best_c0, best_c1 = 0, 0, 0
    for c0, c1 in product(c0r, c1r):
        tp = fp = fn = 0
        for fname in all_imgs:
            gts = all_gt.get(fname, [])
            ps = [(c,s,x1,y1,x2,y2) for c,s,x1,y1,x2,y2 in preds.get(fname, [])
                  if s >= ({0:c0, 1:c1}.get(c, 0.5))]
            matched = set()
            for _, (pc,ps2,px1,py1,px2,py2) in sorted(enumerate(ps), key=lambda x:-x[1][1]):
                bi, bv = -1, 0
                for gi, (gc,gx1,gy1,gx2,gy2) in enumerate(gts):
                    if gi in matched or gc != pc: continue
                    ix1,iy1 = max(px1,gx1), max(py1,gy1)
                    ix2,iy2 = min(px2,gx2), min(py2,gy2)
                    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                    a1 = (px2-px1)*(py2-py1)
                    a2 = (gx2-gx1)*(gy2-gy1)
                    iou = inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0
                    if iou > bv: bv, bi = iou, gi
                if bv >= 0.5 and bi >= 0:
                    tp += 1; matched.add(bi)
                else:
                    fp += 1
            fn += len(gts) - len(matched)
        p = tp/(tp+fp) if tp+fp else 0
        r = tp/(tp+fn) if tp+fn else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        if f1 > best_f1:
            best_f1, best_c0, best_c1 = f1, c0, c1
    return best_f1, best_c0, best_c1


# ── Experiment Definitions ──

EXPERIMENTS_A = {
    "A1_no_v13": dict(v13_count=0),
    "A2_v13_4k": dict(v13_count=4000),
    "A3_v13_filtered": dict(v13_count=8000, v13_filter=True),
    "A4_imgsz_1280": dict(imgsz=1280),
    "A5_coco_pt": dict(base_model=COCO_PT),
    "A6_heavy_aug": dict(extra_kwargs=dict(copy_paste=0.3, mixup=0.2, erasing=0.2)),
    "A7_light_aug": dict(extra_kwargs=dict(mosaic=0.5, mixup=0.0, copy_paste=0.0, erasing=0.0)),
}


def run_phase_a(selected=None):
    """Run Phase A: training experiments"""
    results = {}
    for name, cfg in EXPERIMENTS_A.items():
        if selected and name not in selected:
            continue
        print(f"\n{'='*60}")
        print(f"[{name}]")
        print(f"{'='*60}")
        t0 = time.time()

        # Build dataset
        v13_count = cfg.get("v13_count", 8000)
        v13_filter = cfg.get("v13_filter", False)
        data_yaml = build_dataset(name, v13_count=v13_count, v13_filter=v13_filter)

        # Train
        imgsz = cfg.get("imgsz", 640)
        base_model = cfg.get("base_model", V13_S2)
        extra_kwargs = cfg.get("extra_kwargs", None)
        model_path = quick_train(name, data_yaml, base_model=base_model,
                                 imgsz=imgsz, extra_kwargs=extra_kwargs)

        # Eval
        image_size = imgsz if imgsz > 640 else None
        f1, c0, c1, preds, img_sizes = sahi_eval(model_path, name, image_size=image_size)
        elapsed = time.time() - t0

        results[name] = {"f1": f1, "c0": c0, "c1": c1, "time": elapsed,
                         "preds": preds, "img_sizes": img_sizes}
        print(f"  >> {name}: F1={f1:.3f} @c0={c0},c1={c1} ({elapsed/60:.1f}min)")

    return results


def run_phase_b(phase_a_results=None):
    """Run Phase B: ensemble experiments using existing models + v16"""
    from PIL import Image

    # Load val GT and img sizes
    all_imgs = sorted(f for f in os.listdir(VAL_IMG) if f.endswith(".jpg"))
    all_gt, img_sizes = {}, {}
    for f in all_imgs:
        img = Image.open(os.path.join(VAL_IMG, f))
        img_sizes[f] = img.size
        gt = []
        lbl = os.path.join(VAL_LBL, f.replace(".jpg", ".txt"))
        if os.path.exists(lbl):
            with open(lbl) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        iw, ih = img.size
                        gt.append((cls, (cx-w/2)*iw, (cy-h/2)*ih, (cx+w/2)*iw, (cy+h/2)*ih))
        all_gt[f] = gt

    # Run SAHI for each model (cache)
    print(f"\n{'='*60}")
    print("Phase B: 모델별 SAHI 추론")
    print(f"{'='*60}")

    model_preds = {}
    models_to_eval = {"v16": V16_BEST}
    models_to_eval.update(EXISTING_MODELS)

    for mname, mpath in models_to_eval.items():
        print(f"\n  [{mname}] SAHI inference...")
        isz = 1280 if mname == "v3" else None
        _, _, _, preds, _ = sahi_eval(mpath, f"ens_{mname}", image_size=isz)
        model_preds[mname] = preds

    # Ensemble combinations
    print(f"\n{'='*60}")
    print("Phase B: 앙상블 WBF 평가")
    print(f"{'='*60}")

    ens_configs = [
        ("B1_v16+v2", ["v16", "v2"], None),
        ("B2_v16+v3", ["v16", "v3"], None),
        ("B3_v16+v5", ["v16", "v5"], None),
        ("B4_v16+v2+v3", ["v16", "v2", "v3"], None),
        ("B5_v16+v3+v5", ["v16", "v3", "v5"], None),
    ]

    # If Phase A best exists, add it
    if phase_a_results:
        best_a = max(phase_a_results.items(), key=lambda x: x[1]["f1"])
        if best_a[1]["preds"]:
            model_preds[f"best_A({best_a[0]})"] = best_a[1]["preds"]
            ens_configs.append((f"B6_v16+{best_a[0]}", ["v16", f"best_A({best_a[0]})"], None))

    results = {}
    for label, mnames, weights in ens_configs:
        plist = [model_preds[m] for m in mnames if m in model_preds]
        if len(plist) < 2:
            print(f"  {label}: SKIP (model not available)")
            continue
        ens_preds = wbf_ensemble(plist, img_sizes, iou_thr=0.4, weights=weights)
        f1, c0, c1 = eval_preds(all_gt, ens_preds, all_imgs)
        results[label] = {"f1": f1, "c0": c0, "c1": c1}
        print(f"  {label}: F1={f1:.3f} @c0={c0},c1={c1}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["A", "B", "all"])
    parser.add_argument("--exp", default=None, help="Comma-separated experiment names (e.g. A1_no_v13,A4_imgsz_1280)")
    args = parser.parse_args()

    selected = set(args.exp.split(",")) if args.exp else None

    all_results = {}
    phase_a_results = None

    if args.phase in ("A", "all"):
        phase_a_results = run_phase_a(selected)
        for k, v in phase_a_results.items():
            all_results[k] = {"f1": v["f1"], "c0": v["c0"], "c1": v["c1"],
                              "time": v.get("time", 0)}

    if args.phase in ("B", "all"):
        phase_b_results = run_phase_b(phase_a_results)
        all_results.update(phase_b_results)

    # Summary
    print(f"\n{'='*60}")
    print("전체 실험 결과 순위")
    print(f"{'='*60}")
    print(f"  {'Baseline v16':30s} F1=0.885")
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["f1"]):
        delta = r["f1"] - 0.885
        marker = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"  {name:30s} F1={r['f1']:.3f} ({marker}{abs(delta):.3f}) @c0={r['c0']},c1={r['c1']}")

    # Save
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "preds"}
                 for k, v in all_results.items()}
    with open(RESULTS_FILE, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {RESULTS_FILE}")
