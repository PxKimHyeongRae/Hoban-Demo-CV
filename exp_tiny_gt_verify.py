#!/usr/bin/env python3
"""tiny GT 유효성 검증: FN 47개의 tiny bbox를 크롭+확대하여 시각화"""
import os, sys, cv2, numpy as np

sys.stdout.reconfigure(line_buffering=True)

VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"
OUT_DIR = "/home/lay/hoban/analysis_v19_errors/tiny_gt_crops"
CLASS_NAMES = {0: "helmet_on", 1: "helmet_off"}

os.makedirs(OUT_DIR, exist_ok=True)

# 모든 이미지의 GT 중 tiny 찾기
tiny_count = 0
size_stats = {"tiny(<0.1%)": 0, "small(0.1-0.5%)": 0, "medium(0.5-2%)": 0, "large(>2%)": 0}

for img_dir, lbl_dir in [(VAL_IMG, VAL_LBL), (EXTRA_IMG, EXTRA_LBL)]:
    if not os.path.isdir(img_dir):
        continue
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".jpg"):
            continue
        lbl_path = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                bbox_area = (x2 - x1) * (y2 - y1)
                ratio = bbox_area / img_area

                if ratio < 0.001:
                    size_stats["tiny(<0.1%)"] += 1
                elif ratio < 0.005:
                    size_stats["small(0.1-0.5%)"] += 1
                elif ratio < 0.02:
                    size_stats["medium(0.5-2%)"] += 1
                else:
                    size_stats["large(>2%)"] += 1

                if ratio >= 0.001:
                    continue

                tiny_count += 1
                # 크롭: bbox 주변 3배 확장
                bw, bh = x2 - x1, y2 - y1
                pad = max(bw, bh) * 2
                cx_i, cy_i = (x1 + x2) // 2, (y1 + y2) // 2
                rx1 = max(0, int(cx_i - pad))
                ry1 = max(0, int(cy_i - pad))
                rx2 = min(img_w, int(cx_i + pad))
                ry2 = min(img_h, int(cy_i + pad))

                crop = img[ry1:ry2, rx1:rx2].copy()
                # bbox 표시 (crop 좌표)
                bx1, by1 = x1 - rx1, y1 - ry1
                bx2, by2 = x2 - rx1, y2 - ry1
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 2)

                # 확대 (최소 200x200)
                ch, cw = crop.shape[:2]
                scale = max(200 / cw, 200 / ch, 1)
                if scale > 1:
                    crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                     interpolation=cv2.INTER_NEAREST)

                label = f"{CLASS_NAMES[cls]} {bw}x{bh}px ({ratio*100:.3f}%)"
                cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(crop, fname[:40], (5, crop.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                out_name = f"tiny_{tiny_count:03d}_{CLASS_NAMES[cls]}_{bw}x{bh}_{fname}"
                cv2.imwrite(os.path.join(OUT_DIR, out_name), crop)

print(f"\n=== GT bbox 크기 분포 (전체 {sum(size_stats.values())}개) ===")
for size, cnt in size_stats.items():
    pct = cnt / sum(size_stats.values()) * 100
    print(f"  {size}: {cnt} ({pct:.1f}%)")

print(f"\ntiny GT 크롭: {tiny_count}장 → {OUT_DIR}/")
