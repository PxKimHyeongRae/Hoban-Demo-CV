#!/usr/bin/env python3
"""
v2 데이터 기반 빠른 실험용 축소 데이터셋 생성
- v13: 8K → 2K (랜덤 샘플링)
- go2k: ×8 → ×4 (원본 + x1~x3만)
- valid: 동일 유지
- 심볼릭 링크로 디스크 절약
"""
import os
import random
import yaml
from pathlib import Path

random.seed(42)

SRC = Path("/home/lay/hoban/datasets_go2k_v2")
DST = Path("/home/lay/hoban/datasets_go2k_quick")

# Clean & create
for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
    (DST / sub).mkdir(parents=True, exist_ok=True)

# === Separate go2k vs v13 images ===
train_imgs = sorted(os.listdir(SRC / "train/images"))

go2k_originals = {}  # base_name -> [base, x1, x2, ..., x7]
v13_imgs = []

for fname in train_imgs:
    if fname.startswith("cam"):
        # Extract base name (without _xN suffix)
        stem = Path(fname).stem
        ext = Path(fname).suffix
        if "_x" in stem and stem.rsplit("_x", 1)[1].isdigit():
            base = stem.rsplit("_x", 1)[0]
            idx = int(stem.rsplit("_x", 1)[1])
        else:
            base = stem
            idx = 0  # original
        if base not in go2k_originals:
            go2k_originals[base] = {}
        go2k_originals[base][idx] = fname
    else:
        v13_imgs.append(fname)

print(f"v13 images: {len(v13_imgs)}")
print(f"go2k unique scenes: {len(go2k_originals)}")
print(f"go2k total images: {sum(len(v) for v in go2k_originals.values())}")

# === Sample v13: 2000 ===
v13_sample = random.sample(v13_imgs, min(2000, len(v13_imgs)))
print(f"v13 sampled: {len(v13_sample)}")

# === go2k: keep original + x1~x3 (×4) ===
go2k_selected = []
for base, variants in go2k_originals.items():
    for idx in [0, 1, 2, 3]:  # original, x1, x2, x3
        if idx in variants:
            go2k_selected.append(variants[idx])

print(f"go2k selected (×4): {len(go2k_selected)}")

# === Create symlinks ===
selected_train = v13_sample + go2k_selected
print(f"\nTotal train: {len(selected_train)}")

linked = 0
missing_labels = 0
for fname in selected_train:
    # Image symlink
    src_img = SRC / "train/images" / fname
    dst_img = DST / "train/images" / fname
    if not dst_img.exists():
        os.symlink(src_img, dst_img)

    # Label symlink
    label_name = Path(fname).stem + ".txt"
    src_lbl = SRC / "train/labels" / label_name
    dst_lbl = DST / "train/labels" / label_name
    if src_lbl.exists() and not dst_lbl.exists():
        os.symlink(src_lbl, dst_lbl)
        linked += 1
    elif not src_lbl.exists():
        missing_labels += 1

print(f"Labels linked: {linked}, missing: {missing_labels}")

# === Valid: symlink all ===
valid_linked = 0
for sub in ["images", "labels"]:
    src_dir = SRC / "valid" / sub
    dst_dir = DST / "valid" / sub
    for fname in os.listdir(src_dir):
        src = src_dir / fname
        dst = dst_dir / fname
        if not dst.exists():
            os.symlink(src, dst)
            valid_linked += 1

print(f"Valid linked: {valid_linked}")

# === data.yaml ===
data_yaml = {
    "path": str(DST),
    "train": "train/images",
    "val": "valid/images",
    "nc": 2,
    "names": {0: "person_with_helmet", 1: "person_without_helmet"},
}
with open(DST / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\n=== Quick Dataset Ready ===")
print(f"Train: {len(selected_train)} images")
print(f"Valid: {len(os.listdir(SRC / 'valid/images'))} images")
print(f"Path: {DST}")
print(f"Estimated epoch time: ~55s (vs ~160s)")
