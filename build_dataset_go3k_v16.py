#!/usr/bin/env python3
"""
go3k v16 데이터셋 구축

전략:
  - Train: 3k_finetune/train (2,564 CCTV) + v13 랜덤 서브샘플 (보수적 라벨, 일관성)
  - Val:   3k_finetune/val (641 CCTV, leakage 없음)
  - v13은 bbox area 필터링 적용 (CCTV 도메인에 가까운 것만)
  - byte copy 없음, symlink 사용

실행: python build_dataset_v16.py [--v13-count 8000] [--v13-filter]
"""
import os
import sys
import random
import argparse
import shutil
from pathlib import Path


def count_label_stats(label_path):
    """라벨 파일의 bbox 수와 area 통계"""
    if not os.path.exists(label_path):
        return 0, []
    areas = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                w, h = float(parts[3]), float(parts[4])
                areas.append(w * h)
    return len(areas), areas


def filter_v13_by_bbox(v13_label_dir, v13_images, min_area=0.0001, max_area=0.01):
    """v13 이미지 중 CCTV 도메인에 가까운 bbox 크기를 가진 것만 선택"""
    filtered = []
    for img_name in v13_images:
        lbl_name = img_name.replace(".jpg", ".txt")
        lbl_path = os.path.join(v13_label_dir, lbl_name)
        n_bbox, areas = count_label_stats(lbl_path)
        if n_bbox == 0:
            continue
        # 평균 bbox area가 범위 내인 것만
        avg_area = sum(areas) / len(areas)
        if min_area <= avg_area <= max_area:
            filtered.append(img_name)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Build go3k v16 dataset")
    parser.add_argument("--v13-count", type=int, default=8000,
                        help="v13 서브샘플 수 (default: 8000)")
    parser.add_argument("--v13-filter", action="store_true",
                        help="v13 bbox area 필터링 적용")
    parser.add_argument("--no-v13", action="store_true",
                        help="v13 보조 데이터 제외 (CCTV만)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 파일 작업 없이 통계만 출력")
    args = parser.parse_args()

    random.seed(args.seed)

    # Paths
    threeK_base = Path("/home/lay/hoban/datasets/3k_finetune")
    v13_base = Path("/home/lay/hoban/datasets_v13")
    out_base = Path("/home/lay/hoban/datasets_go3k_v16")

    threeK_train_img = threeK_base / "train" / "images"
    threeK_train_lbl = threeK_base / "train" / "labels"
    threeK_val_img = threeK_base / "val" / "images"
    threeK_val_lbl = threeK_base / "val" / "labels"
    v13_train_img = v13_base / "train" / "images"
    v13_train_lbl = v13_base / "train" / "labels"

    # Validate source dirs
    for d in [threeK_train_img, threeK_train_lbl, threeK_val_img, threeK_val_lbl]:
        if not d.exists():
            print(f"ERROR: {d} not found")
            sys.exit(1)

    # 3k train images
    cctv_train = sorted(f for f in os.listdir(threeK_train_img) if f.endswith(".jpg"))
    cctv_val = sorted(f for f in os.listdir(threeK_val_img) if f.endswith(".jpg"))
    print(f"3k train: {len(cctv_train)} images")
    print(f"3k val:   {len(cctv_val)} images")

    # v13 subsample
    v13_selected = []
    if not args.no_v13:
        v13_all = sorted(f for f in os.listdir(v13_train_img) if f.endswith(".jpg"))
        print(f"v13 total: {len(v13_all)} images")

        if args.v13_filter:
            print("v13 bbox area 필터링 중...")
            v13_candidates = filter_v13_by_bbox(
                str(v13_train_lbl), v13_all,
                min_area=0.0001, max_area=0.01)
            print(f"  필터 통과: {len(v13_candidates)} / {len(v13_all)}")
        else:
            v13_candidates = v13_all

        n_sample = min(args.v13_count, len(v13_candidates))
        v13_selected = sorted(random.sample(v13_candidates, n_sample))
        print(f"v13 subsample: {n_sample} images")

    total_train = len(cctv_train) + len(v13_selected)
    print(f"\n=== Dataset Summary ===")
    print(f"Train: {total_train} ({len(cctv_train)} CCTV + {len(v13_selected)} v13)")
    print(f"Val:   {len(cctv_val)}")
    print(f"CCTV 비율: {len(cctv_train)/total_train*100:.1f}%")

    if args.dry_run:
        print("\n[Dry run] 파일 작업 건너뜀")
        return

    # Create output dirs
    out_train_img = out_base / "train" / "images"
    out_train_lbl = out_base / "train" / "labels"
    out_val_img = out_base / "valid" / "images"
    out_val_lbl = out_base / "valid" / "labels"

    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Clear existing symlinks
    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        for f in d.iterdir():
            if f.is_symlink() or f.is_file():
                f.unlink()

    # Symlink 3k train
    linked = 0
    for name in cctv_train:
        src_img = threeK_train_img / name
        src_lbl = threeK_train_lbl / name.replace(".jpg", ".txt")
        (out_train_img / name).symlink_to(src_img)
        if src_lbl.exists():
            (out_train_lbl / name.replace(".jpg", ".txt")).symlink_to(src_lbl)
        linked += 1
    print(f"3k train symlinked: {linked}")

    # Symlink v13 subsample
    linked_v13 = 0
    for name in v13_selected:
        src_img = v13_train_img / name
        src_lbl = v13_train_lbl / name.replace(".jpg", ".txt")
        (out_train_img / name).symlink_to(src_img)
        if src_lbl.exists():
            (out_train_lbl / name.replace(".jpg", ".txt")).symlink_to(src_lbl)
        linked_v13 += 1
    print(f"v13 subsample symlinked: {linked_v13}")

    # Symlink 3k val
    linked_val = 0
    for name in cctv_val:
        src_img = threeK_val_img / name
        src_lbl = threeK_val_lbl / name.replace(".jpg", ".txt")
        (out_val_img / name).symlink_to(src_img)
        if src_lbl.exists():
            (out_val_lbl / name.replace(".jpg", ".txt")).symlink_to(src_lbl)
        linked_val += 1
    print(f"3k val symlinked: {linked_val}")

    # data.yaml
    yaml_content = f"""path: {out_base}
train: train/images
val: valid/images

nc: 2
names:
  0: person_with_helmet
  1: person_without_helmet
"""
    (out_base / "data.yaml").write_text(yaml_content)
    print(f"\ndata.yaml 생성: {out_base / 'data.yaml'}")
    print(f"\n완료! 데이터셋: {out_base}")


if __name__ == "__main__":
    main()
