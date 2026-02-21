#!/usr/bin/env python3
"""
go3k v23: 현장 데이터만 학습 (외부 AIHub S2-* 제외)

가설: v16의 76% 외부 데이터(S2-*)가 도메인 불일치를 일으켜 현장 성능을 저해할 수 있음.
     val은 100% 현장이므로, train도 현장만 사용하면 도메인 매칭이 완벽해짐.

데이터:
  v16 현장 (cam1+cam2): 2,564
  v19 추가 (helmet_off + neg): 1,906
  → 총 ~4,470 train (100% 현장)
  val: 641 (기존 v16 val, 100% 현장)

모델: yolo26m.pt (COCO pretrained, v17과 동일 접근)
학습: 100ep, SGD lr0=0.005, patience=20

사용법:
  python train_go3k_v23_onsite.py --prepare    # 데이터셋 준비
  python train_go3k_v23_onsite.py              # 학습 시작
  python train_go3k_v23_onsite.py --resume     # 이어서 학습
"""
import argparse
import os
import glob

HOBAN = "/home/lay/hoban"
V16_DATASET = f"{HOBAN}/datasets_go3k_v16"
V19_DATASET = f"{HOBAN}/datasets_go3k_v19"
V23_DATASET = f"{HOBAN}/datasets_go3k_v23"
MODEL = f"{HOBAN}/yolo26m.pt"


def is_onsite(fname):
    """현장 데이터인지 확인 (cam1/cam2)"""
    return fname.startswith("cam1") or fname.startswith("cam2")


def prepare_dataset():
    print("=" * 60)
    print("  v23 데이터셋 준비: 현장(cam) 데이터만")
    print("=" * 60)

    train_img = f"{V23_DATASET}/train/images"
    train_lbl = f"{V23_DATASET}/train/labels"
    val_img = f"{V23_DATASET}/valid/images"
    val_lbl = f"{V23_DATASET}/valid/labels"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        os.makedirs(d, exist_ok=True)

    # 1. v16 train에서 현장 데이터만 링크
    print("\n[1] v16 train → 현장(cam) 필터링...")
    v16_train = f"{V16_DATASET}/train/images"
    onsite_count = 0
    external_count = 0
    for fname in sorted(os.listdir(v16_train)):
        if not fname.endswith(".jpg"):
            continue
        if is_onsite(fname):
            src_img = os.path.join(v16_train, fname)
            src_lbl = os.path.join(f"{V16_DATASET}/train/labels", fname.replace(".jpg", ".txt"))
            dst_img = os.path.join(train_img, fname)
            dst_lbl = os.path.join(train_lbl, fname.replace(".jpg", ".txt"))
            if not os.path.exists(dst_img):
                os.symlink(src_img, dst_img)
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                os.symlink(src_lbl, dst_lbl)
            onsite_count += 1
        else:
            external_count += 1
    print(f"  현장: {onsite_count}장, 외부 제외: {external_count}장")

    # 2. v19 추가분 (helmet_off + neg, 모두 현장)
    print("\n[2] v19 추가분 (helmet_off + negative)...")
    existing = set(os.listdir(train_img))
    v19_added = 0

    # v19 train에서 v16에 없는 것 = 추가분
    v19_train = f"{V19_DATASET}/train/images"
    if os.path.isdir(v19_train):
        v16_fnames = set(os.listdir(v16_train))
        for fname in sorted(os.listdir(v19_train)):
            if not fname.endswith(".jpg"):
                continue
            if fname in v16_fnames or fname in existing:
                continue
            # v19 추가분은 모두 현장 (cam) 데이터
            src_img = os.path.join(v19_train, fname)
            src_lbl = os.path.join(f"{V19_DATASET}/train/labels", fname.replace(".jpg", ".txt"))
            dst_img = os.path.join(train_img, fname)
            dst_lbl = os.path.join(train_lbl, fname.replace(".jpg", ".txt"))

            # symlink가 아닌 실제 파일이면 symlink, 아니면 복사
            if os.path.islink(src_img):
                real_src = os.path.realpath(src_img)
                os.symlink(real_src, dst_img)
            else:
                os.symlink(src_img, dst_img)

            if os.path.exists(src_lbl):
                if os.path.islink(src_lbl):
                    real_src = os.path.realpath(src_lbl)
                    os.symlink(real_src, dst_lbl)
                else:
                    os.symlink(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()
            v19_added += 1
    print(f"  추가: {v19_added}장")

    # 3. val (v16 val 그대로, 이미 100% 현장)
    print("\n[3] val 링크 (v16 val 그대로)...")
    v16_val_imgs = glob.glob(f"{V16_DATASET}/valid/images/*.jpg")
    v16_val_lbls = glob.glob(f"{V16_DATASET}/valid/labels/*.txt")
    val_linked = 0
    for src in v16_val_imgs + v16_val_lbls:
        dst = src.replace(V16_DATASET, V23_DATASET)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            val_linked += 1
    print(f"  val 링크: {val_linked}개")

    # 4. data.yaml
    yaml_path = f"{V23_DATASET}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {V23_DATASET}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n\n")
        f.write("nc: 2\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")

    final_train = len([f for f in os.listdir(train_img) if f.endswith(".jpg")])
    final_val = len([f for f in os.listdir(val_img) if f.endswith(".jpg")])
    print(f"\n{'='*60}")
    print(f"  v23 데이터셋 완성 (현장 only)")
    print(f"  Train: {final_train}장 (v16 현장 {onsite_count} + v19 추가 {v19_added})")
    print(f"  Val:   {final_val}장 (100% 현장)")
    print(f"  YAML:  {yaml_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train go3k v23 (on-site only)")
    parser.add_argument("--prepare", action="store_true", help="데이터셋 준비만")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare_dataset()
        return

    project = HOBAN
    name = "hoban_go3k_v23"
    data_yaml = f"{V23_DATASET}/data.yaml"

    if not os.path.exists(data_yaml):
        print("데이터셋 없음. 먼저 --prepare 실행:")
        print("  python train_go3k_v23_onsite.py --prepare")
        return

    from ultralytics import YOLO

    if args.resume:
        ckpt = f"{project}/{name}/weights/last.pt"
        print(f"Resuming from {ckpt}")
        model = YOLO(ckpt)
        model.train(resume=True)
        return

    print(f"=== go3k v23: 현장 데이터만 (COCO pt) ===")
    print(f"  Model: {MODEL} (yolo26m COCO pt)")
    print(f"  Data: {data_yaml}")
    print(f"  imgsz: 1280, batch: {args.batch}")
    print(f"  epochs: {args.epochs}, lr0: 0.005")
    print()

    model = YOLO(MODEL)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=1280,
        batch=args.batch,
        device="0",
        project=project,
        name=name,
        exist_ok=True,

        # SGD (v17 동일)
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        cos_lr=True,

        # Augmentation (v17 동일)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        degrees=5.0,
        fliplr=0.5,
        erasing=0.15,
        close_mosaic=10,

        # Early stopping
        patience=20,
        amp=True,
        workers=4,
        seed=42,
        plots=True,
        save=True,
        val=True,
    )

    print(f"\nDone! Results: {project}/{name}/")


if __name__ == "__main__":
    main()
