#!/usr/bin/env python3
"""v24 완성형 모델 데이터셋 준비

3-class: person_with_helmet(0), person_without_helmet(1), fallen(2)

구성:
  - 현장 CCTV helmet:  ~4,400 (v19 non-S2)
  - AIHub helmet:      ~2,000 (/data/helmet/, class 반전 매핑)
  - Fallen:            ~2,000 (/data/unified_safety_all/, class 4→2)
  - Background neg:    ~2,500 (/data/aihub_data/helmet_negative_10k_v3/)
  - Hard negative:      ~100  (기존 오탐 이미지)
  ──────────────────────────────
  합계                ~11,000

Val:
  - Helmet val:  605장 (기존 3k+extra)
  - Fallen val: ~100장 (unified_safety_all/valid/)
"""
import os, sys, shutil, random, hashlib, json
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ── 경로 설정 ──
OUT_DIR = "/home/lay/hoban/datasets_go3k_v24"
TRAIN_IMG = os.path.join(OUT_DIR, "train", "images")
TRAIN_LBL = os.path.join(OUT_DIR, "train", "labels")
VAL_IMG = os.path.join(OUT_DIR, "valid", "images")
VAL_LBL = os.path.join(OUT_DIR, "valid", "labels")

# 소스
V19_TRAIN_IMG = "/home/lay/hoban/datasets_go3k_v19/train/images"
V19_TRAIN_LBL = "/home/lay/hoban/datasets_go3k_v19/train/labels"
HELMET_21K_ON_IMG = "/data/aihub_data/helmet_21k/helmet_wearing/images"
HELMET_21K_ON_LBL = "/data/aihub_data/helmet_21k/helmet_wearing/labels"
HELMET_21K_OFF_IMG = "/data/aihub_data/helmet_21k/helmet_not_wearing/images"
HELMET_21K_OFF_LBL = "/data/aihub_data/helmet_21k/helmet_not_wearing/labels"
UNIFIED_TRAIN_IMG = "/data/unified_safety_all/train/images"
UNIFIED_TRAIN_LBL = "/data/unified_safety_all/train/labels"
UNIFIED_VAL_IMG = "/data/unified_safety_all/valid/images"
UNIFIED_VAL_LBL = "/data/unified_safety_all/valid/labels"
NEG_IMG = "/data/aihub_data/helmet_negative_10k_v3/images"
NEG_LBL = "/data/aihub_data/helmet_negative_10k_v3/labels"

# 기존 val
EXISTING_VAL_IMG = "/home/lay/hoban/datasets/3k_finetune/val/images"
EXISTING_VAL_LBL = "/home/lay/hoban/datasets/3k_finetune/val/labels"
EXTRA_VAL_IMG = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/images"
EXTRA_VAL_LBL = "/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/labels"

# 설정
AIHUB_HELMET_COUNT = 2000
FALLEN_COUNT = 2000
FALLEN_SMALL_RATIO = 0.5  # area < 0.10
NEG_COUNT = 2500
FALLEN_VAL_COUNT = 100


def ensure_dirs():
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        os.makedirs(d, exist_ok=True)


def copy_file(src, dst):
    """심볼릭 링크면 실제 파일을 복사"""
    real_src = os.path.realpath(src)
    if os.path.exists(real_src):
        shutil.copy2(real_src, dst)
        return True
    return False


def get_label_path(img_path, lbl_dir):
    """이미지 경로에서 라벨 경로 생성"""
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(lbl_dir, base + ".txt")


def parse_labels(lbl_path):
    """YOLO 라벨 파싱 → [(cls, cx, cy, w, h), ...]"""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls, cx, cy, w, h))
    return boxes


def write_labels(lbl_path, boxes):
    """YOLO 라벨 쓰기"""
    with open(lbl_path, "w") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def roboflow_base_name(fname):
    """Roboflow augmentation에서 원본 이름 추출
    예: image_name.rf.abc123.jpg → image_name"""
    base = os.path.splitext(fname)[0]
    if ".rf." in base:
        return base.split(".rf.")[0]
    return base


# ── Phase 1: 현장 CCTV 데이터 ──
def phase1_onsite():
    print("\n[Phase 1] 현장 CCTV 데이터 (v19 non-S2)...")
    count = 0
    if not os.path.isdir(V19_TRAIN_IMG):
        print(f"  경고: {V19_TRAIN_IMG} 없음")
        return count

    for fname in sorted(os.listdir(V19_TRAIN_IMG)):
        if not fname.endswith(".jpg"):
            continue
        # S2- prefix = AIHub 외부 데이터 → 제외
        if fname.startswith("S2"):
            continue

        src_img = os.path.join(V19_TRAIN_IMG, fname)
        src_lbl = get_label_path(src_img, V19_TRAIN_LBL)

        dst_img = os.path.join(TRAIN_IMG, fname)
        dst_lbl = os.path.join(TRAIN_LBL, os.path.splitext(fname)[0] + ".txt")

        if copy_file(src_img, dst_img):
            if os.path.exists(os.path.realpath(src_lbl)):
                # 라벨은 그대로 (class 0, 1 동일)
                copy_file(src_lbl, dst_lbl)
            else:
                # 빈 라벨
                open(dst_lbl, "w").close()
            count += 1

    print(f"  → {count}장 복사")
    return count


# ── Phase 2: AIHub helmet 외부 (helmet_21k, JSON→YOLO) ──
def parse_aihub_json(json_path):
    """AIHub JSON 라벨 → [(our_class, cx, cy, w, h), ...]
    AIHub class "07"=helmet_on → our 0, "08"=helmet_off → our 1
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return [], (0, 0)

    img_info = data.get("image", {})
    res = img_info.get("resolution", [1920, 1080])
    img_w, img_h = res[0], res[1]

    boxes = []
    for ann in data.get("annotations", []):
        cls_code = ann.get("class", "")
        box = ann.get("box", [])
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        if cls_code == "07":  # helmet_on → our class 0
            boxes.append((0, cx, cy, w, h))
        elif cls_code == "08":  # helmet_off → our class 1
            boxes.append((1, cx, cy, w, h))

    return boxes, (img_w, img_h)


def phase2_aihub_helmet():
    print(f"\n[Phase 2] AIHub helmet_21k 외부 ({AIHUB_HELMET_COUNT}장)...")

    # helmet_wearing (→ class 0) + helmet_not_wearing (→ class 1)
    sources = [
        (HELMET_21K_ON_IMG, HELMET_21K_ON_LBL, "on"),
        (HELMET_21K_OFF_IMG, HELMET_21K_OFF_LBL, "off"),
    ]

    on_files = []
    off_files = []

    for img_dir, lbl_dir, cat in sources:
        if not os.path.isdir(img_dir):
            print(f"  경고: {img_dir} 없음")
            continue

        all_imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        print(f"  {cat}: {len(all_imgs)}장")

        for fname in all_imgs:
            json_name = os.path.splitext(fname)[0] + ".json"
            json_path = os.path.join(lbl_dir, json_name)
            boxes, (img_w, img_h) = parse_aihub_json(json_path)
            if not boxes:
                continue

            # bbox area 필터: 0.001 ~ 0.15 (CCTV 중소형)
            areas = [w * h for _, _, _, w, h in boxes]
            max_area = max(areas)
            if max_area > 0.15 or max_area < 0.001:
                continue

            entry = (fname, img_dir, json_path, boxes)
            if cat == "on":
                on_files.append(entry)
            else:
                off_files.append(entry)

    print(f"  필터 후: on={len(on_files)}, off={len(off_files)}")

    # 선별: on 70%, off 30%
    n_on = min(int(AIHUB_HELMET_COUNT * 0.7), len(on_files))
    n_off = min(AIHUB_HELMET_COUNT - n_on, len(off_files))
    if n_off < AIHUB_HELMET_COUNT - n_on:
        n_on = min(AIHUB_HELMET_COUNT - n_off, len(on_files))

    random.shuffle(on_files)
    random.shuffle(off_files)
    selected = on_files[:n_on] + off_files[:n_off]

    count = 0
    for fname, img_dir, json_path, boxes in selected:
        src_img = os.path.join(img_dir, fname)
        dst_name = f"aihub_{fname}"
        dst_img = os.path.join(TRAIN_IMG, dst_name)
        dst_lbl = os.path.join(TRAIN_LBL, f"aihub_{os.path.splitext(fname)[0]}.txt")

        if not copy_file(src_img, dst_img):
            continue

        write_labels(dst_lbl, boxes)
        count += 1

    print(f"  → {count}장 (on={n_on}, off={n_off})")
    return count


# ── Phase 3: Fallen ──
def phase3_fallen():
    print(f"\n[Phase 3] Fallen 데이터 ({FALLEN_COUNT}장)...")
    if not os.path.isdir(UNIFIED_TRAIN_LBL):
        print(f"  경고: {UNIFIED_TRAIN_LBL} 없음")
        return 0

    # fallen 이미지 수집 + area 분류
    small_files = []   # area < 0.10
    medium_files = []  # 0.10 <= area < 0.30

    all_labels = sorted(os.listdir(UNIFIED_TRAIN_LBL))
    print(f"  전체 라벨: {len(all_labels)}개, 스캔 중...")

    # Roboflow 중복 제거용
    seen_bases = set()

    for lbl_name in all_labels:
        if not lbl_name.endswith(".txt"):
            continue

        lbl_path = os.path.join(UNIFIED_TRAIN_LBL, lbl_name)
        boxes = parse_labels(lbl_path)

        # class 4 (fallen) 포함 여부
        fallen_boxes = [(c, cx, cy, w, h) for c, cx, cy, w, h in boxes if c == 4]
        if not fallen_boxes:
            continue

        # Roboflow 중복 제거
        base = roboflow_base_name(lbl_name)
        if base in seen_bases:
            continue
        seen_bases.add(base)

        # fallen bbox area 계산
        max_area = max(w * h for _, _, _, w, h in fallen_boxes)

        img_name = os.path.splitext(lbl_name)[0] + ".jpg"
        img_path = os.path.join(UNIFIED_TRAIN_IMG, img_name)
        if not os.path.exists(img_path):
            # .png 시도
            img_name = os.path.splitext(lbl_name)[0] + ".png"
            img_path = os.path.join(UNIFIED_TRAIN_IMG, img_name)
            if not os.path.exists(img_path):
                continue

        entry = (img_name, lbl_name, max_area)
        if max_area < 0.10:
            small_files.append(entry)
        elif max_area < 0.30:
            medium_files.append(entry)

    print(f"  중복제거 후: small={len(small_files)}, medium={len(medium_files)}")

    # 선별
    n_small = min(int(FALLEN_COUNT * FALLEN_SMALL_RATIO), len(small_files))
    n_medium = min(FALLEN_COUNT - n_small, len(medium_files))
    # 부족하면 다른 쪽에서 보충
    if n_medium < FALLEN_COUNT - n_small:
        n_small = min(FALLEN_COUNT - n_medium, len(small_files))

    random.shuffle(small_files)
    random.shuffle(medium_files)
    selected = small_files[:n_small] + medium_files[:n_medium]

    count = 0
    for img_name, lbl_name, _ in selected:
        src_img = os.path.join(UNIFIED_TRAIN_IMG, img_name)
        src_lbl = os.path.join(UNIFIED_TRAIN_LBL, lbl_name)

        # prefix 추가
        dst_img_name = f"fallen_{img_name}"
        dst_lbl_name = f"fallen_{os.path.splitext(lbl_name)[0]}.txt"
        dst_img = os.path.join(TRAIN_IMG, dst_img_name)
        dst_lbl = os.path.join(TRAIN_LBL, dst_lbl_name)

        if not copy_file(src_img, dst_img):
            continue

        # class 매핑: 4→2, 나머지 무시
        boxes = parse_labels(src_lbl)
        remapped = []
        for cls, cx, cy, w, h in boxes:
            if cls == 4:
                remapped.append((2, cx, cy, w, h))
            # cls 0,1 (helmet) → 매핑하여 포함
            elif cls == 0:  # person_without_helmet → 1
                remapped.append((1, cx, cy, w, h))
            elif cls == 1:  # person_wearing_helmet → 0
                remapped.append((0, cx, cy, w, h))
            # cls 2,3,5,6,7 → 무시

        write_labels(dst_lbl, remapped)
        count += 1

    print(f"  → {count}장 (small={n_small}, medium={n_medium})")
    return count


# ── Phase 4: Background negative ──
def phase4_negative():
    print(f"\n[Phase 4] Background negative ({NEG_COUNT}장)...")
    if not os.path.isdir(NEG_IMG):
        print(f"  경고: {NEG_IMG} 없음")
        return 0

    all_files = sorted(f for f in os.listdir(NEG_IMG) if f.endswith(".jpg"))
    print(f"  전체: {len(all_files)}장")

    random.shuffle(all_files)
    selected = all_files[:NEG_COUNT]

    count = 0
    for fname in selected:
        src_img = os.path.join(NEG_IMG, fname)
        dst_name = f"neg_{fname}"
        dst_img = os.path.join(TRAIN_IMG, dst_name)
        dst_lbl = os.path.join(TRAIN_LBL, f"neg_{os.path.splitext(fname)[0]}.txt")

        if copy_file(src_img, dst_img):
            # 빈 라벨
            open(dst_lbl, "w").close()
            count += 1

    print(f"  → {count}장")
    return count


# ── Phase 5: Val ──
def phase5_val():
    print(f"\n[Phase 5] Val 데이터 구성...")

    # A. 기존 helmet val (605장)
    helmet_count = 0
    for val_img_dir, val_lbl_dir in [
        (EXISTING_VAL_IMG, EXISTING_VAL_LBL),
        (EXTRA_VAL_IMG, EXTRA_VAL_LBL),
    ]:
        if not os.path.isdir(val_img_dir):
            print(f"  경고: {val_img_dir} 없음")
            continue
        for fname in sorted(os.listdir(val_img_dir)):
            if not fname.endswith(".jpg"):
                continue
            dst_img = os.path.join(VAL_IMG, fname)
            if os.path.exists(dst_img):
                continue  # 중복 스킵

            src_img = os.path.join(val_img_dir, fname)
            src_lbl = os.path.join(val_lbl_dir, fname.replace(".jpg", ".txt"))

            if copy_file(src_img, dst_img):
                dst_lbl = os.path.join(VAL_LBL, fname.replace(".jpg", ".txt"))
                if os.path.exists(src_lbl):
                    copy_file(src_lbl, dst_lbl)
                else:
                    open(dst_lbl, "w").close()
                helmet_count += 1

    # B. Fallen val (100장)
    fallen_val_count = 0
    if os.path.isdir(UNIFIED_VAL_LBL):
        fallen_val_candidates = []
        seen_bases = set()

        for lbl_name in sorted(os.listdir(UNIFIED_VAL_LBL)):
            if not lbl_name.endswith(".txt"):
                continue
            lbl_path = os.path.join(UNIFIED_VAL_LBL, lbl_name)
            boxes = parse_labels(lbl_path)
            fallen_boxes = [b for b in boxes if b[0] == 4]
            if not fallen_boxes:
                continue

            base = roboflow_base_name(lbl_name)
            if base in seen_bases:
                continue
            seen_bases.add(base)

            max_area = max(w * h for _, _, _, w, h in fallen_boxes)
            if max_area < 0.30:
                img_name = os.path.splitext(lbl_name)[0] + ".jpg"
                img_path = os.path.join(UNIFIED_VAL_IMG, img_name)
                if not os.path.exists(img_path):
                    img_name = os.path.splitext(lbl_name)[0] + ".png"
                    img_path = os.path.join(UNIFIED_VAL_IMG, img_name)
                if os.path.exists(img_path):
                    fallen_val_candidates.append((img_name, lbl_name))

        random.shuffle(fallen_val_candidates)
        for img_name, lbl_name in fallen_val_candidates[:FALLEN_VAL_COUNT]:
            src_img = os.path.join(UNIFIED_VAL_IMG, img_name)
            dst_img_name = f"fallen_{img_name}"
            dst_img = os.path.join(VAL_IMG, dst_img_name)

            if not copy_file(src_img, dst_img):
                continue

            src_lbl = os.path.join(UNIFIED_VAL_LBL, lbl_name)
            boxes = parse_labels(src_lbl)
            remapped = []
            for cls, cx, cy, w, h in boxes:
                if cls == 4:
                    remapped.append((2, cx, cy, w, h))
                elif cls == 0:
                    remapped.append((1, cx, cy, w, h))
                elif cls == 1:
                    remapped.append((0, cx, cy, w, h))

            dst_lbl = os.path.join(VAL_LBL, f"fallen_{os.path.splitext(lbl_name)[0]}.txt")
            write_labels(dst_lbl, remapped)
            fallen_val_count += 1

    print(f"  → helmet val: {helmet_count}장, fallen val: {fallen_val_count}장")
    return helmet_count, fallen_val_count


# ── Phase 6: Train-Val 중복 제거 ──
def phase6_dedup():
    print(f"\n[Phase 6] Train-Val 중복 제거...")
    val_names = set()
    for f in os.listdir(VAL_IMG):
        if f.endswith((".jpg", ".png")):
            val_names.add(os.path.splitext(f)[0])

    removed = 0
    for f in list(os.listdir(TRAIN_IMG)):
        base = os.path.splitext(f)[0]
        if base in val_names:
            os.remove(os.path.join(TRAIN_IMG, f))
            lbl = os.path.join(TRAIN_LBL, base + ".txt")
            if os.path.exists(lbl):
                os.remove(lbl)
            removed += 1

    print(f"  → {removed}장 제거")
    return removed


# ── Phase 7: 통계 + data.yaml ──
def phase7_finalize():
    print(f"\n[Phase 7] 통계 + data.yaml 생성...")

    # Train 통계
    train_imgs = [f for f in os.listdir(TRAIN_IMG) if f.endswith((".jpg", ".png"))]
    val_imgs = [f for f in os.listdir(VAL_IMG) if f.endswith((".jpg", ".png"))]

    # Class 분포
    cls_count = defaultdict(int)
    empty_count = 0
    for lbl_name in os.listdir(TRAIN_LBL):
        if not lbl_name.endswith(".txt"):
            continue
        lbl_path = os.path.join(TRAIN_LBL, lbl_name)
        boxes = parse_labels(lbl_path)
        if not boxes:
            empty_count += 1
        for cls, _, _, _, _ in boxes:
            cls_count[cls] += 1

    cls_count_val = defaultdict(int)
    for lbl_name in os.listdir(VAL_LBL):
        if not lbl_name.endswith(".txt"):
            continue
        boxes = parse_labels(os.path.join(VAL_LBL, lbl_name))
        for cls, _, _, _, _ in boxes:
            cls_count_val[cls] += 1

    print(f"\n  Train: {len(train_imgs)}장")
    print(f"    helmet_on (0):  {cls_count[0]:,} bbox")
    print(f"    helmet_off (1): {cls_count[1]:,} bbox")
    print(f"    fallen (2):     {cls_count[2]:,} bbox")
    print(f"    empty (neg):    {empty_count}")

    print(f"\n  Val: {len(val_imgs)}장")
    print(f"    helmet_on (0):  {cls_count_val[0]:,} bbox")
    print(f"    helmet_off (1): {cls_count_val[1]:,} bbox")
    print(f"    fallen (2):     {cls_count_val[2]:,} bbox")

    # Train-Val 중복 검사
    train_names = set(os.path.splitext(f)[0] for f in train_imgs)
    val_names = set(os.path.splitext(f)[0] for f in val_imgs)
    overlap = train_names & val_names
    if overlap:
        print(f"\n  경고: train-val 중복 {len(overlap)}개!")
        for name in list(overlap)[:5]:
            print(f"    {name}")
    else:
        print(f"\n  train-val 중복: 없음 ✓")

    # data.yaml
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: 3\n")
        f.write("names:\n")
        f.write("  0: person_with_helmet\n")
        f.write("  1: person_without_helmet\n")
        f.write("  2: fallen\n")

    print(f"\n  data.yaml: {yaml_path}")
    return len(train_imgs), len(val_imgs)


if __name__ == "__main__":
    print("=" * 60)
    print("  v24 데이터셋 준비 (3-class)")
    print("=" * 60)

    if os.path.exists(OUT_DIR):
        print(f"\n기존 {OUT_DIR} 삭제 중...")
        shutil.rmtree(OUT_DIR)

    ensure_dirs()

    n1 = phase1_onsite()
    n2 = phase2_aihub_helmet()
    n3 = phase3_fallen()
    n4 = phase4_negative()
    h_val, f_val = phase5_val()
    n_dup = phase6_dedup()
    n_train, n_val = phase7_finalize()

    print(f"\n{'=' * 60}")
    print(f"  완료!")
    print(f"  Train: {n_train}장 (onsite={n1}, aihub={n2}, fallen={n3}, neg={n4})")
    print(f"  Val: {n_val}장 (helmet={h_val}, fallen={f_val})")
    print(f"  출력: {OUT_DIR}")
    print(f"{'=' * 60}")
