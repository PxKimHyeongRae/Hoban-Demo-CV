"""
v8 데이터셋 빌드: helmet_pool(헬멧) + coco_person(사람) + fallen_pool(쓰러짐)
- 4클래스: helmet_o(0), helmet_x(1), person(2), fallen(3)
- 클래스별 bbox 10,000개 기준 균형 (helmet_x는 최대한)
"""
import shutil
import random
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

HELMET_POOL = Path(r"D:\task\hoban\helmet_pool")
COCO_PERSON = Path(r"D:\task\hoban\dataset\coco_person")
FALLEN_POOL = Path(r"D:\task\hoban\fallen_pool")
OUT = Path(r"D:\task\hoban\datasets_v8")
TARGET_BBOX = 10000
VALID_RATIO = 0.2


def select_by_bbox(file_class_map, target_cls, target_bbox):
    """이미지를 셔플 후 target_cls의 bbox가 target_bbox에 도달할 때까지 선택"""
    candidates = list(file_class_map.items())
    random.shuffle(candidates)
    selected = []
    bbox_sum = 0
    for stem, cls_counts in candidates:
        if target_cls not in cls_counts:
            continue
        selected.append(stem)
        bbox_sum += cls_counts[target_cls]
        if bbox_sum >= target_bbox:
            break
    return selected, bbox_sum


def scan_labels(lbl_dir, img_dir, extensions=None):
    """라벨 디렉토리를 스캔하여 {stem: Counter({cls: count})} 반환"""
    file_cls = {}
    for lf in lbl_dir.iterdir():
        if lf.suffix != ".txt":
            continue
        # 이미지 존재 확인
        img_found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            if (img_dir / f"{lf.stem}{ext}").exists():
                img_found = True
                break
        if not img_found:
            continue

        cls_counts = Counter()
        with open(lf) as f:
            for line in f:
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1
        if cls_counts:
            file_cls[lf.stem] = cls_counts
    return file_cls


def copy_files(stems, img_dir, lbl_dir, out_split, prefix="", remap=None):
    """파일 복사. remap: {원본cls: 새cls} 딕셔너리"""
    copied = 0
    for stem in stems:
        # 이미지 찾기
        img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img = p
                break
        lf = lbl_dir / f"{stem}.txt"
        if not img or not lf.exists():
            continue

        out_name = f"{prefix}{stem}" if prefix else stem

        # 라벨 복사 (remap 적용)
        if remap:
            with open(lf) as f:
                lines = []
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        old_cls = int(parts[0])
                        if old_cls in remap:
                            parts[0] = str(remap[old_cls])
                            lines.append(" ".join(parts))
            with open(OUT / out_split / "labels" / f"{out_name}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")
        else:
            shutil.copy2(lf, OUT / out_split / "labels" / f"{out_name}.txt")

        shutil.copy2(img, OUT / out_split / "images" / f"{out_name}{img.suffix}")
        copied += 1
    return copied


def main():
    random.seed(42)

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1. Helmet (cls 0, 1) from helmet_pool
    # ========================================
    print("=== 1. Helmet 데이터 (helmet_pool) ===")
    helmet_cls = scan_labels(HELMET_POOL / "labels", HELMET_POOL / "images")
    print(f"  총 helmet 이미지: {len(helmet_cls)}장")

    # train/valid 분리
    all_stems = list(helmet_cls.keys())
    random.shuffle(all_stems)
    valid_count = int(len(all_stems) * VALID_RATIO)
    valid_stems = set(all_stems[:valid_count])
    train_stems = set(all_stems[valid_count:])

    # train: class 0, class 1 각각 bbox 기준 선택
    train_cls = {s: c for s, c in helmet_cls.items() if s in train_stems}
    sel_0, bbox_0 = select_by_bbox(train_cls, 0, TARGET_BBOX)
    sel_1, bbox_1 = select_by_bbox(train_cls, 1, TARGET_BBOX)
    train_selected = set(sel_0) | set(sel_1)

    copied = copy_files(train_selected, HELMET_POOL / "images", HELMET_POOL / "labels", "train")
    print(f"  train: {copied}장 (cls0: {len(sel_0)}장/{bbox_0} bbox, cls1: {len(sel_1)}장/{bbox_1} bbox)")

    # valid: class 0, class 1 각각
    valid_cls = {s: c for s, c in helmet_cls.items() if s in valid_stems}
    vsel_0, vbbox_0 = select_by_bbox(valid_cls, 0, int(TARGET_BBOX * VALID_RATIO))
    vsel_1, vbbox_1 = select_by_bbox(valid_cls, 1, int(TARGET_BBOX * VALID_RATIO))
    valid_selected = set(vsel_0) | set(vsel_1)

    copied = copy_files(valid_selected, HELMET_POOL / "images", HELMET_POOL / "labels", "valid")
    print(f"  valid: {copied}장 (cls0: {len(vsel_0)}장/{vbbox_0} bbox, cls1: {len(vsel_1)}장/{vbbox_1} bbox)")

    # ========================================
    # 2. Person (cls 2) from coco_person
    # ========================================
    print("\n=== 2. Person 데이터 (coco_person) ===")
    coco_labels = list((COCO_PERSON / "labels").glob("*.txt"))
    random.shuffle(coco_labels)

    for split_name, target in [("train", TARGET_BBOX), ("valid", int(TARGET_BBOX * VALID_RATIO))]:
        copied = 0
        bbox_sum = 0
        for lf in coco_labels:
            if bbox_sum >= target:
                break
            stem = lf.stem
            img = COCO_PERSON / "images" / f"{stem}.jpg"
            if not img.exists():
                continue
            with open(lf) as f:
                n_bbox = sum(1 for line in f if line.strip())

            shutil.copy2(img, OUT / split_name / "images" / f"coco_{stem}.jpg")
            # coco_person은 이미 cls 2
            shutil.copy2(lf, OUT / split_name / "labels" / f"coco_{stem}.txt")
            bbox_sum += n_bbox
            copied += 1
        coco_labels = coco_labels[copied:]
        print(f"  {split_name}: {copied}장 ({bbox_sum} bbox)")

    # ========================================
    # 3. Fallen (cls 3) from fallen_pool
    # ========================================
    print("\n=== 3. Fallen 데이터 (fallen_pool, 정제 후) ===")
    fallen_cls = scan_labels(FALLEN_POOL / "labels", FALLEN_POOL / "images")
    print(f"  총 fallen 이미지: {len(fallen_cls)}장")

    # train/valid 분리
    all_fallen = list(fallen_cls.keys())
    random.shuffle(all_fallen)
    fv_count = int(len(all_fallen) * VALID_RATIO)
    fallen_valid = set(all_fallen[:fv_count])
    fallen_train = set(all_fallen[fv_count:])

    # train
    train_f_cls = {s: c for s, c in fallen_cls.items() if s in fallen_train}
    fsel, fbbox = select_by_bbox(train_f_cls, 0, TARGET_BBOX)
    # remap: fallen_pool의 cls 0 → 최종 cls 3
    copied = copy_files(fsel, FALLEN_POOL / "images", FALLEN_POOL / "labels", "train",
                        prefix="fallen_", remap={0: 3})
    print(f"  train: {copied}장 ({fbbox} bbox)")

    # valid
    valid_f_cls = {s: c for s, c in fallen_cls.items() if s in fallen_valid}
    vfsel, vfbbox = select_by_bbox(valid_f_cls, 0, int(TARGET_BBOX * VALID_RATIO))
    copied = copy_files(vfsel, FALLEN_POOL / "images", FALLEN_POOL / "labels", "valid",
                        prefix="fallen_", remap={0: 3})
    print(f"  valid: {copied}장 ({vfbbox} bbox)")

    # ========================================
    # 4. 최종 클래스 분포
    # ========================================
    print("\n=== 최종 클래스 분포 ===")
    names = {0: "helmet_o", 1: "helmet_x", 2: "person", 3: "fallen"}
    for sn in ["train", "valid"]:
        counter = Counter()
        total = 0
        for lf in (OUT / sn / "labels").iterdir():
            if lf.suffix != ".txt":
                continue
            total += 1
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        counter[int(line.split()[0])] += 1
        print(f"  {sn} ({total}장):")
        for cls in sorted(counter):
            print(f"    {cls} ({names.get(cls, '?')}): {counter[cls]} bbox")

    # ========================================
    # 5. data.yaml
    # ========================================
    yaml_content = """path: datasets_v8
train: train/images
val: valid/images
nc: 4
names:
  0: person_with_helmet
  1: person_without_helmet
  2: person
  3: fallen
"""
    with open(OUT / "data.yaml", "w") as f:
        f.write(yaml_content)
    print("\ndone!")


if __name__ == "__main__":
    freeze_support()
    main()
