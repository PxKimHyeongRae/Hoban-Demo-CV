"""
v6: aihub(헬멧) + WiderPerson(사람) + robo(fallen만) 데이터셋 구축
- robo person 데이터 제거, fallen만 유지
- WiderPerson → YOLO 포맷 변환
- WiderPerson class 1,2,3 → person (class 2)
- WiderPerson class 4,5 → 제외 (ignore, crowd)
- 4클래스: helmet_o(0), helmet_x(1), person(2), fallen(3)
"""
import shutil
from pathlib import Path
from collections import Counter
from PIL import Image

AIHUB_BASE = Path(r"D:\task\hoban\datasets_merged")
WIDER = Path(r"D:\task\hoban\WiderPerson")
OUT = Path(r"D:\task\hoban\datasets_v6")

def convert_wider_annotation(anno_path, img_path, output_label_path):
    """WiderPerson 어노테이션 → YOLO 포맷 변환"""
    try:
        img = Image.open(img_path)
        img_w, img_h = img.size
    except:
        return False

    with open(anno_path) as f:
        lines = f.readlines()

    n_annos = int(lines[0].strip())
    yolo_lines = []

    for i in range(1, n_annos + 1):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) != 5:
            continue

        cls_label = int(parts[0])
        x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

        # class 1(pedestrian), 2(rider), 3(partial) → person (class 2)
        # class 4(ignore), 5(crowd) → 제외
        if cls_label not in [1, 2, 3]:
            continue

        # 좌표 보정
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 <= x1 or y2 <= y1:
            continue

        # YOLO 포맷 변환 (정규화)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # 너무 작은 bbox 제외
        if w * h < 0.0005:
            continue

        yolo_lines.append(f"2 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not yolo_lines:
        return False

    with open(output_label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")
    return True

def main():
    if OUT.exists():
        shutil.rmtree(OUT)

    for s in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)

    # ========================================
    # 1. aihub 데이터 복사 (헬멧 - class 0, 1)
    # ========================================
    print("=== aihub 데이터 복사 (헬멧) ===")
    aihub_count = {"train": 0, "valid": 0}

    for split_src, split_dst in [("train", "train"), ("valid", "valid")]:
        img_dir = AIHUB_BASE / split_src / "images"
        lbl_dir = AIHUB_BASE / split_src / "labels"

        for lf in lbl_dir.iterdir():
            if lf.suffix != ".txt" or not lf.stem.startswith("aihub"):
                continue

            # 이미지 찾기
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{lf.stem}{ext}"
                if p.exists():
                    img = p
                    break
            if not img:
                continue

            shutil.copy2(img, OUT / split_dst / "images" / img.name)
            shutil.copy2(lf, OUT / split_dst / "labels" / lf.name)
            aihub_count[split_dst] += 1

    print(f"  train: {aihub_count['train']}장, valid: {aihub_count['valid']}장")

    # ========================================
    # 2. WiderPerson 데이터 변환 (사람 - class 2)
    # ========================================
    print("\n=== WiderPerson 데이터 변환 ===")

    # train/val 리스트 읽기
    with open(WIDER / "train.txt") as f:
        train_ids = [l.strip() for l in f if l.strip()]
    with open(WIDER / "val.txt") as f:
        val_ids = [l.strip() for l in f if l.strip()]

    print(f"  WiderPerson train: {len(train_ids)}개, val: {len(val_ids)}개")

    wider_count = {"train": 0, "valid": 0}

    for split_name, id_list in [("train", train_ids), ("valid", val_ids)]:
        for img_id in id_list:
            img_path = WIDER / "Images" / f"{img_id}.jpg"
            anno_path = WIDER / "Annotations" / f"{img_id}.jpg.txt"

            if not img_path.exists() or not anno_path.exists():
                continue

            label_out = OUT / split_name / "labels" / f"wider_{img_id}.txt"
            success = convert_wider_annotation(anno_path, img_path, label_out)

            if success:
                shutil.copy2(img_path, OUT / split_name / "images" / f"wider_{img_id}.jpg")
                wider_count[split_name] += 1

    print(f"  변환 완료 - train: {wider_count['train']}장, valid: {wider_count['valid']}장")

    # ========================================
    # 3. robo fallen 데이터 복사 (class 3만)
    # ========================================
    print("\n=== robo fallen 데이터 복사 ===")
    robo_count = {"train": 0, "valid": 0}

    for split_src, split_dst in [("train", "train"), ("valid", "valid")]:
        img_dir = AIHUB_BASE / split_src / "images"
        lbl_dir = AIHUB_BASE / split_src / "labels"

        for lf in lbl_dir.iterdir():
            if lf.suffix != ".txt" or not lf.stem.startswith("robo"):
                continue

            # 라벨에서 fallen(class 3)만 추출
            with open(lf) as f:
                lines = [l.strip() for l in f if l.strip()]

            fallen_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) == 3:
                    fallen_lines.append(line)

            if not fallen_lines:
                continue

            # 이미지 찾기
            img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{lf.stem}{ext}"
                if p.exists():
                    img = p
                    break
            if not img:
                continue

            # fallen 라벨만 저장
            out_label = OUT / split_dst / "labels" / lf.name
            with open(out_label, "w") as f:
                f.write("\n".join(fallen_lines) + "\n")

            shutil.copy2(img, OUT / split_dst / "images" / img.name)
            robo_count[split_dst] += 1

    print(f"  train: {robo_count['train']}장, valid: {robo_count['valid']}장")

    # ========================================
    # 4. 클래스 분포 확인
    # ========================================
    print("\n=== 최종 클래스 분포 ===")
    names = {0: "person_with_helmet", 1: "person_without_helmet", 2: "person", 3: "fallen"}
    for sn in ["train", "valid"]:
        counter = Counter()
        src_counter = Counter()
        for lf in (OUT / sn / "labels").iterdir():
            if lf.suffix != ".txt":
                continue
            if lf.stem.startswith("wider"):
                src = "wider"
            elif lf.stem.startswith("aihub"):
                src = "aihub"
            else:
                src = "robo"
            src_counter[src] += 1
            with open(lf) as f:
                for line in f:
                    if line.strip():
                        counter[int(line.split()[0])] += 1
        print(f"  {sn} (aihub={src_counter.get('aihub',0)}, wider={src_counter.get('wider',0)}, robo={src_counter.get('robo',0)}):")
        for cls in sorted(counter):
            print(f"    {cls} ({names.get(cls, '?')}): {counter[cls]}")

    # ========================================
    # 5. data.yaml 생성
    # ========================================
    yaml_content = """path: D:/task/hoban/datasets_v6
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

    print("\ndata.yaml 생성 완료")
    print("done!")

if __name__ == "__main__":
    main()
