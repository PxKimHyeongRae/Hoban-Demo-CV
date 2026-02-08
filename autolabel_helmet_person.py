"""
helmet_60k_yolo 이미지에 person bbox auto-labeling
- COCO pre-trained YOLO로 person 탐지
- 기존 helmet label (cls 0, 1)에 person label (cls 2) 추가
- 필터: confidence >= 0.5, area >= 2%
- 검증: helmet bbox가 person bbox 안에 포함되는지 IoA 통계
- 결과: dataset/helmet_60k_labeled/ (images + labels)
- 시각 검증: autolabel_check/ (랜덤 100장)
"""
import shutil
import random
from pathlib import Path
from collections import Counter
from multiprocessing import freeze_support

from ultralytics import YOLO

# === 설정 ===
HELMET_DIR = Path(r"D:\task\hoban\dataset\helmet_60k_yolo")
OUT_DIR = Path(r"D:\task\hoban\dataset\helmet_60k_labeled")
CHECK_DIR = Path(r"D:\task\hoban\autolabel_check")

CONF_THRESHOLD = 0.5
AREA_MIN = 0.02        # 2%
IOA_THRESHOLD = 0.3    # helmet bbox가 person bbox 안에 30% 이상 포함
BATCH_SIZE = 32
TARGET_CLS = 2         # person class in our dataset
COCO_PERSON = 0        # person class in COCO

SEED = 42


def ioa(inner, outer):
    """inner box가 outer box 안에 포함되는 비율 (Intersection / Area of inner)
    boxes: [cx, cy, w, h] normalized
    """
    ix1 = inner[0] - inner[2] / 2
    iy1 = inner[1] - inner[3] / 2
    ix2 = inner[0] + inner[2] / 2
    iy2 = inner[1] + inner[3] / 2

    ox1 = outer[0] - outer[2] / 2
    oy1 = outer[1] - outer[3] / 2
    ox2 = outer[0] + outer[2] / 2
    oy2 = outer[1] + outer[3] / 2

    xx1, yy1 = max(ix1, ox1), max(iy1, oy1)
    xx2, yy2 = min(ix2, ox2), min(iy2, oy2)

    if xx2 <= xx1 or yy2 <= yy1:
        return 0.0

    inter = (xx2 - xx1) * (yy2 - yy1)
    inner_area = inner[2] * inner[3]
    return inter / inner_area if inner_area > 0 else 0.0


def main():
    random.seed(SEED)

    print("=== Helmet → Person Auto-labeling ===\n")
    print(f"  소스: {HELMET_DIR}")
    print(f"  출력: {OUT_DIR}")
    print(f"  conf >= {CONF_THRESHOLD}, area >= {AREA_MIN * 100}%\n")

    # 1) Pre-trained COCO 모델 로드
    print("[1] COCO pre-trained 모델 로드...")
    model = YOLO("yolov8m.pt")

    img_dir = HELMET_DIR / "images"
    lbl_dir = HELMET_DIR / "labels"
    images = sorted(img_dir.iterdir())
    print(f"    대상 이미지: {len(images)}장\n")

    # 2) 출력 디렉토리 준비
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images").mkdir(parents=True)
    (OUT_DIR / "labels").mkdir(parents=True)

    # 3) 시각 검증용 샘플 선택
    check_indices = set(random.sample(range(len(images)), min(100, len(images))))
    if CHECK_DIR.exists():
        shutil.rmtree(CHECK_DIR)
    CHECK_DIR.mkdir(parents=True)

    # 4) 배치 추론 + 라벨 병합
    print("[2] Auto-labeling 진행...")
    stats = Counter()

    for batch_start in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[batch_start:batch_start + BATCH_SIZE]
        batch_paths = [str(p) for p in batch_imgs]

        results = model(batch_paths, conf=0.3, classes=[COCO_PERSON],
                        verbose=False, imgsz=640)

        for j, (img_path, result) in enumerate(zip(batch_imgs, results)):
            idx = batch_start + j
            stats["total"] += 1
            stem = img_path.stem

            # 기존 라벨 읽기
            lbl_path = lbl_dir / f"{stem}.txt"
            existing_lines = []
            helmet_boxes = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            existing_lines.append(line.strip())
                            cls = int(parts[0])
                            if cls in [0, 1]:
                                helmet_boxes.append(
                                    [float(x) for x in parts[1:5]])

            # Person 탐지 결과 처리
            person_lines = []
            img_h, img_w = result.orig_shape

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        stats["filt_conf"] += 1
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2 / img_w
                    cy = (y1 + y2) / 2 / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    area = bw * bh

                    if area < AREA_MIN:
                        stats["filt_area"] += 1
                        continue

                    person_box = [cx, cy, bw, bh]

                    # helmet↔person IoA 매칭 (통계용)
                    matched = any(
                        ioa(hb, person_box) >= IOA_THRESHOLD
                        for hb in helmet_boxes
                    )
                    if matched:
                        stats["helmet_matched"] += 1

                    person_lines.append(
                        f"{TARGET_CLS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    stats["person_bbox"] += 1

            if person_lines:
                stats["img_with_person"] += 1
            else:
                stats["img_no_person"] += 1

            # 라벨 저장 (기존 helmet + 새 person)
            all_lines = existing_lines + person_lines
            with open(OUT_DIR / "labels" / f"{stem}.txt", "w") as f:
                f.write("\n".join(all_lines) + "\n")

            # 이미지 복사
            shutil.copy2(img_path, OUT_DIR / "images" / img_path.name)

            # 시각 검증용 저장
            if idx in check_indices and person_lines:
                save_check_image(img_path, helmet_boxes, person_lines,
                                 CHECK_DIR / f"{stem}.jpg")

            if stats["total"] % 10000 == 0:
                print(f"    진행: {stats['total']:,}/{len(images):,}")

    # 5) 결과 출력
    t = stats["total"]
    wp = stats["img_with_person"]
    print(f"\n=== 결과 ===")
    print(f"  총 이미지: {t:,}")
    print(f"  person 추가: {wp:,}장 ({wp / t * 100:.1f}%)")
    print(f"  person 없음: {stats['img_no_person']:,}장")
    print(f"  추가된 person bbox: {stats['person_bbox']:,}개")
    print(f"  helmet↔person 매칭: {stats['helmet_matched']:,}개")
    print(f"  필터(conf < {CONF_THRESHOLD}): {stats['filt_conf']:,}")
    print(f"  필터(area < {AREA_MIN * 100}%): {stats['filt_area']:,}")
    print(f"\n  시각 검증: {CHECK_DIR}/  (랜덤 100장)")
    print("done!")


def save_check_image(img_path, helmet_boxes, person_lines, out_path):
    """bbox 시각화 저장 (helmet=green, person=blue)"""
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]

        # Helmet bbox (green)
        for hb in helmet_boxes:
            cx, cy, bw, bh = hb
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "helmet", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Person bbox (blue)
        for pl in person_lines:
            parts = pl.split()
            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, "person(auto)", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imwrite(str(out_path), img)
    except ImportError:
        pass


if __name__ == "__main__":
    freeze_support()
    main()
