"""
fastdup 결과 기반 중복 이미지 제거 (유사도 0.9 이상)
- datasets_v7/train에서 중복 쌍 중 하나를 제거
- 이미지 + 라벨 함께 제거
"""
from pathlib import Path
from collections import Counter

WORK_DIR = Path(r"D:\task\hoban\fastdup_results")
DATASET_DIR = Path(r"D:\task\hoban\datasets_v7")
THRESHOLD = 0.9

def main():
    # 1. 인덱스 → 파일명 매핑 로드
    mapping = {}
    with open(WORK_DIR / "atrain_features.dat.csv") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",", 1)
            mapping[int(parts[0])] = parts[1]
    print(f"파일 매핑 로드: {len(mapping)}개")

    # 2. similarity.csv에서 유사도 >= 0.9 쌍 추출
    pairs = []
    with open(WORK_DIR / "similarity.csv") as f:
        f.readline()  # skip header: from,to,distance
        for line in f:
            parts = line.strip().split(",")
            from_idx, to_idx, dist = int(parts[0]), int(parts[1]), float(parts[2])
            if dist >= THRESHOLD:
                pairs.append((from_idx, to_idx, dist))
    print(f"유사도 >= {THRESHOLD} 쌍: {len(pairs)}개")

    # 3. 그리디 중복 제거 (유사도 높은 순)
    pairs.sort(key=lambda x: -x[2])
    remove_idx = set()

    for from_idx, to_idx, dist in pairs:
        if from_idx in remove_idx or to_idx in remove_idx:
            continue
        # to를 제거 (from 유지)
        remove_idx.add(to_idx)

    remove_files = {mapping[idx] for idx in remove_idx if idx in mapping}
    print(f"제거 대상: {len(remove_files)}장")

    # 소스별 제거 통계
    src_counter = Counter()
    for f in remove_files:
        name = Path(f).name
        if name.startswith("aihub"):
            src_counter["aihub"] += 1
        elif name.startswith("coco"):
            src_counter["coco"] += 1
        elif name.startswith("robo"):
            src_counter["robo"] += 1
    print(f"  소스별: {dict(src_counter)}")

    # 4. 이미지 + 라벨 제거
    removed = 0
    for rel_path in remove_files:
        # rel_path: "datasets_v7/train/images/xxx.jpg"
        img_path = DATASET_DIR.parent / Path(rel_path)
        if not img_path.exists():
            continue

        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        img_path.unlink()
        if label_path.exists():
            label_path.unlink()
        removed += 1

    print(f"\n실제 제거: {removed}장")

    # 5. 제거 후 통계
    names = {0: "helmet_o", 1: "helmet_x", 2: "person", 3: "fallen"}
    for split in ["train", "valid"]:
        img_count = len(list((DATASET_DIR / split / "images").iterdir()))
        counter = Counter()
        for lf in (DATASET_DIR / split / "labels").iterdir():
            if lf.suffix != ".txt":
                continue
            with open(lf) as fh:
                for line in fh:
                    if line.strip():
                        counter[int(line.split()[0])] += 1
        print(f"\n  {split} ({img_count}장):")
        for cls in sorted(counter):
            print(f"    {cls} ({names.get(cls,'?')}): {counter[cls]} bbox")

if __name__ == "__main__":
    main()
