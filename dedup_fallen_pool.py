"""
fastdup 결과 기반 fallen_pool 중복/이상치 제거
- 유사도 > 0.9: 그리디 제거 (소스 우선순위 적용)
- 이상치: fastdup outlier score 기반 제거
"""
from pathlib import Path
from collections import Counter

WORK_DIR = Path(r"D:\task\hoban\fastdup_fallen")
POOL_DIR = Path(r"D:\task\hoban\fallen_pool")
THRESHOLD = 0.9

# 소스 우선순위 (높을수록 보존 우선)
SOURCE_PRIORITY = {
    "fall8": 10,  # fallen.v2i - 최대, 고품질
    "fall4": 7,   # fall.v1i (2)
    "fall1": 6,   # ip camera
    "fall3": 5,   # Fall.v1i
    "fall6": 5,   # Fall.v3i
    "fall5": 4,   # fall.v2i
    "fall7": 4,   # fall.v4i
    "fall2": 1,   # aug3x - 최저 우선순위
}


def get_source(filename):
    """파일명에서 소스 prefix 추출"""
    name = Path(filename).name
    return name.split("_")[0]


def get_priority(filename):
    src = get_source(filename)
    return SOURCE_PRIORITY.get(src, 0)


def main():
    # 1. 인덱스 → 파일명 매핑 로드
    mapping = {}
    csv_path = WORK_DIR / "atrain_features.dat.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    with open(csv_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    print(f"파일 매핑 로드: {len(mapping)}개")

    # 2. similarity.csv에서 유사도 >= threshold 쌍 추출
    pairs = []
    sim_path = WORK_DIR / "similarity.csv"
    with open(sim_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                from_idx, to_idx, dist = int(parts[0]), int(parts[1]), float(parts[2])
                if dist >= THRESHOLD:
                    pairs.append((from_idx, to_idx, dist))
    print(f"유사도 >= {THRESHOLD} 쌍: {len(pairs)}개")

    # 3. 그리디 중복 제거 (우선순위 기반)
    pairs.sort(key=lambda x: -x[2])  # 유사도 높은 순
    remove_idx = set()

    for from_idx, to_idx, dist in pairs:
        if from_idx in remove_idx or to_idx in remove_idx:
            continue
        # 우선순위 낮은 쪽 제거
        f1 = mapping.get(from_idx, "")
        f2 = mapping.get(to_idx, "")
        p1 = get_priority(f1)
        p2 = get_priority(f2)
        if p1 >= p2:
            remove_idx.add(to_idx)
        else:
            remove_idx.add(from_idx)

    remove_files = {mapping[idx] for idx in remove_idx if idx in mapping}
    print(f"중복 제거 대상: {len(remove_files)}장")

    # 소스별 제거 통계
    src_counter = Counter()
    for f in remove_files:
        src_counter[get_source(f)] += 1
    print(f"  소스별: {dict(sorted(src_counter.items()))}")

    # 4. 이미지 + 라벨 제거
    removed = 0
    for rel_path in remove_files:
        img_path = Path(rel_path)
        if not img_path.is_absolute():
            # try relative to pool
            img_path = POOL_DIR / "images" / img_path.name
            if not img_path.exists():
                img_path = Path(rel_path)

        if not img_path.exists():
            continue

        label_path = POOL_DIR / "labels" / (img_path.stem + ".txt")
        img_path.unlink()
        if label_path.exists():
            label_path.unlink()
        removed += 1

    print(f"\n실제 제거: {removed}장")

    # 5. 제거 후 통계
    remaining_imgs = len(list((POOL_DIR / "images").iterdir()))
    remaining_bbox = 0
    for lf in (POOL_DIR / "labels").iterdir():
        if lf.suffix == ".txt":
            with open(lf) as fh:
                remaining_bbox += sum(1 for line in fh if line.strip())

    print(f"\n=== 정제 후 ===")
    print(f"남은 이미지: {remaining_imgs}장")
    print(f"남은 fallen bbox: {remaining_bbox}개")

    # 소스별 남은 이미지
    src_remain = Counter()
    for img in (POOL_DIR / "images").iterdir():
        src_remain[get_source(img.name)] += 1
    print(f"소스별: {dict(sorted(src_remain.items()))}")
    print("done!")


if __name__ == "__main__":
    main()
