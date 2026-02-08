"""
fastdup로 datasets_merged 전체 이미지 품질 분석
- aihub + robo 원본 데이터 중복 탐지
"""
import fastdup
import os

# aihub, robo 각각 분석
SOURCES = {
    "aihub": {
        "input": "/mnt/d/task/hoban/datasets_merged/train/images",
        "work": "/mnt/d/task/hoban/fastdup_aihub",
        "prefix": "aihub",
    },
    "robo": {
        "input": "/mnt/d/task/hoban/datasets_merged/train/images",
        "work": "/mnt/d/task/hoban/fastdup_robo",
        "prefix": "robo",
    },
    "coco": {
        "input": "/mnt/d/task/hoban/coco_person/images",
        "work": "/mnt/d/task/hoban/fastdup_coco",
        "prefix": None,
    },
}

def analyze(name, input_dir, work_dir, prefix=None):
    print(f"\n{'='*50}")
    print(f"=== {name} 분석 시작 ===")
    print(f"{'='*50}")

    os.makedirs(work_dir, exist_ok=True)

    # prefix 필터가 있으면 파일 목록 생성
    if prefix:
        import glob
        files = sorted(glob.glob(os.path.join(input_dir, f"{prefix}*")))
        print(f"  {prefix}* 파일: {len(files)}장")
        # 파일 목록을 텍스트로 저장
        list_file = os.path.join(work_dir, "file_list.txt")
        with open(list_file, "w") as f:
            f.write("\n".join(files))
        fd = fastdup.create(work_dir=work_dir, input_dir=input_dir)
        fd.run(subset=files)
    else:
        fd = fastdup.create(work_dir=work_dir, input_dir=input_dir)
        fd.run()

    # 유사도 > 0.9
    sim = fd.similarity()
    if sim is not None and len(sim) > 0:
        col = sim.columns[2]  # distance column
        high = sim[sim[col] > 0.9]
        print(f"  유사도 > 0.9 쌍: {len(high)}개")
        if len(high) > 0:
            print(high.head(5))
    else:
        print("  유사 이미지 없음")

    # 이상치
    out = fd.outliers()
    print(f"  이상치: {len(out) if out is not None else 0}개")

    # 통계
    stats = fd.img_stats()
    if stats is not None:
        print(f"  총 이미지: {len(stats)}장")
        invalid = stats[~stats["is_valid"]] if "is_valid" in stats.columns else None
        if invalid is not None and len(invalid) > 0:
            print(f"  깨진 이미지: {len(invalid)}장")

    return fd

def main():
    for name, cfg in SOURCES.items():
        analyze(name, cfg["input"], cfg["work"], cfg.get("prefix"))
    print("\n\ndone!")

if __name__ == "__main__":
    main()
