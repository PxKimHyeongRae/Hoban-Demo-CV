"""
fallen_pool에 대한 fastdup 분석
- WSL에서 실행: ~/fastdup_env/bin/python /mnt/d/task/hoban/analyze_fallen_pool.py
- 유사도 > 0.9 중복 쌍, 이상치, 이미지 통계
"""
import fastdup
import os

INPUT_DIR = "/mnt/d/task/hoban/fallen_pool/images"
WORK_DIR = "/mnt/d/task/hoban/fastdup_fallen"

def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    print("=== fastdup 분석 시작 ===")
    print(f"  input: {INPUT_DIR}")
    print(f"  work:  {WORK_DIR}")

    fd = fastdup.create(work_dir=WORK_DIR, input_dir=INPUT_DIR)
    fd.run()

    # 1. 유사도 > 0.9
    sim = fd.similarity()
    if sim is not None and len(sim) > 0:
        col = sim.columns[2]  # distance column
        high = sim[sim[col] > 0.9]
        print(f"\n유사도 > 0.9 쌍: {len(high)}개")

        # 소스 간 교차 중복 분석
        if len(high) > 0:
            cross_source = 0
            for _, row in high.head(10000).iterrows():
                f1 = str(row.iloc[0])
                f2 = str(row.iloc[1])
                # prefix에서 소스 추출 (fall1_, fall2_, ...)
                src1 = os.path.basename(f1).split("_")[0] if "/" in f1 or "\\" in f1 else f1.split("_")[0]
                src2 = os.path.basename(f2).split("_")[0] if "/" in f2 or "\\" in f2 else f2.split("_")[0]
                if src1 != src2:
                    cross_source += 1
            print(f"  교차 소스 중복 (상위 10K): {cross_source}개")
            print(f"\n  상위 5개:")
            print(high.head(5))
    else:
        print("\n유사 이미지 없음")

    # 2. 이상치
    out = fd.outliers()
    if out is not None:
        print(f"\n이상치: {len(out)}개")
        if len(out) > 0:
            print(f"  상위 5개:")
            print(out.head(5))
    else:
        print("\n이상치: 0개")

    # 3. 이미지 통계
    stats = fd.img_stats()
    if stats is not None:
        print(f"\n총 이미지: {len(stats)}장")
        if "is_valid" in stats.columns:
            invalid = stats[~stats["is_valid"]]
            print(f"깨진 이미지: {len(invalid)}장")
        # 해상도 분포
        if "width" in stats.columns and "height" in stats.columns:
            print(f"해상도 범위: {stats['width'].min()}x{stats['height'].min()} ~ {stats['width'].max()}x{stats['height'].max()}")
            print(f"평균 해상도: {stats['width'].mean():.0f}x{stats['height'].mean():.0f}")

    print("\n=== 분석 완료 ===")

if __name__ == "__main__":
    main()
