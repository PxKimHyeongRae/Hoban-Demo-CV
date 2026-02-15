# 라벨링 이미지 추적 (중복 방지용)

## go2k_manual (수동 라벨링 완료)
- 위치: `/home/lay/hoban/datasets/go2k_manual/`
- 604장 (cam1+cam2+cam3), 1,680 bbox
- 소스: snapshots_raw에서 선별
- 상태: 라벨링 완료

## captures_3k_gated (CVAT 검수 대상)
- 위치: `/home/lay/hoban/datasets/captures_labels_3k/` (라벨)
- 소스: `/home/lay/video_indoor/static/captures/cam1,cam2/`
- 3,000장 (cam1: 1,219, cam2: 1,781), 5,058 bbox (pseudo-label)
- CVAT 패키지: `/home/lay/hoban/datasets/cvat_all/` (3개 task)
  - Part 1: 1,202장 (go2k 604 + captures 598)
  - Part 2: 1,202장 (captures only)
  - Part 3: 1,200장 (captures only)
- 선별 조건:
  - 전체 시간대 (주간 필터 없음)
  - 10초 간격 샘플링
  - go2k_manual 타임스탬프 제외
  - cam1+cam2 타임스탬프 기준 섞기 (interleave)
- 탐지 방법: SAHI + Full-Image Gate (#2)
  - SAHI: conf=0.50, 640x640, overlap=0.2, NMS/0.4/IOS
  - Gate: conf=0.20, radius=40px (FP 19% 필터)
- 모델: go2k_v2 best.pt (mAP50=0.927)
- 원본 프리즈 중복 제거 완료 (43,056장/38.8GB 삭제)

## 중복 검사 방법
```python
# 타임스탬프 키: cam1_20260213_131225 (카메라_날짜_시분초)
key = "_".join(filename.split("_")[:3])
```
- go2k_manual vs captures_3k: 타임스탬프 겹침 0개 확인됨
- captures 원본 프리즈 중복: dedup_captures.py로 제거 완료
- 추후 데이터 추가 시 동일 방식으로 중복 확인 필요
