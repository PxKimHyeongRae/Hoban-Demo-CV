# 라벨링 이미지 추적 (중복 방지용)

## go2k_manual (수동 라벨링 완료)
- 위치: `/home/lay/hoban/datasets/go2k_manual/`
- 604장 (cam1+cam2+cam3), 1,680 bbox
- 소스: snapshots_raw에서 선별

## captures_3k (CVAT 검수 대상)
- 위치: `/home/lay/hoban/datasets/captures_labels_3k/` (라벨)
- 소스: `/home/lay/video_indoor/static/captures/cam1,cam2/`
- 3,000장 (cam1: 1,926, cam2: 1,074), 4,141 bbox (pseudo-label)
- CVAT 패키지: `/home/lay/hoban/datasets/cvat_captures/` (3개 task x 1,000장)
- 선별 조건:
  - 주간 08:00~18:00
  - 30초 간격 샘플링
  - go2k_manual 타임스탬프 제외
  - SAHI 탐지 있는 이미지만 (conf=0.50, NMS/0.4/IOS)
- 모델: go2k_v2 best.pt (mAP50=0.927)

## 중복 검사 방법
```python
# 타임스탬프 키: cam1_20260213_131225 (카메라_날짜_시분초)
key = "_".join(filename.split("_")[:3])
```
- go2k_manual vs captures_3k: 타임스탬프 겹침 0개 확인됨
- 추후 데이터 추가 시 동일 방식으로 중복 확인 필요
