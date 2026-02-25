# Fallen AP50 개선 종합 분석 및 계획

## 1. 현재 상황

- **목표**: helmet F1 0.958 유지 + fallen AP50 0.9+ 달성
- **현재 최고**: v26 (helmet F1=0.958, fallen AP50=0.774, Recall=0.691)

## 2. v25~v28 버전별 비교

### 데이터 구성

| 항목 | v25 | v26 | v27 | v28 |
|------|-----|-----|-----|-----|
| Total train | 2,568 | 3,074 | 4,000 | 4,073 |
| Helmet | 2,000 | 2,000 | 2,000 | 2,000 |
| Fallen | 568 | 1,074 | 2,000 | 2,073 |
| Fallen 비율 | 22.1% | 34.9% | 50.0% | 50.9% |
| Synth fallen | 0 | 0 | 1,000 | 0 |
| Unified fallen | 0 | 0 (cap bug) | 0 | ~1,000 |
| Fallen 소스 | v24 area<5% | v24 area<10% | v24+synth | v24+unified |

### 하이퍼파라미터 차이

| 항목 | v25 | v26 | v27 | v28 |
|------|-----|-----|-----|-----|
| copy_paste | 0.15 | **0.4** | 0.3 | **0.4** |
| scale | 0.5 | **0.7** | **0.7** | **0.7** |
| batch | 6 | 6 | 4 | 4 |

### 평가 결과

| 메트릭 | v25 | **v26** | v27 | v28 |
|--------|-----|---------|-----|-----|
| SAHI helmet F1 | - | **0.958** | 0.956 | 0.957 |
| Fallen AP50 | 0.551 | **0.774** | 0.716 | 0.784 |
| Fallen Recall | 0.459 | **0.691** | 0.587 | 0.673 |
| mAP50 | 0.812 | **0.896** | 0.875 | 0.895 |
| Best epoch | 50/51 | 80/100 | 87/100 | 85/100 |

## 3. 성공/실패 패턴 분석

### v25 → v26: +22.3%p fallen AP50 (대성공)

**핵심 원인**: area 필터 <5% → <10% 완화 + copy_paste 0.15→0.4 + scale 0.5→0.7

- v25는 tiny fallen(area<5%)만 학습 → val median(5.96%)과 큰 gap
- v26은 중형 fallen(5~10%)을 포함 → val 분포에 근접
- **분포 overlap** (Bhattacharyya): 0.678 → 0.823 (+21%)
- 더 강한 augmentation이 regularization 역할

### v26 → v27: -5.8%p fallen AP50 (역행)

**핵심 원인**: 합성 데이터 품질 부족 + 1:1 비율 과잉

- 합성 fallen의 crop+paste 아티팩트가 잘못된 특징 학습
- 분포 overlap 0.823 → 0.729 (-11%) 하락
- 1:1 비율 강제로 fallen 과대표현 → helmet 성능도 하락

### v26 → v28: +1.0%p fallen AP50 (미세 개선)

**핵심 원인**: unified 데이터가 area<5% 위주 → val 분포와 gap 여전

- Fallen bbox 2배(1,948→4,133)지만 성능은 flat
- KS statistic: 0.311 → 0.389 (분포 gap 오히려 증가)
- **데이터 양 < 분포 매칭** 확인됨

## 4. 핵심 발견

### 발견 1: 분포 매칭이 데이터 양보다 결정적

```
r(분포 overlap, mAP50) = 0.915 (강한 양의 상관)
r(데이터 양, mAP50) = 0.687 (중간 상관)
```

v28은 v26 대비 fallen 2배지만, area<5% 소형 위주라 효과 없음.

### 발견 2: Val fallen은 넓은 area 범위에 분포

| Area 구간 | Val 비율 | v26 Train 비율 | Gap |
|-----------|----------|---------------|-----|
| <1% | 25% | 38% | 과잉 |
| 1~5% | 23% | 31% | 과잉 |
| 5~10% | 21% | 14% | **부족** |
| 10~20% | 19% | 12% | **부족** |
| >20% | 13% | 5% | **부족** |

v26 train은 소형(area<5%)에 편중, 중~대형(>5%)이 부족.

### 발견 3: close_mosaic=10은 역효과

모든 버전에서 epoch 90(mosaic 해제)부터 val mAP50 약 -1.0%p 하락.
Mosaic이 regularizer 역할 → 끄면 과적합 증가.

## 5. 개선 방향 (우선순위 순)

### 방향 1: Val-Aligned Area Stratified Sampling (v29)

| 항목 | 내용 |
|------|------|
| 핵심 | val과 동일한 area 분포로 fallen 재샘플링 |
| 방법 | unified+v24에서 area 구간별 비율 맞춰 추출 (25%/23%/21%/19%/12%) |
| 예상 효과 | fallen AP50 +3~5%p (→ 0.82~0.85) |
| 리스크 | 낮음 |
| 난이도 | 쉬움 (1일) |

**가장 높은 기대 효과**. v25→v26의 성공 패턴(분포 매칭)을 체계적으로 적용.

### 방향 2: close_mosaic=0 + patience 확대

| 항목 | 내용 |
|------|------|
| 핵심 | mosaic을 100 epoch 내내 유지 |
| 방법 | close_mosaic=0, patience=25~30 |
| 예상 효과 | +1.0~1.5%p |
| 리스크 | 매우 낮음 |
| 난이도 | 매우 쉬움 (코드 1줄) |

모든 버전에서 확인된 close_mosaic 역효과를 제거.

### 방향 3: Per-Class Loss Weight

| 항목 | 내용 |
|------|------|
| 핵심 | fallen class에 더 높은 loss weight 부여 |
| 방법 | cls_loss weight 조정 또는 focal loss gamma 증가 |
| 예상 효과 | fallen AP50 +1~3%p |
| 리스크 | helmet 성능 하락 가능 (모니터링 필요) |
| 난이도 | 쉬움 |

### 방향 4: Multi-Scale Fallen Curriculum

| 항목 | 내용 |
|------|------|
| 핵심 | 큰 fallen → 점진적으로 작은 fallen 학습 |
| 방법 | Stage 1: area>5% only (50ep) → Stage 2: 전체 (50ep) |
| 예상 효과 | +2~4%p |
| 리스크 | 낮음 |
| 난이도 | 중간 (2일) |

### 방향 5: Val Set 개편 (CCTV Fallen 수집)

| 항목 | 내용 |
|------|------|
| 핵심 | 실제 CCTV 환경의 fallen 이미지로 val set 교체/보강 |
| 방법 | 현장 CCTV에서 fallen 시나리오 촬영 + 라벨링 |
| 예상 효과 | 실전 성능 측정 정확도 확보 |
| 리스크 | 중간 (촬영 필요) |
| 난이도 | 중간 (2~3일) |

현재 val set이 실제 배포 환경을 대표하는지 불명확. 측정이 정확해야 개선 방향도 정확.

## 6. 추천 실행 순서

```
v29: 방향 1 + 2 조합
  - Val-aligned area sampling + close_mosaic=0
  - 예상: fallen AP50 0.82~0.85, helmet F1 0.958 유지
  - 소요: 1일

v30: v29 + 방향 3
  - Per-class loss weight 추가
  - 예상: fallen AP50 0.85~0.88
  - 소요: 1일

v31: v30 + 방향 4
  - Multi-scale curriculum 적용
  - 예상: fallen AP50 0.88~0.92
  - 소요: 2일

병행: 방향 5
  - CCTV val set 확보로 실전 성능 측정 체계 구축
```

## 7. 참고 시각화

- 학습 커브 비교: `.omc/scientist/figures/v25_v28_learning_curves.png`
- Fallen area 분포: `.omc/scientist/figures/v25_v28_fallen_distribution.png`
- close_mosaic 영향: `.omc/scientist/figures/v25_v28_close_mosaic.png`
- 핵심 인사이트: `.omc/scientist/figures/v25_v28_key_insights.png`
