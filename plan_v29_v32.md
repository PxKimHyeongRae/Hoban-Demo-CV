# Fallen 개선 실험 계획: v29 ~ v32

## 목표

- helmet F1 0.958 유지
- fallen AP50 0.774 (v26) → 0.90+ 달성

## 핵심 전략

v25→v26에서 확인된 성공 패턴: **train-val 분포 매칭이 데이터 양보다 결정적** (r=0.915)

---

## v29: 랜덤 샘플링 + close_mosaic=0 (Baseline)

### 목적
- area 필터 없이 랜덤으로 fallen을 뽑았을 때 성능 확인 (v30 stratified와 비교 기준)
- close_mosaic=0 효과 단독 검증

### 변경점 (v26 대비)
1. **Fallen 랜덤 샘플링**: v24 + unified에서 area 필터 없이 랜덤 추출 (~1,200장)
2. **close_mosaic=0**: mosaic 100 epoch 유지 (매 버전 -1%p 역효과 제거)
3. **patience=25**: early stopping 여유 확대

### 데이터 소스
- **v24 fallen**: area 필터 없이 전체 사용 (~1,073장)
- **unified_safety_all**: class 4(fallen) 보유 이미지에서 랜덤 추출 (~200장 보충)
- unified는 class 4 → class 2 리매핑, 나머지 클래스 제외

### 학습 설정

```python
# v26 기반 (변경점만 표시)
close_mosaic=0,    # v26: 10 → 0 (mosaic 항상 유지)
patience=25,       # v26: 20 → 25
# 나머지 v26 동일: SGD lr0=0.005, copy_paste=0.4, scale=0.7, batch=4
```

### 데이터셋 구성

| 구분 | 수량 |
|------|------|
| Helmet (L-tier) | 2,000 |
| Fallen (랜덤) | ~1,200 |
| **Total** | ~3,200 |
| Val | 617 (기존 유지) |

### 예상 효과
- close_mosaic=0만으로 +1%p 기대
- 랜덤 샘플링이므로 분포 매칭 효과는 제한적
- fallen AP50: 0.774 → **0.78~0.80**
- v30 stratified와의 비교 기준선

---

## v30: Val-Aligned Area Stratified Sampling + close_mosaic=0

### 목적
- val 분포에 정확히 맞춘 fallen 데이터로 분포 매칭 효과 극대화
- v29(랜덤) 대비 개선폭으로 분포 매칭의 실제 효과 정량화

### 변경점 (v29 대비)
1. **Area stratified sampling**: val 분포에 맞춘 구간별 쿼터 샘플링

### Val fallen area 분포 (목표)

| Area 구간 | Val 비율 | v26 비율 | v30 목표 비율 |
|-----------|----------|----------|---------------|
| <1% | 25% | 38% | 25% |
| 1~5% | 23% | 31% | 23% |
| 5~10% | 21% | 14% | 21% |
| 10~20% | 19% | 12% | 19% |
| >20% | 13% | 5% | 13% |

### 구현

```python
# Val 분포에 맞춘 area 구간별 쿼터
AREA_BINS = [
    (0.00, 0.01, 0.25),  # <1%: 25%
    (0.01, 0.05, 0.23),  # 1~5%: 23%
    (0.05, 0.10, 0.21),  # 5~10%: 21%
    (0.10, 0.20, 0.19),  # 10~20%: 19%
    (0.20, 1.00, 0.13),  # >20%: 13%
]
TARGET_FALLEN = 1200

# 각 구간별 목표 수량
# <1%: 300, 1~5%: 276, 5~10%: 252, 10~20%: 228, >20%: 156

# 소스 우선순위: v24 먼저, 부족하면 unified에서 보충
# unified는 class 4 → class 2 리매핑
```

### 학습 설정

```python
# v29와 동일
close_mosaic=0, patience=25
# 나머지 v26 동일: SGD lr0=0.005, copy_paste=0.4, scale=0.7
```

### 데이터셋 구성

| 구분 | 수량 |
|------|------|
| Helmet (L-tier) | 2,000 |
| Fallen (area-stratified) | ~1,200 |
| **Total** | ~3,200 |
| Val | 617 (기존 유지) |

### 예상 효과
- Bhattacharyya overlap: 0.823 → 0.95+ (val 분포 직접 매칭)
- fallen AP50: 0.78~0.80 → **0.82~0.85**
- v29 대비 +2~5%p 개선 기대 (분포 매칭 효과)

---

## v31: v30 + Per-Class Loss Weight (cls 강화)

### 목적
- fallen 분류 능력 강화를 위한 loss weight 조정

### 변경점 (v30 대비)
1. **cls=1.0**: classification loss weight 0.5→1.0 (분류 loss 2배)
2. **scale=0.8**: 더 공격적 소형 객체 학습

### 학습 설정

```python
# v30 기반 + 변경점
cls=1.0,       # v30: 0.5 → 1.0 (classification loss 2배)
scale=0.8,     # v30: 0.7 → 0.8
# 나머지 v30 동일
```

### 근거
- Ultralytics는 per-class loss weight 미지원 → cls 전체 증가로 대체
- 3-class에서 fallen이 가장 어려운 클래스 → cls 강화가 fallen에 가장 큰 효과
- scale 0.8로 소형 fallen 학습 기회 증가

### 예상 효과
- fallen AP50: 0.82~0.85 → **0.85~0.88**
- helmet F1 유지 (cls 강화는 전체 분류에 도움)

---

## v32: v31 + Multi-Scale Curriculum

### 목적
- 큰 fallen부터 학습 → 작은 fallen으로 점진적 확장

### 변경점 (v31 대비)
1. **2-stage curriculum learning**:
   - Stage 1 (50ep): 중~대형 fallen만 학습 (area > 3%)
   - Stage 2 (50ep): 전체 fallen으로 fine-tune (Stage 1 weights 기반)

### 구현

```python
# Stage 1: 큰 fallen 먼저 학습
def prepare_stage1():
    # fallen 중 area > 3%만 선택 (~700장)
    # helmet 2,000장 유지
    # 총 ~2,700장
    # v31과 동일 하이퍼파라미터 (cls=1.0, scale=0.8)

# Stage 2: 전체 데이터로 fine-tune
def prepare_stage2():
    # v30 데이터셋 그대로 사용 (~3,200장, area-stratified)
    # Stage 1 weights에서 시작
    # lr0=0.001 (1/5 감소), warmup_epochs=1.0
```

### 근거
- 큰 fallen → 작은 fallen 순서로 학습하면 feature hierarchy가 안정적으로 형성
- v13에서 curriculum learning 성공 사례 있음 (mAP50=0.945)
- Stage 2에서 lr 감소로 Stage 1 지식 보존

### 예상 효과
- fallen AP50: 0.85~0.88 → **0.88~0.92**

---

## 실행 일정

| 단계 | 작업 | 소요 |
|------|------|------|
| v29 준비 | train_go3k_v29.py 작성 + 데이터셋 빌드 | 30분 |
| v29 학습 | 100ep, batch=4, 1280px | ~7시간 |
| v29 평가 | SAHI + YOLO val | 5분 |
| v30 준비 | stratified sampling 스크립트 작성 + 빌드 | 30분 |
| v30 학습 | 100ep | ~7시간 |
| v30 평가 | SAHI + YOLO val | 5분 |
| v31 준비 | cls/scale 변경 | 10분 |
| v31 학습 | 100ep | ~7시간 |
| v31 평가 | SAHI + YOLO val | 5분 |
| v32 준비 | 2-stage curriculum 스크립트 작성 | 30분 |
| v32 학습 | Stage1 50ep + Stage2 50ep | ~7시간 |
| v32 평가 | SAHI + YOLO val | 5분 |

### 중간 판단 기준

- v29 후: fallen AP50 < 0.78이면 close_mosaic 효과 부족 → 데이터 재검토
- v30 후: fallen AP50 < v29이면 stratified sampling 전략 수정
- v31 후: helmet F1 < 0.950이면 cls 롤백
- v32 후: fallen AP50 < 0.85이면 curriculum 전략 수정

---

## 파일 구조

```
train_go3k_v29.py   # 랜덤 샘플링 + close_mosaic=0
train_go3k_v30.py   # area stratified sampling + close_mosaic=0
train_go3k_v31.py   # v30 + cls=1.0, scale=0.8
train_go3k_v32.py   # v31 + 2-stage curriculum
datasets_go3k_v29/  # 랜덤 fallen dataset
datasets_go3k_v30/  # area-stratified fallen dataset
datasets_go3k_v31/  # v30과 동일 (하이퍼파라미터만 변경)
datasets_go3k_v32/  # Stage1/Stage2 데이터셋
```
