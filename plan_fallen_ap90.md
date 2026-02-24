# Fallen AP50 0.9+ 달성 계획

## 현재 상태 (v26)

| 클래스 | AP50 | Recall | Precision |
|--------|------|--------|-----------|
| helmet_on | 0.970 | 0.944 | 0.902 |
| helmet_off | 0.943 | 0.908 | 0.820 |
| **fallen** | **0.774** | **0.691** | **0.845** |

- 학습 데이터: helmet 2,000장 + fallen 1,074장 = 3,074장
- SAHI helmet F1=0.958 (ignore-aware), helmet 성능 유지됨

---

## 근본 원인 분석

### 1. 도메인 갭 (62배 스케일 불일치)

| 지표 | CCTV 헬멧 (L-tier) | Fallen 학습 데이터 | 배율 |
|------|-------------------|-------------------|------|
| 해상도 | 1920x1080 | 640x640 | 3x |
| bbox 면적 중앙값 | 0.04% | 2.5% | **62x** |
| bbox < 0.5% 비율 | 100% | 19.3% | - |

학습 데이터의 fallen은 640x640 스튜디오 이미지에서 면적 2.5%로 크게 보이지만, 실제 CCTV에서 쓰러진 사람은 1920x1080 프레임의 0.04% 수준. 모델이 큰 fallen만 학습하고 작은 fallen을 못 잡는 구조적 문제.

### 2. v26 데이터 추출 버그

`train_go3k_v26.py:190`에서:
```python
remaining = max(0, 1000 - v24_count)  # v24_count=1073 → remaining=0!
```
v24 데이터만 1,073장이라 unified_safety 데이터가 사실상 0장 추가됨. unified_safety_all의 29,378장 fallen 이미지를 전혀 활용하지 못함.

### 3. 활용 가능하지만 미사용 데이터

| 소스 | Fallen 이미지 | Fallen bbox | area<0.5% | area<1% |
|------|--------------|-------------|-----------|---------|
| unified_safety_all (train) | 29,378 | 50,211 | 5,606 | 8,646 |
| hoban_archive v12 (train) | ~17,189 | 27,105 | - | - |
| **현재 v26 사용** | **1,074** | **1,948** | - | - |

---

## 전략 (우선순위순)

### Phase 1: 즉시 실행 (1-2일) - 예상 +5~8%p

#### 전략 1. Unified 데이터 추출 버그 수정 + 소형 fallen 대량 추가
- **작업**: `remaining` 캡 제거, unified_safety_all에서 area<1% fallen 5,000~8,000장 추가
- **근거**: 50,211개 fallen bbox 중 34.9%가 area<1% (CCTV 스케일에 가까움)
- **구현**:
  - unified_safety_all class 4 → class 2 리매핑 (기존 코드 활용)
  - area<1% 필터로 CCTV 스케일 유사 데이터만 선별
  - 기존 v24 1,074장 + unified 5,000장 = fallen 6,074장
- **예상 효과**: +3~5%p AP50
- **리스크**: 낮음 (코드 수정만 필요)

#### 전략 2. Class-weighted Loss (fallen 가중치 증가)
- **작업**: fallen 클래스 loss 가중치 2~3배 증가
- **구현**: Ultralytics callback으로 per-class weight 적용, 또는 `cls` 파라미터 0.5→1.0
- **예상 효과**: +1~3%p AP50
- **리스크**: 낮음, helmet 성능 모니터링 필요

#### 전략 3. SAHI 추론 적용 (fallen 전용)
- **작업**: 추론 시 fallen 전용 SAHI 타일링 추가
- **구현**: 기존 helmet SAHI (1280x720) + fallen 전용 소형 타일 (640x480)
- **예상 효과**: +3~8%p Recall (추론 전용, 재학습 불필요)
- **리스크**: 낮음, 추론 속도 2~4x 느려짐

### Phase 2: 합성 데이터 파이프라인 (3-5일) - 예상 +5~10%p

#### 전략 4. CCTV 배경에 Fallen 합성 붙여넣기
- **작업**: 스튜디오 fallen 크롭 → CCTV 배경(1920x1080)에 축소 붙여넣기
- **구현**:
  1. 640x640 스튜디오 데이터에서 fallen 사람 크롭 추출 (마스크/bbox)
  2. L-tier CCTV 이미지(2,000장)를 배경으로 사용
  3. fallen 크롭을 0.02~0.5% 면적으로 축소하여 바닥 위치에 배치
  4. Gaussian blur + 밝기 매칭으로 합성 아티팩트 감소
  5. 2,000~3,000장 합성 이미지 생성
- **예상 효과**: +5~10%p AP50 (62x 스케일 갭 직접 해결)
- **리스크**: 중간 (합성 아티팩트 → FP 학습 가능)

#### 전략 5. 커리큘럼 학습 (대→소 점진적)
- **작업**: 큰 fallen부터 학습 → 점진적으로 작은 fallen 추가
- **구현**:
  - Stage 1 (20ep): fallen area>5% (쉬운 예제, ~1,200장)
  - Stage 2 (30ep): fallen area 1~5% 추가 (~4,500장)
  - Stage 3 (50ep): fallen area<1% 추가 (~5,600장), lr=0.001
- **예상 효과**: +2~5%p AP50
- **리스크**: 낮음 (v13에서 검증된 방법)

### Phase 3: 고급 기법 (1주+) - 예상 +2~5%p

#### 전략 6. Multi-scale 학습
- **작업**: 입력 해상도 랜덤 변동 (640~1920px)
- **구현**: `multi_scale=0.5` 파라미터 활성화
- **예상 효과**: +2~4%p (다양한 스케일 학습)
- **리스크**: batch 4→2 감소 필요 (VRAM 16GB 한계)

#### 전략 7. 실제 CCTV Fallen 검증셋 구축
- **작업**: video_indoor 녹화에서 fallen 시뮬레이션 촬영 + 라벨링
- **구현**:
  - `/data/video_indoor_archive/`에서 프레임 추출
  - 실제 카메라 앞에서 쓰러짐 시뮬레이션 50~100장 촬영
  - 수동 라벨링으로 CCTV 도메인 검증셋 구축
- **예상 효과**: 측정 정확도 대폭 향상 (현재 AP50=0.774는 스튜디오 val 기준)
- **리스크**: 수동 작업 필요

#### 전략 8. P2 소형 객체 탐지 헤드 추가
- **작업**: YOLO에 stride-4 탐지 헤드 추가 (최소 ~8px 객체 탐지)
- **구현**: 커스텀 모델 YAML 수정
- **예상 효과**: +2~4%p (극소형 fallen 전용)
- **리스크**: VRAM +2~3GB, batch=2로 제한, 구현 복잡

---

## 추천 실행 순서

```
v27 (즉시 실행):
  전략 1 (unified 데이터 대량 추가) + 전략 2 (class weight)
  → 예상: AP50 0.82~0.85

v28 (합성 데이터):
  v27 + 전략 4 (CCTV 합성 붙여넣기)
  → 예상: AP50 0.85~0.90

v29 (최적화):
  v28 + 전략 5 (커리큘럼) + 전략 3 (SAHI 추론)
  → 예상: AP50 0.88~0.93
```

### 현실적 전망

| 시나리오 | 기대 AP50 | 조건 |
|----------|-----------|------|
| 스튜디오 val 기준 | **0.90~0.93** | Phase 1+2 전략 조합 |
| 실제 CCTV 기준 (추정) | **0.70~0.85** | 합성 데이터 품질에 의존 |
| CCTV fallen 데이터 확보 시 | **0.90+** | 50장 이상 실제 데이터 필요 |

**핵심**: 스튜디오 val 기준 0.9+는 Phase 1+2로 달성 가능. 실제 CCTV에서 0.9+ 달성하려면 실제 CCTV fallen 데이터(촬영/라벨링) 확보가 거의 필수.

---

## 데이터 소스 경로

```
# 가장 큰 소스 (29K fallen 이미지, class 4)
/data/unified_safety_all/train/{images,labels}/

# 현재 사용 중 (v24 curated, class 2)
/home/lay/hoban/datasets_go3k_v24/train/{images,labels}/

# 레거시 아카이브 (27K fallen bbox)
/data/hoban_archive/datasets_v12/train/{images,labels}/

# CCTV 배경용 (합성 붙여넣기)
/home/lay/hoban/datasets_minimal_l/train/images/   # 1920x1080

# CCTV 녹화 (검증셋 구축용)
/data/video_indoor_archive/
```

---

## 즉시 시작할 수 있는 v27 스크립트 변경사항

```python
# train_go3k_v27.py 핵심 변경점:

# 1. unified 캡 제거 → 독립 파라미터
MAX_V24_FALLEN = 1074      # 기존 v24 전부
MAX_UNIFIED_FALLEN = 5000   # unified area<1% 전부

# 2. area 필터 강화 (CCTV 스케일 매칭)
unified_filter = 0.01       # area < 1% (vs v26의 5%)

# 3. class weight (Ultralytics callback)
# cls_loss에 fallen weight 2.0 적용

# 4. Augmentation 미세 조정
copy_paste = 0.3            # 0.4→0.3 (과도 방지)
scale = 0.7                 # 유지
mosaic = 1.0                # 유지
```
