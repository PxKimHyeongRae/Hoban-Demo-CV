# Hoban 프로젝트 종합 분석 & 전략 로드맵

> 작성일: 2026-02-23
> 목표: 헬멧 착용 / 헬멧 미착용 / 쓰러짐 3-class 감지 최적화 전략

---

## 1. 현재 상황 진단

### 1.1 모델 성능 비교

| 지표 | v24 (Ep79, 3-class) | v17 (2-class, 배포중) | v19 (2-class, 최고) | v24 vs v19 |
|------|---------------------|----------------------|---------------------|------------|
| mAP50 | 0.8704 | 0.9580 | 0.9600 | **-9.0%p** |
| mAP50-95 | 0.5617 | 0.7220 | 0.7240 | **-16.2%p** |
| Precision | 0.8497 | 0.9146 | - | -6.5%p |
| Recall | 0.8210 | 0.9140 | - | -9.3%p |

**v24는 3-class 확장 + AdamW 전환으로 심각한 성능 하락 발생.**

### 1.2 버전별 성능 추이

| Version | Date | Classes | Model | Data | mAP50 | mAP50-95 | SAHI F1 | Status |
|---------|------|---------|-------|------|-------|----------|---------|--------|
| v9 | Feb 8 | 4 | yolo26m | multi-source | 0.940 | 0.702 | - | overfitting collapse |
| v13 | Feb 13 | 2 | yolo26m | curriculum | 0.945 | 0.727 | - | Archive |
| v16 | Feb 17 | 2 | yolo26m v13pt | 10,564 | 0.885 | 0.680 | 0.885 | Baseline |
| **v17** | Feb 18 | 2 | yolo26m COCOpt | 10,564 | **0.958** | 0.722 | **0.918** | **배포중** |
| v18 | Feb 18 | 2 | yolo26m v17ft | 10,652 | 0.955 | 0.715 | 0.916 | Experiment |
| **v19** | Feb 19 | 2 | yolo26m v17ft | 10,852 | **0.960** | **0.724** | **0.928** | **최고 성능** |
| v20 | Feb 20 | 2 | yolo26m v17 | 12,470 | 0.725 | 0.727 | 0.915 | 데이터↑ 성능↓ |
| v21-l | Feb 20 | 2 | yolo26l COCOpt | 12,470 | 0.958 | 0.723 | 0.923 | 모델↑ 효과없음 |
| v22 | Feb 21 | 2 | yolo26x COCOpt | 12,470 | 0.945 | 0.688 | - | 중단 (batch=2) |
| v23 | Feb 22 | 2 | yolo26m COCOpt | 4,038 | 0.951 | 0.704 | 0.912 | 현장only 부족 |
| v24 | Feb 23 | **3** | yolo26m COCOpt | 10,921 | 0.870 | 0.562 | TBD | **학습중** |

### 1.3 사용 가능한 데이터 현황

| 데이터소스 | 경로 | 크기 | 이미지 수 | 설명 |
|-----------|------|------|----------|------|
| AIHub helmet_60k | `/data/aihub_data/helmet_60k/` | 36GB | 60,000 | area 기준 정렬된 고품질 헬멧 |
| helmet 2-class | `/data/helmet/` | 50GB | 75,928 | AIHub 추출 2-class |
| unified_safety_all | `/data/unified_safety_all/` | 258GB | 493,123 | 8-class (fallen=class4) |
| negative_10k_v3 | `/data/aihub_data/helmet_negative_10k_v3/` | 4.7GB | 10,000 | YOLO 검증 무인 배경 |
| video_indoor_archive | `/data/video_indoor_archive/` | 4.2GB | - | 실제 이벤트 녹화 |

**Fallen 데이터**: 29,378개 파일에 50,211개 bbox (unified_safety_all train)

---

## 2. 13개 버전에서 확인된 핵심 법칙

### 2.1 확실히 효과 있는 것

| # | 법칙 | 근거 | 영향도 | 확신도 |
|---|------|------|--------|--------|
| 1 | **데이터 품질 > 양** | v19(10.8K)=0.928 > v20(12.5K)=0.915 | +1.3%p | 매우 높음 |
| 2 | **COCO pretrain 필수** | domain-specific pt 대비 +2%p | +2.0%p | 높음 |
| 3 | **1280px 해상도 고정** | 640→1280: +3.4%p F1 | +3.4%p | 매우 높음 |
| 4 | **SGD 옵티마이저** | AdamW는 장기 학습 시 overfitting collapse | 안정성 | 높음 |
| 5 | **SAHI 필수** | 타일 추론으로 F1 0.804→0.912 | +10.8%p | 매우 높음 |
| 6 | **Gate 후처리** | FP -40개 제거 | +0.8%p | 높음 |
| 7 | **Per-class confidence** | ON≥0.40, OFF≥0.15 비대칭 임계값 | +0.2%p | 높음 |
| 8 | **Warm restart** | v19(resume)=0.928 > v20(fresh)=0.915 | +1.0%p | 중간 |
| 9 | **Hard negative mining** | FP 패턴 이미지 학습에 포함 | +0.5-1.0%p | 중간 |

### 2.2 확실히 효과 없는 것

| # | 시도 | 결과 | 근거 |
|---|------|------|------|
| 1 | **모델 스케일업** (yolo26l/x) | 무효 | mAP50-95 ≈ 0.72 천장 동일 |
| 2 | **앙상블** (NMS/WBF) | 무효 | v17+v16+v13 모두 v17 단독과 동일 |
| 3 | **고해상도 추론** (1536/1920) | 역효과 | F1 -0.6~1.4%p 하락 |
| 4 | **TTA** (multi-scale) | 역효과 | FP 증가, F1 하락 |
| 5 | **데이터 단순 증량** | 역효과 | v20(12.5K)이 v19(10.8K)보다 나쁨 |
| 6 | **현장 데이터만 사용** | 부족 | v23(4K)=0.912 < v17(10.5K)=0.918 |

### 2.3 핵심 병목 요인

| # | 병목 | 현재 영향 | 설명 |
|---|------|----------|------|
| 1 | **94% 오류 = 초소형 객체** | F1 천장 | <0.1% area (~20px), 물리적 한계 |
| 2 | **mAP50-95 천장 = 0.72** | 개선 불가 | GT ±2-3px 오차 → IoU 0.73 |
| 3 | **GT 미라벨링** | F1 -2~3%p | 132 FP 중 50-60개는 실제 정답 |
| 4 | **Train/Val bbox 크기 차이** | 평가 왜곡 | Train tiny 6.2% vs Val tiny 23% |

---

## 3. v24 실패 원인 분석

### 3.1 복합 변수 문제 (Confounded Variables)

v24는 **3가지를 동시에 변경**하여 원인 분리가 불가합니다:

1. **2-class → 3-class** (fallen 추가)
2. **SGD → AdamW** (옵티마이저 변경)
3. **데이터 구성 변경** (unified_safety_all + AIHub helmet 혼합)

### 3.2 도메인 갭 (Domain Gap)

| 항목 | unified_safety_all (학습) | 실제 CCTV (추론) |
|------|--------------------------|-----------------|
| 촬영 환경 | 스튜디오/연출 | 건설현장 실내/외 |
| 객체 크기 | 중앙값 area 15.9% | area < 1% |
| 카메라 각도 | 정면/측면 | 상방(오버헤드) |
| 조명 | 균일 조명 | 역광/어두운 환경 |

area < 0.10 필터링 적용했으나 배경/조명/각도 차이는 여전히 존재.

### 3.3 클래스 간섭 (Task Interference)

- 2-class에서 헬멧에 집중하던 feature가 3-class에서 fallen까지 학습하며 **희석**
- 헬멧 전용 feature 공간이 fallen과 공유되어 **정밀도 하락**
- Per-class 성능 분석이 필수 (현재 aggregate 메트릭만 확인됨)

### 3.4 AdamW 리스크

- CLAUDE.md에 **"SGD 사용, AdamW 금지"** 명시
- AdamW weight_decay=0.01은 SGD의 0.0005 대비 **20배 강함**
- 3-class + AdamW 조합이 학습 불안정 초래 가능

---

## 4. 전략적 방향: 3가지 접근법

### 접근법 A: "분리 모델" 전략 (권장)

**핵심**: 헬멧 감지와 쓰러짐 감지를 별도 모델로 분리

```
Model 1: 헬멧 감지 (v19 best.pt 그대로)
  ├── person_with_helmet (class 0)
  └── person_without_helmet (class 1)
  → F1=0.928 (ignore-aware 0.958) 100% 보존

Model 2: 쓰러짐 감지 (신규 학습)
  └── fallen_person (class 0)
  → 독립 최적화, 헬멧 성능에 영향 없음

추론: 두 모델 병렬 실행 → 결과 병합 (NMS)
```

| 장점 | 단점 |
|------|------|
| 헬멧 성능 100% 보존 | 추론 시간 ~1.5배 |
| 각 모델 독립 최적화 가능 | 배포 복잡성 증가 |
| 실패 시 영향 범위 제한 | 모델 2개 관리 |
| fallen 모델만 교체 가능 | GPU 메모리 사용량 증가 |

### 접근법 B: "단계적 통합" 전략

**핵심**: 변수를 하나씩 변경하며 영향도 측정

```
Step 1: v25a = SGD + 3-class + v24 데이터
  → AdamW 영향 분리

Step 2: v25b = SGD + 2-class + v24 데이터 (fallen 제외)
  → 데이터 구성 변경 영향 분리

Step 3: 결과 비교
  → v25a 하락 < 1%p: 3-class 통합 성공
  → v25a 하락 > 1%p: 접근법 A 또는 C로 전환
```

| 장점 | 단점 |
|------|------|
| 과학적 접근 (변수 분리) | 실험 3-4회 필요 (각 8-15시간) |
| 단일 모델 배포 가능 | 헬멧 성능 하락 가능성 |
| 원인 파악 후 정확한 대응 | 시간 소요 |

### 접근법 C: "데이터 중심 통합" 전략

**핵심**: v24 프레임워크 유지, 데이터 품질과 비율에 집중

```
v25 데이터 구성:
├── v19 헬멧 데이터   10,852장 (88%)  ← 비율 대폭 증가
├── 고품질 fallen       500장  (4%)   ← 소량 고품질
├── Background neg    1,000장  (8%)
└── Total            12,352장

설정:
├── optimizer: SGD (lr0=0.005)
├── weights: v17 best.pt (warm restart, 3-class head 재초기화)
├── fallen을 소수 클래스로 취급
└── Per-class loss weighting 고려
```

| 장점 | 단점 |
|------|------|
| 단일 모델, 간단한 배포 | 헬멧 성능 소폭 하락 가능 |
| 헬멧 데이터 비율 유지 | Fallen 성능 제한적 |
| SGD 안정성 확보 | 데이터 선별 노력 필요 |

---

## 5. 권장 실행 계획

### Phase 1: v24 진단 (즉시, 2-4시간)

v24 학습 완료 (epoch 100) 후:

```bash
# 1. Per-class 메트릭 확인
python eval_go3k_v18.py --model hoban_go3k_v24/weights/best.pt

# 2. 헬멧만 평가 (fallen 제외)
#    → v19 대비 helmet 성능 하락 정도 측정

# 3. Fallen val 100장 단독 평가
#    → Recall/Precision 확인
```

**판단 기준**:
- 헬멧 F1 하락 < 1%p → 통합 모델 가능성 있음
- 헬멧 F1 하락 > 3%p → 분리 모델 전략 전환
- Fallen Recall > 0.70 → 데이터 품질 양호
- Fallen Recall < 0.50 → 도메인 갭 심각, 데이터 재선별 필요

### Phase 2: 변수 분리 실험 (1-2일)

```python
# Experiment v25a: SGD + 3-class (AdamW 영향 분리)
# v24와 동일 데이터, SGD만 변경
model.train(
    optimizer="SGD", lr0=0.005, weight_decay=0.0005,
    data="datasets_go3k_v24/data.yaml",
    imgsz=1280, batch=6, epochs=100, patience=20
)

# Experiment v25b: SGD + 2-class on v24 data (fallen 영향 분리)
# v24 데이터에서 fallen만 제거
model.train(
    optimizer="SGD", lr0=0.005,
    data="datasets_go3k_v25b/data.yaml",  # fallen 제외
    imgsz=1280, batch=6, epochs=100, patience=20
)
```

**비교 매트릭스**:

| 실험 | 옵티마이저 | 클래스 | 데이터 | 기대 결과 |
|------|-----------|--------|--------|----------|
| v24 | AdamW | 3 | v24 mix | 0.870 (실측) |
| v25a | SGD | 3 | v24 mix | 0.88-0.92? |
| v25b | SGD | 2 | v24 mix-fallen | 0.93-0.95? |
| v19 | SGD | 2 | v19 데이터 | 0.960 (실측) |

### Phase 3: Fallen 데이터 품질 개선 (2-3일)

#### 현재 문제점과 개선안

| 문제 | 현재 (v24) | 개선안 |
|------|-----------|--------|
| 도메인 갭 | 스튜디오/연출 낙상 | **실내/오버헤드 각도만 선별** |
| bbox 크기 | area < 0.10 (너무 넓음) | **area 0.005-0.05** (CCTV 현실적) |
| 포즈 혼재 | 누움+앉음+쓰러짐 | **w/h > 1.5 (가로 누운 형태만)** |
| 배경 차이 | 다양한 환경 | **실내/건설현장 유사 우선** |
| 수량 | 2,000장 | **500-1,000 고품질** |

#### 고품질 Fallen 데이터 선별 기준

```python
# 1. bbox 크기 필터 (CCTV 현실적 크기)
0.005 <= area <= 0.05

# 2. 가로 비율 필터 (쓰러진 포즈)
aspect_ratio (w/h) >= 1.2

# 3. 이미지당 fallen 수 제한 (자연스러운 장면)
max_fallen_per_image <= 3

# 4. Roboflow 중복 제거 (augmentation chain)
use base_name only (strip .rf.{hash})

# 5. 수동 검수 100장 샘플링
```

### Phase 4: 최종 모델 학습 (3-5일)

#### 분리 모델 접근 시

```python
# 헬멧 모델: v19 best.pt 그대로 사용 (변경 없음)
helmet_model = "hoban_go3k_v19/weights/best.pt"

# 쓰러짐 모델: 신규 학습
fallen_model = YOLO("yolo26m.pt")  # COCO pretrained
fallen_model.train(
    data="datasets_fallen_v1/data.yaml",
    optimizer="SGD", lr0=0.005, lrf=0.01,
    imgsz=1280, batch=6, epochs=100,
    patience=20, seed=42,
    mosaic=1.0, mixup=0.1, copy_paste=0.15,
    close_mosaic=10
)
# 데이터: 고품질 fallen 500-1,000 + neg 1,000
```

#### 통합 모델 접근 시

```python
# v25 최종
model = YOLO("yolo26m.pt")  # COCO pretrained (3-class head 초기화)
model.train(
    data="datasets_go3k_v25/data.yaml",
    optimizer="SGD", lr0=0.005, lrf=0.01,
    imgsz=1280, batch=6, epochs=100,
    patience=20, seed=42,
    # augmentation: 표준 설정
)
# 데이터: v19 helmet 10,852 (88%) + fallen 500 (4%) + neg 1,000 (8%)
```

### Phase 5: 후처리 & 배포 (2-3일)

#### 3-class 후처리 파이프라인

```python
# 1. Cross-class NMS (helmet_on ↔ helmet_off)
cross_class_nms(iou_threshold=0.3)

# 2. Min area filter
min_area >= 5e-05  # 이미지 면적 대비

# 3. Full-image Gate (SAHI 아티팩트 제거)
gate_conf = 0.20
gate_radius = 30  # px

# 4. Per-class confidence
CLASS_CONFIDENCE = {
    0: 0.40,   # helmet_on (엄격)
    1: 0.15,   # helmet_off (관대, 위반 포착)
    2: 0.30,   # fallen (보수적, FP 최소화)
}

# 5. Fallen 전용 추가 필터
FALLEN_FILTERS = {
    "consecutive_frames": 30,       # 헬멧(15)보다 2배 엄격
    "aspect_ratio_min": 1.2,        # 가로로 누운 형태만
    "min_confidence_streak": 0.25,  # 연속 프레임 최소 conf
    "min_area": 1e-04,              # 극소형 fallen 제거
}
```

#### video_indoor 배포 설정 변경

```python
# 현재 (v17, 2-class)
MODEL_PATH = "hoban_go3k_v17/weights/best.pt"
CLASS_CONFIDENCE_THRESHOLD = {0: 0.40, 1: 0.15}
CONSECUTIVE_FRAMES_REQUIRED = 15

# 업데이트 (분리 모델 시)
HELMET_MODEL = "hoban_go3k_v19/weights/best.pt"
FALLEN_MODEL = "hoban_fallen_v1/weights/best.pt"
CLASS_CONFIDENCE_THRESHOLD = {0: 0.40, 1: 0.15, 2: 0.30}
FALLEN_CONSECUTIVE_FRAMES = 30  # 쓰러짐은 더 엄격한 temporal filter

# 업데이트 (통합 모델 시)
MODEL_PATH = "hoban_go3k_v25/weights/best.pt"
CLASS_CONFIDENCE_THRESHOLD = {0: 0.40, 1: 0.15, 2: 0.30}
FALLEN_CONSECUTIVE_FRAMES = 30
```

---

## 6. 성능 목표 & 현실적 기대치

### 6.1 목표 메트릭

| 메트릭 | 현재 (v19, 2-class) | 목표 (3-class) | 현실적 예상 |
|--------|---------------------|----------------|-------------|
| Helmet SAHI F1 | 0.928 | ≥ 0.920 | 0.915-0.925 |
| Helmet F1 (ignore-aware) | 0.958 | ≥ 0.950 | 0.945-0.955 |
| Helmet_off Recall | ~0.95 | ≥ 0.93 | 0.93-0.95 |
| Fallen Recall | N/A | ≥ 0.80 | **0.60-0.75** |
| Fallen Precision | N/A | ≥ 0.70 | 0.65-0.80 |
| Critical FP (위험 오탐) | 24 | ≤ 20 | 20-30 |
| Critical FN (미감지) | 6 | ≤ 10 | 8-15 |

> **분리 모델 시**: 헬멧 F1=0.928 완전 보존, fallen만 별도 평가

### 6.2 Fallen 초기 성능이 낮을 수밖에 없는 이유

1. **도메인 갭**: 학습 데이터(스튜디오) ≠ 실제 환경(CCTV)
2. **실제 이벤트 부재**: 현장에서 쓰러짐은 극히 드문 이벤트
3. **bbox 크기 차이**: 학습 area 중앙값 15.9% vs CCTV < 1%
4. **포즈 다양성**: 실제 낙상은 예측 불가능한 다양한 형태

→ **Recall 60-75%를 초기 목표로 설정하고, 현장 데이터 축적으로 점진 개선**

---

## 7. Fallen 감지 특화 분석

### 7.1 Fallen 데이터 가용성

| 출처 | 파일 수 | bbox 수 | 품질 |
|------|---------|---------|------|
| unified_safety_all train | 29,378 | 50,211 | 혼합 (스튜디오+연출) |
| unified_safety_all valid | ~5,000 | ~5,023 | 혼합 |
| v24에서 사용 | 2,000 | ~2,915 | area < 0.10 필터 |

### 7.2 Fallen 도전 과제별 대응

| 도전 과제 | 현상 | 대응 전략 |
|----------|------|----------|
| **도메인 갭** | 스튜디오 vs CCTV | 실내/오버헤드 각도 이미지만 선별 |
| **bbox 크기 불일치** | 중앙값 area 15.9% | area 0.005-0.05만 선별 |
| **포즈 혼재** | 앉음/누움/쓰러짐 구분 불가 | w/h > 1.2 가로형만 선별 |
| **이벤트 희소성** | 실제 현장 쓰러짐 거의 없음 | 높은 conf(0.30+), 30프레임 temporal |
| **FP 위험** | 앉은 사람, 쉬는 사람 오탐 | aspect ratio + temporal + conf 3중 필터 |

### 7.3 Fallen 감지 최적화 로드맵

```
Phase 1 (현재): 기존 데이터로 기본 모델 → Recall 60-75%
Phase 2 (1개월): 실제 CCTV에서 FP/FN 수집 → hard example mining
Phase 3 (2개월): 현장 실제 이벤트 축적 → domain-specific fine-tune
Phase 4 (3개월): 포즈 추정 결합 (skeleton) → Recall 85%+ 목표
```

---

## 8. 오류 분석 상세 (v19 기준)

### 8.1 FP (False Positive) 분포 — 총 132개

| 카테고리 | 수량 | 비율 | 위험도 |
|----------|------|------|--------|
| GT 미라벨링 (실제 정답) | 50-60 | 38-45% | 없음 (평가 오류) |
| 배경 형상 (표지판/장비) | 40-50 | 30-38% | 중간 |
| SAHI 경계 아티팩트 | 15-20 | 11-15% | 낮음 |
| **BG→helmet_off (위험 오탐)** | **17** | **13%** | **높음** |
| **ON→OFF 혼동** | **7** | **5%** | **높음** |

### 8.2 FN (False Negative) 분포 — 총 54개

| 카테고리 | 수량 | 비율 | 위험도 |
|----------|------|------|--------|
| 초소형 (<20px) | 53 | 98% | 물리적 한계 |
| 정상 크기 미감지 | 1 | 2% | 개선 가능 |

### 8.3 Critical Error 분류 (ignore-aware 후)

| 유형 | 수량 | 설명 | 위험 |
|------|------|------|------|
| ON→OFF 혼동 | 5 | 어두운/역광 씬 | 위험 경보 |
| BG→OFF 오탐 | 4-7 | 밝은 머리카락 | 위험 경보 |
| OFF 미감지 (FN) | 3 | 밝은 머리=헬멧 혼동 | **가장 위험** |
| 집중 이미지 | 3장 | 전체 critical error의 40% | 특정 씬 문제 |

### 8.4 가장 효과적인 개선 방법 (ROI 순)

| 방법 | 예상 개선 | 노력 | ROI |
|------|----------|------|-----|
| **GT 재라벨링** (50-60개 FP 수정) | +2-3%p F1 | 1-2일 | **최고** |
| Hard negative mining (배경 FP) | +0.5-1.0%p | 1일 | 높음 |
| 밝은 머리 학습 데이터 추가 | +0.2-0.5%p | 1-2일 | 중간 |
| P2 feature pyramid 추가 | +1-2%p (추정) | 1주+ | 낮음 |
| 모델 아키텍처 변경 (RT-DETR) | 불확실 | 2주+ | 낮음 |

---

## 9. 장기 로드맵

### 즉시 (이번 주)

- [ ] v24 완료 후 per-class 진단 (helmet vs fallen 분리 평가)
- [ ] v25a(SGD+3class) / v25b(SGD+2class) 변수 분리 실험
- [ ] Fallen 데이터 수동 검수 (100장 샘플)

### 단기 (1-2주)

- [ ] 분리 vs 통합 모델 결정 (Phase 2 결과 기반)
- [ ] 최종 모델 학습 & SAHI 평가
- [ ] 3-class 후처리 파이프라인 확장
- [ ] GT 재라벨링 (605 val set, 50+ 미라벨링 수정)

### 중기 (1개월)

- [ ] video_indoor 배포 업데이트 (v19 또는 v25)
- [ ] 실제 CCTV에서 fallen 성능 모니터링
- [ ] 실제 이벤트 수집 → hard example mining
- [ ] 다중 카메라 일반화 테스트

### 장기 (2-3개월)

- [ ] P2 feature pyramid 추가 검토 (초소형 객체)
- [ ] RT-DETR/DINO 아키텍처 탐색 (YOLO 한계 도달 시)
- [ ] 포즈 추정 결합 (skeleton-based fallen detection)
- [ ] 연간 데이터 축적 계획 수립

---

## 10. 핵심 결론

### 10.1 가장 중요한 5가지 결론

1. **v24 (3-class + AdamW)는 헬멧 성능을 크게 희생** (-9%p mAP50).
   → 변수 분리 실험이 먼저 필요합니다.

2. **분리 모델 전략을 1순위로 권장.**
   → 헬멧 v19를 그대로 유지하면서 fallen 전용 모델을 별도 최적화하는 것이 리스크 최소.

3. **통합 모델을 원한다면, SGD + 헬멧 88%+ + fallen 고품질 소량(500)**이 핵심.
   → v24처럼 fallen 2,000장(18%)은 헬멧 task를 희석시킴.

4. **Fallen 초기 성능은 겸손하게 설정** (Recall 60-75%).
   → 도메인 갭이 크므로, 현장 데이터 축적으로 점진 개선이 현실적.

5. **가장 큰 ROI는 GT 품질 개선.**
   → 132개 FP 중 50-60개가 GT 미라벨링. 수정만으로 F1 +2-3%p.

### 10.2 최적 경로 요약

```
[현재] v19 (2-class, F1=0.928)
  │
  ├─→ [즉시] v24 진단 + 변수 분리 실험
  │
  ├─→ [1주] 분리 모델 결정
  │     ├── 헬멧: v19 유지 (F1=0.928)
  │     └── Fallen: 전용 모델 학습 (Recall 60-75%)
  │
  ├─→ [2주] 후처리 최적화 + 배포
  │
  ├─→ [1개월] 현장 데이터 수집 + GT 재라벨링
  │     ├── 헬멧: F1 → 0.950+ (GT 수정 효과)
  │     └── Fallen: Recall → 0.80+ (현장 데이터 효과)
  │
  └─→ [3개월] 아키텍처 개선 (P2/RT-DETR) + 포즈 추정 결합
        └── 최종 목표: 3-class F1 ≥ 0.95
```

---

## 부록: 프로젝트 구조 참조

```
/home/lay/hoban/
├── .claude/CLAUDE.md                        # 프로젝트 규칙
├── report_strategic_analysis.md             # 이 문서
├── report_v19_analysis.md                   # v19 오류 분석
├── plan_v24_final_model.md                  # v24 설계 문서
├── plan_f1_breakthrough.md                  # FP/FN 분석
├── plan_next_steps.md                       # 실험 요약
│
├── train_go3k_v{17,19,20,21_l,22,23,24}.py # 학습 스크립트
├── eval_go3k_v18.py                         # SAHI 평가
├── eval_ignore_aware.py                     # Ignore-aware 평가
├── eval_v17_postprocess.py                  # 후처리 실험
├── prepare_v24_dataset.py                   # v24 데이터 준비
│
├── datasets_go3k_v{16,19,20,23,24}/        # 학습 데이터셋
├── hoban_go3k_v{17,19,20,21_l,22,23,24}/   # 학습 결과 (weights/)
│
└── docs/                                    # 기술 문서
    ├── v17_report.md
    ├── IMPROVEMENT_PLAN.md
    └── DATA_QUALITY_ISSUES.md
```
