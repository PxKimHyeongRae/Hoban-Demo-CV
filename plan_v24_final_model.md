# v24 완성형 모델 계획 (2026-02-22)

## 목표

3-class 건설현장 안전 감지 최종 모델:
- **오탐(FP) / 미탐(FN) 최소화**
- **정확도 최대화** (SAHI F1 > 0.95 ignore-aware 기준)
- **fallen 탐지 추가** (3-class)

## 클래스 구성

| ID | Class | 설명 |
|----|-------|------|
| 0 | person_with_helmet | 헬멧 착용자 |
| 1 | person_without_helmet | 헬멧 미착용자 |
| 2 | fallen | 쓰러진 사람 |

---

## 1. 데이터셋 설계

### 1.1 현재 상황

| 소스 | 이미지 수 | 특성 |
|------|-----------|------|
| v19 현장 데이터 (non-S2) | ~4,470 | CCTV, tiny object, 최고 품질 |
| v16 외부 데이터 (S2-*) | ~8,000 | AIHub 외부, 큰 bbox, 도메인 갭 |
| v19 best F1 | 0.928 (기존) / 0.958 (ignore) | 2-class 기준 |

### 1.2 v24 데이터 구성 계획

```
v24 Train (~11,000장 목표):
├── 현장 CCTV (helmet)     ~4,400장  ← v19 non-S2 (검증된 데이터)
├── AIHub helmet (외부)     ~2,000장  ← /data/helmet/ 에서 선별
├── Fallen                  ~2,000장  ← /data/unified_safety_all/ 에서 추출
│   ├── small (area<0.10)  ~1,000장  (CCTV 유사)
│   └── medium (0.10~0.30) ~1,000장  (스케일 다양성)
├── Background (negative)   ~2,500장  ← /data/aihub_data/helmet_negative_10k_v3/
└── Hard negative             ~100장  ← 기존 오탐 이미지
    ─────────────────────────────
    합계                  ~11,000장
```

### 1.3 데이터 소스 상세

#### A. 현장 CCTV (4,400장)
- **소스**: v19 train에서 S2-* prefix 제외한 이미지
- **특성**: cam1/cam2 CCTV, tiny bbox, 실제 배포 환경과 동일
- **처리**: 그대로 사용 (class 0, 1)

#### B. AIHub helmet 외부 (2,000장)
- **소스**: `/data/helmet/train/images/` (20,945장 중 선별)
- **주의**: class 순서 반전 → `0:without → 1(우리)`, `1:with → 0(우리)`로 매핑
- **선별 기준**:
  - bbox area 0.001~0.05 (CCTV와 유사한 중소형 bbox만)
  - 이미지당 bbox 2개 이상 (복수 인물 장면 우선)
  - helmet_on 1,400장 + helmet_off 600장 (비율 7:3)
- **목적**: 일반화 능력 유지, CCTV 외 환경에서도 동작

#### C. Fallen (1,000장)
- **소스**: `/data/unified_safety_all/train/` (29,378장 중 선별)
- **현실**: median bbox area 15.9% → 대부분 근접/연출 사진, CCTV와 도메인 갭 큼
- **선별 기준**:
  - bbox area < 0.10 (이미지 대비 10% 미만, 상대적으로 작은 것만)
  - 이미지 해상도 640px 이상
  - 실외 건설현장 유사 배경 우선 (실내/연출 제외)
  - Roboflow augmentation 중복 제거 (`.rf.` hash 기준 원본만)
- **클래스 매핑**: unified class 4 → 우리 class 2
- **보완**: fallen 이미지에 helmet annotation이 없으므로 (co-labeling 0%) fallen만 라벨링

#### D. Background negative (500장)
- **소스**: `/data/aihub_data/helmet_negative_10k_v3/` (9,994장 중 선별)
- **특성**: YOLO-verified 사람 없는 건설현장 이미지
- **목적**: 배경 오탐(BG→OFF, BG→ON) 감소
- **선별**: 무작위 500장 (빈 라벨 포함)

#### E. Hard negative (100장)
- **소스**: v19 오탐 이미지 중 BG→OFF, ON→OFF 패턴
- **목적**: 기존 오탐 패턴 직접 학습
- **처리**: 빈 라벨 또는 올바른 라벨로 재라벨링

### 1.4 Val 데이터

```
v24 Val (~605장 + fallen):
├── 기존 val (3k+extra)  605장  ← helmet 평가 (검증됨)
└── fallen val           ~100장  ← unified_safety_all/valid/ 에서 추출
    ──────────────────────────
    합계                 ~705장
```

- helmet 평가는 기존 605장 그대로 유지 (비교 가능성)
- fallen val은 별도 추출 (train과 겹치지 않게)

---

## 2. 데이터 파이프라인

### 2.1 스크립트 구성

```
prepare_v24_dataset.py
├── Phase 1: 현장 데이터 복사 (v19 non-S2)
├── Phase 2: AIHub helmet 선별 + class 매핑
├── Phase 3: Fallen 추출 + 필터링 + class 매핑
├── Phase 4: Background negative 선별
├── Phase 5: Hard negative 수집
├── Phase 6: Val 구성
├── Phase 7: 중복 검사 (train-val 겹침 방지)
└── Phase 8: data.yaml 생성 + 통계 출력
```

### 2.2 Class 매핑 상세

| 소스 | 원본 class | → v24 class | 비고 |
|------|-----------|-------------|------|
| v19 현장 | 0 (with_helmet) | 0 | 그대로 |
| v19 현장 | 1 (without_helmet) | 1 | 그대로 |
| /data/helmet/ | 0 (without_helmet) | 1 | **반전** |
| /data/helmet/ | 1 (with_helmet) | 0 | **반전** |
| unified_safety | 4 (fallen) | 2 | 매핑 |
| unified_safety | 0,1,2,3,5,6,7 | 무시 | 비관련 class 제거 |
| negative | (없음) | (빈 라벨) | 그대로 |

### 2.3 Fallen 데이터 전처리 주의사항

1. **Roboflow 중복 제거**: 파일명의 `.rf.{hash}` 부분에서 원본 식별, 동일 원본의 augmentation 중 1장만 사용
2. **도메인 필터링**: bbox area > 0.10인 근접 사진 제외 (CCTV에서 쓰러진 사람은 작게 보임)
3. **co-labeling 부재**: fallen 이미지에 helmet annotation 없음 → fallen만 학습
   - 이로 인해 모델이 fallen 이미지에서 helmet 탐지를 시도하지 않도록 학습됨
   - 의도적: fallen 상황에서는 helmet 여부보다 쓰러짐 자체가 중요

---

## 3. 학습 전략

### 3.1 모델 설정

```python
# Base
model = "yolo26m.pt"  # COCO pretrained (v17에서 검증된 최적)
imgsz = 1280
batch = 6  # RTX 4080 16GB 한계
device = "0"

# Optimizer
optimizer = "AdamW"
lr0 = 0.001
lrf = 0.01
weight_decay = 0.01
cos_lr = True
warmup_epochs = 3.0

# Augmentation
mosaic = 1.0
mixup = 0.1
copy_paste = 0.15
hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.4
scale = 0.5
translate = 0.1
degrees = 5.0
fliplr = 0.5
erasing = 0.15
close_mosaic = 10

# Training
epochs = 100
patience = 20
seed = 42
```

### 3.2 학습 방식: Scratch from COCO pt

- v17 경험: COCO pt가 domain-specific pt보다 +2%p 우수
- 3-class로 변경되므로 v19 weights 재사용 불가 (head 구조 변경)
- COCO pretrained → full training이 가장 안정적

### 3.3 대안: 2-stage 학습

v19 weights를 최대한 활용하려면:

```
Stage 1: v19 best.pt의 backbone/neck weights만 추출
         → 3-class head 새로 초기화
         → lr0=0.005, 100ep (full training)

또는

Stage 1: COCO pt → 3-class, 100ep (기본)
Stage 2: Stage 1 best.pt → fine-tune with 더 높은 fallen 비율, 30ep
```

**권장: COCO pt 단일 학습** (가장 안정적, v17에서 검증됨)

---

## 4. 후처리 파이프라인

### 4.1 기존 파이프라인 확장

```python
# 1. Cross-class NMS (helmet_on ↔ helmet_off만 적용)
cross_class_nms(iou_threshold=0.3, classes=[0, 1])
# fallen(2)은 helmet과 겹쳐도 유지 (쓰러진 사람이 헬멧 쓸 수 있음)

# 2. Min area filter
min_area >= 5e-05  # 기존과 동일

# 3. Full-image Gate (SAHI 아티팩트 제거)
gate_conf = 0.20, radius = 30px

# 4. Per-class confidence
helmet_on >= 0.40    # (기존)
helmet_off >= 0.15   # (기존)
fallen >= 0.30       # (새로, 오탐 방지)
```

### 4.2 Fallen 전용 후처리

- **aspect ratio 필터**: 쓰러진 사람은 가로로 긴 bbox (w/h > 1.0) → 세로로 긴 bbox는 FP 가능
- **temporal consistency** (video_indoor): fallen은 연속 30프레임 이상 유지해야 이벤트 발생
  - 기존 helmet 15프레임보다 길게: 실제 쓰러짐은 지속적, 일시적 구부림과 구분

---

## 5. 평가 계획

### 5.1 평가 메트릭

```
A. Helmet 평가 (기존 605장)
   - SAHI F1 (ignore-aware, area < 0.00020)
   - Per-class: helmet_on, helmet_off
   - 기존 v19 F1=0.958과 직접 비교

B. Fallen 평가 (새 val ~100장)
   - mAP50, Recall
   - FP 수 (배경/앉은자세 오탐)

C. 종합
   - 3-class mAP50, mAP50-95
   - Critical error count (OFF false alarm + missed danger + fallen miss)
```

### 5.2 성공 기준

| 메트릭 | 목표 | 비고 |
|--------|------|------|
| Helmet SAHI F1 (ignore) | ≥ 0.955 | v19 0.958 대비 ≤0.3%p 하락 허용 |
| Helmet_off Recall | ≥ 0.95 | 미착용 놓치면 안 됨 |
| Fallen Recall | ≥ 0.80 | 초기 목표 (데이터 도메인 갭 감안) |
| Fallen FP (605장) | ≤ 5 | helmet 평가셋에서 fallen 오탐 극소 |
| Critical errors | ≤ 15 | v19 수준 유지 |

---

## 6. 리스크 및 대응

### Risk 1: 3-class 전환으로 helmet 성능 하락
- **원인**: 모델 capacity 분산, fallen 학습이 helmet에 간섭
- **대응**: helmet 데이터 비율 유지 (80%+), fallen은 보조
- **모니터링**: 학습 중 helmet-only val F1 추적
- **fallback**: helmet 성능 하락 > 1%p면 fallen 데이터 축소

### Risk 2: Fallen 도메인 갭 (연출 vs CCTV)
- **원인**: unified_safety_all의 fallen이 대부분 근접/연출 사진
- **대응**: bbox area < 0.10 필터링, CCTV 유사 이미지만 선별
- **보완**: 실제 CCTV에서 자세 변경(구부림, 앉기) 프레임 수집 → hard negative
- **장기**: 실제 현장 fallen 이벤트 발생 시 데이터 축적

### Risk 3: AIHub class 매핑 오류
- **원인**: /data/helmet/의 class 순서가 hoban과 반대
- **대응**: 매핑 후 100장 샘플링하여 시각적 검증 (bbox + class 확인)

### Risk 4: Fallen ↔ Person 혼동
- **원인**: 구부린 자세, 앉은 자세가 fallen으로 오탐
- **대응**: fallen conf threshold 높게 설정 (0.30+), temporal 필터 강화
- **장기**: 자세 분류 2-stage (탐지 → 자세 분류) 고려

---

## 7. 실행 순서

```
Phase 1: 데이터 준비 (2-3시간)
├── prepare_v24_dataset.py 작성
├── 현장 데이터 복사 (v19 non-S2)
├── AIHub helmet 선별 + class 매핑
├── Fallen 추출 + 필터링
├── Background negative 선별
├── 중복 검사 + data.yaml 생성
└── 100장 샘플링 시각 검증

Phase 2: 학습 (8-12시간)
├── train_go3k_v24.py 작성
├── COCO pt → 3-class, 100ep
└── 학습 모니터링 (box_loss, cls_loss 추이)

Phase 3: 평가 (1시간)
├── eval_v24.py (기존 eval_ignore_aware.py 확장)
├── Helmet 605장 SAHI F1
├── Fallen val 100장 mAP
└── v19 대비 비교표 출력

Phase 4: 후처리 최적화 (1시간)
├── per-class conf sweep (3-class)
├── fallen aspect ratio 필터 테스트
└── 최적 파라미터 확정

Phase 5: 배포 준비
├── video_indoor/app.py 업데이트 (3-class 지원)
├── fallen 이벤트 로직 추가
├── temporal smoothing 파라미터 조정
└── 24시간 모니터링
```

### 예상 소요

| Phase | 소요 | GPU |
|-------|------|-----|
| 데이터 준비 | 2-3시간 | 부분적 (검증용) |
| 학습 | 8-12시간 | 전체 |
| 평가 | 1시간 | 전체 |
| 후처리 | 1시간 | 부분적 |
| 배포 | 30분 | 없음 |

---

## 8. 파일 구조

```
# 데이터셋
datasets_go3k_v24/
├── data.yaml                 # nc=3, names=[with_helmet, without_helmet, fallen]
├── train/
│   ├── images/               # ~8,000장
│   └── labels/               # class 0,1,2
└── valid/
    ├── images/               # ~705장
    └── labels/

# 스크립트
prepare_v24_dataset.py        # 데이터셋 빌드 파이프라인
train_go3k_v24.py             # 학습 스크립트
eval_v24.py                   # 3-class 평가 (ignore-aware 포함)

# 모델 출력
hoban_go3k_v24/weights/{best,last}.pt
```

---

## 9. 핵심 요약

| 항목 | 내용 |
|------|------|
| **클래스** | 3개 (helmet_on, helmet_off, fallen) |
| **Train** | ~8,000장 (현장 4,400 + AIHub 2,000 + fallen 1,000 + neg 600) |
| **Val** | ~705장 (helmet 605 + fallen 100) |
| **모델** | yolo26m COCO pt, 1280px, SGD, 100ep |
| **목표** | Helmet F1≥0.955, Fallen Recall≥0.80, Critical errors≤15 |
| **핵심 전략** | 현장 데이터 중심 + 소량 외부 보강 + fallen 추가 |
