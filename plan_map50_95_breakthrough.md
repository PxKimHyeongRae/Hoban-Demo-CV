# mAP50-95 천장 돌파 분석 (2026-02-22)

## 1. 현상: mAP50-95 = 0.72 천장

모든 버전에서 mAP50-95가 0.72 부근에서 정체:

| Version | Model | Data | mAP50 | mAP50-95 | **Gap** |
|---------|-------|------|-------|----------|---------|
| v17 | yolo26m | 10,564 | 0.958 | 0.722 | 0.236 |
| v19 | yolo26m | 10,852 | 0.959 | 0.724 | 0.235 |
| v20 | yolo26m | 12,470 | 0.961 | 0.727 | 0.234 |
| v21-l | yolo26l | 12,470 | 0.958 | 0.723 | 0.235 |
| v23 | yolo26m | 4,038 | 0.951 | 0.704 | 0.247 |

**mAP50은 0.96으로 높지만 mAP50-95와의 gap이 0.23~0.25로 일정.**
이 gap은 "탐지는 되지만 bbox가 정확하지 않다"를 의미.

---

## 2. 근본 원인 분석

### 2.1 원인 #1: Val GT의 87-96%가 tiny object

```
Val GT bbox 크기 분포:
  helmet_on  (869개): tiny(<0.1%) = 87%, small(0.1-0.5%) = 13%
  helmet_off  (79개): tiny(<0.1%) = 96%, small(0.1-0.5%) =  4%

  Median bbox: 0.015w × 0.027h = ~19×35 pixels (at 1280px)
  Min bbox: 0.008w × 0.008h = ~10×10 pixels
```

**왜 이것이 문제인가:**

tiny bbox(19×35px)에서의 IoU 민감도:
```
위치 오차:
  2px offset → IoU = 0.730 (mAP75에서 miss)
  3px offset → IoU = 0.626 (mAP65에서 miss)
  5px offset → IoU = 0.462 (mAP50에서도 miss)

크기 오차:
  ±1px expand → IoU = 0.856
  ±2px expand → IoU = 0.741
  ±3px expand → IoU = 0.649

비교: 큰 bbox(100×150px)에서는:
  5px offset → IoU = 0.849
  10px offset → IoU = 0.724
```

**결론: 19×35px bbox에서 2-3px 오차는 IoU를 0.3 이상 떨어뜨린다.**
mAP50-95의 IoU=0.75, 0.80, 0.85, 0.90, 0.95 구간에서 거의 모든 tiny object가 miss됨.

### 2.2 원인 #2: Train-Val bbox 크기 불일치 (10배 차이)

```
v16 Train (15,561 bboxes):
  tiny: 26%, small: 33%, medium: 36%, large: 5%
  Median area: 0.00423 (0.42%)

Val (948 bboxes):
  tiny: 87-96%, small: 4-13%, medium: 0%, large: 0%
  Median area: 0.00041 (0.04%)

→ Train median이 Val median보다 10배 큼!
```

모델은 주로 medium/large bbox에서 regression을 학습하지만,
평가는 거의 모든 게 tiny bbox → bbox regression 정밀도가 전이되지 않음.

**예외 - v23 (onsite only):**
```
v23 Train (5,955 bboxes):
  tiny: 89%, small: 11%, medium: 0%, large: 0%
  Median area: 0.00040 (0.04%) — Val과 동일!
```
그런데 v23의 mAP50-95=0.704로 오히려 더 낮음.
→ tiny object만으로는 4,038장으로 학습이 부족 (bbox regression 학습에 다양한 스케일 필요)

### 2.3 원인 #3: box_loss의 train-val gap

```
v17:  train=0.860, val=0.878, gap=0.017 (early stop, 41ep)
v19:  train=0.701, val=0.868, gap=0.167 (86ep, overfitting)
v20:  train=0.626, val=0.841, gap=0.215 (100ep, overfitting)
v23:  train=0.734, val=0.918, gap=0.184 (96ep)
```

- train box_loss는 계속 하락하지만 val box_loss는 0.82-0.88에서 정체
- 특히 v20/v23에서 train-val gap이 커짐 = bbox regression overfitting
- **모델이 train의 큰 bbox에는 정확해지지만, val의 tiny bbox에는 전이 안 됨**

### 2.4 원인 #4: YOLO의 구조적 한계 (tiny object bbox regression)

- YOLO v8/v26는 anchor-free, P3-P5 feature pyramid (stride 8, 16, 32)
- stride 8에서 feature map 1 pixel = 원본 8×8px
- 19×35px 객체 = feature map에서 2.4×4.4 cells → 매우 거친 regression
- **P2 layer (stride 4)가 없으면** tiny object의 sub-pixel regression이 구조적으로 어려움

### 2.5 원인 #5: Annotation 정밀도의 한계

- 19×35px에서 annotator가 2-3px 이내로 정확히 bbox를 그리기 어려움
- GT 자체의 불확실성이 ±2-3px이면, 이론적 mAP50-95 상한이 존재
- 특히 CCTV의 흐릿한 소형 객체에서 helmet/person 경계가 모호

---

## 3. 돌파 전략 (우선순위순)

### Strategy A: Annotation 품질 검증 (즉시, 2시간)

**근거**: GT 자체가 부정확하면 어떤 모델도 mAP50-95를 올릴 수 없음.

**방법**:
1. Val GT 중 tiny bbox 상위 50개를 시각화 (크롭 확대)
2. Annotator 간 일관성 확인 (같은 객체의 다른 프레임에서 bbox 변동)
3. IoU=0.75 기준으로 모델 pred와 GT의 실제 offset 분포 측정
4. GT annotation error가 ±2px 이상이면, mAP50-95 천장이 GT 품질에 의해 결정됨을 확인

**예상**: GT 정밀도가 ±3px이면 이론적 mAP50-95 상한 = ~0.75
→ 현재 0.72는 이미 상한 근처

**결론 시나리오**:
- GT가 정확 → 모델 개선 여지 있음 → Strategy B, C 진행
- GT가 부정확 → mAP50-95 개선은 GT re-annotation 필요 → Strategy E

### Strategy B: Train 데이터의 tiny object 비중 증가 (1일)

**근거**: Train의 tiny 26% vs Val의 tiny 87%. 이 불일치를 줄이면 bbox regression이 개선될 수 있음.

**방법 B-1: Tiny object oversampling**
- v16 train에서 tiny bbox(area<0.001) 포함 이미지만 추출
- 이 이미지를 2-3x 반복 (oversampling)
- 또는 weighted dataset sampler 사용

**방법 B-2: Mosaic에서 tiny object 우선 배치**
- YOLO mosaic augmentation에서 tiny bbox 이미지를 우선 배치하는 custom sampler

**방법 B-3: Large object downsampling**
- Train에서 medium/large bbox만 있는 이미지의 비중을 줄여서
- tiny:medium 비율을 현재 26:36에서 50:50으로 맞춤

**예상 효과**: +0.01~0.02 mAP50-95

### Strategy C: Crop-based Training (2일)

**근거**: 1920×1080 이미지에서 19×35px 객체 → 비율 극히 작음.
객체 주변을 crop해서 학습하면 effective resolution 증가.

**방법**:
1. 원본 이미지에서 GT bbox 중심으로 640×640 crop
2. crop 내에서 객체가 차지하는 비율 증가 (19×35 → crop 대비 3% vs 원본 대비 0.04%)
3. crop 데이터로 추가 학습 또는 mixed training (원본 + crop)

**주의**: SAHI 추론이 이미 tile 기반이므로, tile과 유사한 학습이 효과적일 수 있음

**예상 효과**: +0.02~0.04 mAP50-95 (crop에서의 regression 정밀도 향상)

### Strategy D: Loss Function 개선 (1일)

**근거**: 기본 CIoU loss는 tiny object에서 gradient가 약함.

**방법 D-1: Inner-IoU Loss**
- bbox 내부의 작은 영역에서 IoU를 계산 → tiny object에서 gradient 강화
- Ultralytics에서 custom loss 적용 필요

**방법 D-2: Wise-IoU (WIoU) v3**
- 동적 non-monotonic focusing mechanism
- 품질이 낮은 bbox에 대한 gradient 조정
- tiny object의 noisy supervision 문제 완화

**방법 D-3: box loss weight 증가**
- 현재 box=7.5 (기본값)
- box=15.0으로 올려서 bbox regression에 더 큰 gradient 할당
- 단순하지만 효과적일 수 있음

**예상 효과**: +0.01~0.03 mAP50-95

### Strategy E: GT Re-annotation (3-5일)

**근거**: Strategy A에서 GT 품질이 문제로 확인된 경우.

**방법**:
1. Val set의 tiny bbox를 확대 뷰에서 re-annotation
2. 반자동: 모델 pred를 기반으로 annotator가 미세 조정
3. 일관성 가이드라인: helmet 경계 정의 (머리 상단부터 턱 아래까지 등)
4. Inter-annotator agreement 측정

**예상 효과**: GT 품질 개선 → 이론적 mAP50-95 상한 증가

### Strategy F: Feature Pyramid 확장 - P2 추가 (3-5일)

**근거**: P3(stride=8)로는 19×35px 객체가 2.4×4.4 feature cells밖에 안 됨.
P2(stride=4)를 추가하면 4.8×8.8 cells로 2배 정밀.

**방법**:
- YOLO v8/v26 모델의 neck에 P2 layer 추가
- 또는 YOLO 소형 객체 특화 변형 (YOLOv8-P2 등) 사용
- VRAM 사용 증가 (batch 4→2 가능성)

**예상 효과**: +0.02~0.05 mAP50-95 (소형 객체 regression 정밀도 근본 개선)
**리스크**: custom architecture 수정 필요, 추론 속도 저하

### Strategy G: Multi-scale Training with Tiny Focus (2일)

**근거**: 현재 1280px 고정. multi_scale로 다양한 스케일에서 학습하면
모델이 스케일 불변 regression을 학습할 수 있음.

**방법**:
- `multi_scale=0.5` (imgsz ±50% 변동: 640~1920px)
- 또는 imgsz=1536으로 학습 (tiny object가 feature map에서 더 큼)
  - batch=3~4 (VRAM 한계)
  - 1536px에서 19px 객체 → stride=8 feature map에서 2.4 cells → 약간 개선

**예상 효과**: +0.005~0.015 mAP50-95

---

## 4. 검증 필요 사항

### 4.1 GT 품질 검증 (최우선)
- [ ] Val tiny bbox 50개 크롭 시각화 → 사람이 볼 수 있는 수준인지?
- [ ] GT bbox가 ±2px 이내로 정확한지?
- [ ] 동일 객체 다른 프레임에서 GT bbox 일관성?
- [ ] GT가 부정확한 경우 이론적 mAP50-95 상한 계산

### 4.2 모델 예측 분석 (Strategy A와 함께)
- [ ] 모델 pred vs GT의 IoU 분포 (0.5~0.95 구간별 히스토그램)
- [ ] tiny/small 별 IoU 분포 차이
- [ ] 어느 IoU 구간에서 가장 큰 손실이 발생하는지

### 4.3 Train-Val 분포 정합 실험 (Strategy B)
- [ ] v16 train에서 tiny-only subset으로 fine-tune 시 val box_loss 개선?
- [ ] v23의 낮은 mAP가 데이터 부족인지 분포 매칭 덕인지 분리

---

## 5. 추천 실행 순서

```
Day 1 AM: Strategy A (GT 품질 검증) — 2시간
  │
  ├─ GT 정확 → Day 1 PM: Strategy D-3 (box loss weight 증가, 간단) — 8시간 학습
  │                + Strategy B-3 (large downsampling) — 준비 1시간
  │
  └─ GT 부정확 → 이론적 상한 계산 후 의사결정
                   ├─ 상한 > 0.80 → Strategy E (re-annotation)
                   └─ 상한 ≈ 0.75 → mAP50-95 개선 포기, SAHI F1에 집중

Day 2: Strategy C (crop-based training) — 12시간

Day 3: 결과 종합 평가 → Strategy F (P2 layer) 결정
```

## 6. 핵심 결론

**mAP50-95 = 0.72 천장의 근본 원인은 "tiny object":**

1. Val의 87-96%가 tiny (19×35px) → 2-3px 오차에 IoU 급락
2. Train은 medium 위주 (median 10배 차이) → bbox regression 전이 안 됨
3. YOLO P3(stride=8)에서 2.4×4.4 cells = 구조적 정밀도 한계
4. GT annotation 자체의 ±2-3px 불확실성 → 이론적 상한 존재 가능

**가장 먼저 해야 할 것: GT 품질 검증** — GT가 부정확하면 모든 모델 개선이 무의미.
GT가 정확하다면: crop training + bbox loss 강화 + P2 layer가 돌파 후보.
