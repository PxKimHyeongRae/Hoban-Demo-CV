# v9 데이터셋 핵심 개선 과제

## 최대 문제: Helmet 이미지에 Person 라벨 부재

### 문제 구조

현재 helmet_60k 데이터(60,000장)의 라벨:
```
[helmet 이미지]
- cls 0 (helmet_o): 머리 bbox만 존재
- cls 1 (helmet_x): 머리 bbox만 존재
- cls 2 (person): ❌ 없음
- cls 3 (fallen): ❌ 없음
```

실제 이미지에는 **사람의 전신이 보이지만, 머리 bbox만 라벨링**되어 있다.

### 왜 심각한가

모델이 학습할 때 이런 일이 발생한다:

```
[학습 시]
helmet 이미지 → "여기에 머리(helmet) bbox가 있다" ✅
             → "그 외 영역(사람 몸통)은 background다" ❌ ← 잘못된 학습

[추론 시]
실제 현장 → 사람 몸통이 보이면?
         → "이건 background니까 무시하자" → person miss 발생
```

**helmet 데이터가 전체의 50% (30K+30K / 117K bbox)**를 차지하므로, 모델의 절반 학습이 "사람 몸통 = background"라는 잘못된 신호를 받고 있다.

이것이 person miss rate 21% (121건)의 주요 원인일 가능성이 높다.

### confusion matrix 근거

```
                 helmet_o  helmet_x  person  fallen  background
person 예측 →       0         1       467      0       154
background →       19        24       121    114        -
```

- person 정탐: 467건 (79%)
- person 누락: **121건 (21%)** ← helmet 이미지에서 person 미학습의 영향
- person background FP: **154건** ← 반대로 person을 학습한 이미지에서는 background가 부족

---

## 해결 방안: Auto-labeling

### 방법: Pre-trained 모델로 Person bbox 자동 생성

```
[파이프라인]
1. COCO pre-trained YOLO (person 탐지 성능 검증됨) 로드
2. helmet_60k 이미지 60,000장에 대해 추론
3. person (COCO cls 0) 탐지 결과 중 confidence >= threshold 필터
4. 기존 helmet label (cls 0, 1)에 person label (cls 2) 추가
5. 머리 bbox와 겹치는 person bbox만 유지 (검증)
```

### 장점
- 수작업 불필요 (60,000장을 사람이 라벨링하는 건 비현실적)
- COCO pre-trained YOLO의 person 탐지는 이미 검증된 성능
- helmet bbox와 person bbox의 공간적 포함관계로 교차 검증 가능

### 고려사항

| 항목 | 내용 |
|------|------|
| **모델 선택** | COCO pre-trained YOLOv8/v11 등 (person class만 사용) |
| **confidence threshold** | 높게 설정 (0.5~0.7) → 오탐 최소화, 정확한 것만 추가 |
| **bbox 겹침 검증** | helmet bbox가 person bbox 내부에 포함되는지 IoA 확인 |
| **크기 필터** | area < 2% person은 제외 (기존 COCO 필터 기준 유지) |
| **기존 라벨 보존** | helmet label은 그대로, person label만 추가 |

### 검증 방법
```
1. 랜덤 100장 시각화 → 사람이 눈으로 auto-label 품질 확인
2. helmet bbox 1개당 person bbox 1개가 매칭되는지 비율 확인
3. auto-label 추가 전/후 서브셋 실험으로 효과 수치 확인
```

### 예상 효과

| 지표 | 현재 (v9_sub) | 예상 (auto-label 후) |
|------|--------------|---------------------|
| person miss rate | 21% (121건) | 10% 이하로 감소 기대 |
| person mAP50 | 0.795 | 0.85+ 기대 |
| person BG FP | 154건 | 함께 감소 기대 |

helmet 이미지에서 person도 학습하면:
- "사람 몸통 = background" 잘못된 학습 제거
- person 탐지 일반화 능력 향상
- helmet + person 동시 탐지 시너지 (실제 현장에서는 항상 함께 등장)

---

## 함께 개선할 사항

### 1. Negative samples 확대 (1K → 3~5K)

| 현재 | 문제 | 개선 |
|------|------|------|
| negative 1,000장 | Background FP 535건 | 3,000~5,000장으로 확대 |

소스:
- COCO에서 person 없는 이미지 추출
- 산업현장/공사현장 빈 배경 이미지
- aihub background 폴더 추가 활용

### 2. Fallen 이미지에도 Person 라벨 추가 검토

fallen 이미지에는 "쓰러진 사람"이 있지만 fallen(cls 3) label만 있음.
쓰러지기 전 서 있는 사람이 같은 프레임에 있을 수 있으나 라벨 없음.
→ helmet과 동일한 auto-labeling 적용 가능

### 3. 에폭 + 데이터 양 확대 (이미 준비 완료)

- 30K bbox/cls, 300에폭, patience=30
- 학습 곡선이 30에폭에서 수렴 안 됨 → 에폭 증가 효과 확실

---

## 실행 우선순위

```
[즉시] 서버 학습 실행 (30K/300에폭) - 이미 준비 완료
  ↓
[병렬] Auto-labeling 파이프라인 구축
  → pre-trained YOLO로 helmet 이미지에 person bbox 자동 추가
  → 100장 시각 검증 → 품질 확인 후 전량 적용
  ↓
[병렬] Negative samples 확대 (1K → 3~5K)
  ↓
[이후] Auto-label 반영 데이터로 재학습 (v10)
  → person miss rate 개선 확인
```

## 요약

> helmet 데이터 60,000장에 person 라벨이 없는 것이 **person miss 21%의 근본 원인**.
> Pre-trained 모델로 auto-labeling하면 수작업 없이 해결 가능.
> 단, auto-label 품질 검증(confidence threshold + bbox 겹침 + 시각 확인)이 필수.
