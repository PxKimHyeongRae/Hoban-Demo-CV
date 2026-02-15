# GO2K 개발 일지

> 2026-02-15 작성

## 개요

현장 CCTV 스냅샷 1,815장에서 수동 라벨링 672장(go2k_manual)을 기반으로,
SAHI 추론 파이프라인 최적화 및 파인튜닝 모델 비교 실험을 진행한 기록.

---

## 1. 데이터셋

### go2k_manual (수동 라벨링 GT)

| 항목 | 값 |
|------|-----|
| 이미지 | 672장 (1920×1080) |
| 총 bbox | 1,901개 |
| person_with_helmet (cls0) | 1,449개 |
| person_without_helmet (cls1) | 231개 |
| 소스 | CVAT 4분할 (part1~4) 수동 라벨링 |
| 경로 | `/home/lay/hoban/datasets/go2k_manual/` |

- Part1: CVAT XML (box) → 701 labels
- Part2: CVAT XML (mask → box 변환) → labels
- Part3~4: CVAT XML (box) → labels
- 오탐 제거: 고정 위치 소형 FP (w<0.03, h<0.05) 4,401개 제거

### go2k_v2 (학습용 혼합 데이터)

| 구분 | v13 서브샘플 | go2k ×8 오버샘플 | 합계 |
|------|-------------|-----------------|------|
| Train | 8,000장 | 3,832장 (479×8) | **11,832장** |
| Valid | 1,000장 | 120장 | **1,120장** |

- go2k_manual을 80/20 분리 (train 479 / valid 120)
- v13 train에서 8K 랜덤 샘플, v13 valid에서 1K 랜덤 샘플
- go2k train을 8배 복제하여 도메인 비중 ~32%
- 경로: `/home/lay/hoban/datasets_go2k_v2/`

---

## 2. 모델 학습

### go2k finetune (SGD) — go2k만으로 파인튜닝

| 항목 | 값 |
|------|-----|
| 베이스 | v13 stage2 best.pt |
| 데이터 | go2k_manual (train 479장) |
| optimizer | SGD (lr0=0.005) |
| 결과 | P=0.857, R=0.669, mAP50=0.787, mAP50-95=0.423 |
| 경로 | `/home/lay/hoban/hoban_go2k_finetune/` |

### go2k finetune (AdamW) — go2k만으로 파인튜닝

| 항목 | 값 |
|------|-----|
| 베이스 | v13 stage2 best.pt |
| 데이터 | go2k_manual (train 479장) |
| optimizer | AdamW (lr0=0.0002) |
| 경로 | `/home/lay/hoban/hoban_go2k_finetune_adamw/` |

### go2k_v2 — v13 서브샘플 + go2k 오버샘플

| 항목 | 값 |
|------|-----|
| 베이스 | v13 stage2 best.pt |
| 데이터 | datasets_go2k_v2 (11,832 train / 1,120 valid) |
| optimizer | SGD (lr0=0.005, cos_lr) |
| epochs | 97 (patience=20 조기 종료) |
| 증강 | mosaic=1.0, mixup=0.1, degrees=5, erasing=0.1 |
| Best (epoch 77) | **P=0.860, R=0.892, mAP50=0.927, mAP50-95=0.674** |
| 학습 시간 | ~5시간 |
| 경로 | `/home/lay/hoban/hoban_go2k_v2/` |

### 모델 비교 (validation 기준)

| 모델 | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| go2k finetune (SGD) | 0.857 | 0.669 | 0.787 | 0.423 |
| **go2k_v2** | **0.860** | **0.892** | **0.927** | **0.674** |
| v13 stage2 (base) | - | - | 0.945 | 0.727 |

---

## 3. SAHI 추론 평가 (go2k_manual GT 대비)

### 평가 방식

- go2k_manual 수동 라벨링 1,680 bbox를 GT로 사용 (604장, 라벨 없는 이미지 포함)
- SAHI (640×640 슬라이스, overlap 0.2) 추론 결과와 IoU≥0.5 매칭
- Precision / Recall / F1 산출

### go500 finetune + SAHI (이전 베이스라인)

| conf | 예측 | TP | FP | FN | Prec | Recall | F1 |
|------|-----:|---:|---:|---:|-----:|-------:|---:|
| 0.15 | 2,282 | ~1,538 | ~744 | ~142 | 0.674 | 0.916 | 0.777 |

### go2k_v2 + SAHI — conf 임계값 sweep

| conf | 예측 | TP | FP | FN | Prec | Recall | F1 |
|------|-----:|---:|---:|---:|-----:|-------:|---:|
| 0.15 | 2,706 | 1,591 | 1,115 | 89 | 0.588 | 0.947 | 0.725 |
| 0.20 | 2,618 | 1,586 | 1,032 | 94 | 0.606 | 0.944 | 0.738 |
| 0.25 | 2,505 | 1,579 | 926 | 101 | 0.630 | 0.940 | 0.755 |
| 0.30 | 2,419 | 1,575 | 844 | 105 | 0.651 | 0.938 | 0.768 |
| 0.35 | 2,344 | 1,567 | 777 | 113 | 0.669 | 0.933 | 0.779 |
| 0.40 | 2,265 | 1,553 | 712 | 127 | 0.686 | 0.924 | 0.787 |
| **0.50** | **2,095** | **1,517** | **578** | **163** | **0.724** | **0.903** | **0.804** |

### SAHI 후처리 파라미터 sweep

10개 SAHI 설정 조합 × 4개 conf 임계값 = 40가지 테스트

#### conf=0.35 기준 SAHI 설정 비교

| SAHI 설정 | 예측 | FP | Prec | Recall | F1 |
|-----------|-----:|---:|-----:|-------:|---:|
| NMS/0.5/IOU (기존) | 2,490 | 889 | 0.643 | 0.953 | 0.768 |
| NMS/0.3/IOU | 2,362 | 795 | 0.663 | 0.933 | 0.775 |
| NMS/0.3/IOS | 2,339 | 773 | 0.670 | 0.932 | 0.779 |
| NMS/0.4/IOS | 2,342 | 775 | 0.669 | 0.933 | 0.779 |
| NMM/0.5/IOU | 2,481 | 903 | 0.636 | 0.939 | 0.758 |
| NMM/0.3/IOS | 2,339 | 834 | 0.643 | 0.896 | 0.749 |

#### 주요 발견

1. **conf 임계값이 지배적**: 후처리 파라미터 차이는 F1 ±0.01, conf 차이는 F1 ±0.08
2. **NMS > NMM**: NMM(병합)이 오히려 성능 하락
3. **IOS ≥ IOU**: 작은 박스 기준 겹침 계산이 미세하게 유리
4. **match_threshold 0.3~0.4**: 더 공격적 중복 제거가 약간 유리

---

## 4. 최적 설정 (현재 채택)

| 항목 | 값 |
|------|-----|
| 모델 | `hoban_go2k_v2/weights/best.pt` |
| conf | **0.50** |
| SAHI slice | 640×640, overlap 0.2 |
| postprocess_type | **NMS** |
| postprocess_match_threshold | **0.4** |
| postprocess_match_metric | **IOS** |
| **Precision** | **0.725** |
| **Recall** | **0.903** |
| **F1** | **0.804** |

스크립트: `detect_go2k_sahi.py` (최적 설정 반영 완료)

---

## 5. 분석 및 한계

### FP 구성 분석

- 총 예측 2,093개 중 FP 576개 (27.5%)
- `person_with_helmet` FP가 대부분 — 타일 경계 중복 탐지 + 배경 오탐
- SAHI 타일링 구조상 경계 객체 중복은 완전 제거 불가

### v13 혼합 데이터의 영향

- mAP50: go2k finetune 0.787 → go2k_v2 **0.927** (+14%p) — validation 크게 향상
- 하지만 go2k_manual GT 대비 SAHI 평가에서는 FP 증가
- v13 범용 데이터가 go2k 도메인에서 오탐 유발 (다른 도메인 패턴)

### conf vs Recall 트레이드오프

- conf=0.15: R=0.947이지만 P=0.588 (FP 1,115개)
- conf=0.50: P=0.724이지만 R=0.903 (FN 163개)
- 용도에 따라 conf 조정 필요 (안전 모니터링은 Recall 우선)

---

## 6. 스크립트 목록

| 스크립트 | 설명 |
|----------|------|
| `train_go2k_finetune.py` | go2k만으로 파인튜닝 (SGD/AdamW) |
| `train_go2k_v2.py` | v13 서브샘플 + go2k 오버샘플 학습 |
| `train_go2k_v3.py` | 타일 학습 + 1280px + copy_paste |
| `detect_go2k_sahi.py` | SAHI 추론 (최적 설정 반영) |
| `detect_captures_gated.py` | captures SAHI+Gate pseudo-labeling |
| `dedup_captures.py` | captures 프리즈 중복 제거 |
| `eval_go2k_sahi.py` | go500 모델 SAHI 평가 |
| `eval_go2k_v2.py` | go2k_v2 모델 SAHI 평가 |
| `eval_fullimage_gate.py` | Full-Image Gate 효과 측정 |
| `eval_gate_finetune.py` | Gate 파라미터 정밀 튜닝 (48 조합) |
| `prepare_cvat_all.py` | go2k+captures CVAT 패키징 |

---

## 7. Full-Image Gate (FP 억제)

SAHI FP 문제 해결을 위해 풀이미지 추론을 게이트로 사용하는 기법 개발.

### 방법
1. 풀이미지 640px 추론 (conf=0.20) → 대략적 사람 위치 후보
2. SAHI 타일 추론 → 정밀 탐지
3. SAHI 결과 중 풀이미지 후보 중심 40px 반경 내만 채택

### 효과 (go2k_manual 604장 기준)

| 설정 | P | R | F1 | FP |
|---|---|---|---|---|
| SAHI only | 0.725 | 0.906 | 0.804 | 575 |
| **+gate (conf=0.20, r=40)** | **0.867** | **0.893** | **0.880** | **231** |

- F1 +7.6%p, FP 60% 감소
- pseudo-label 생성에만 적용 (실시간 추론은 SAHI only)

스크립트: `eval_fullimage_gate.py`, `eval_gate_finetune.py`
전략 문서: `GATE_STRATEGY.md`

---

## 8. Pseudo-Label 파이프라인 (captures 3K)

### 데이터 수집
- 소스: `/home/lay/video_indoor/static/captures/cam1,cam2/`
- 원본 프리즈 중복 제거: 43,056장 (38.8GB) 삭제 (`dedup_captures.py`)
- 선별: 전체 시간대, 10초 간격, go2k 타임스탬프 제외, cam1+cam2 interleave

### 탐지 결과
- 3,000장 탐지 (cam1: 1,219, cam2: 1,781)
- 게이트 전 6,233 bbox → 게이트 후 5,058 bbox (19% FP 필터)
- 스크립트: `detect_captures_gated.py`

### CVAT 패키징
- `/home/lay/hoban/datasets/cvat_all/` (3개 task, 3.07GB)
- Part 1: 1,202장 (go2k 604 + captures 598)
- Part 2: 1,202장 (captures)
- Part 3: 1,200장 (captures)
- 스크립트: `prepare_cvat_all.py`

---

## 9. 다음 단계

1. **CVAT 검수** — 3,604장 pseudo-label 수동 검토/수정
2. **go2k_v3 학습** — 타일 학습 + 1280px + copy_paste (`train_go2k_v3.py`)
3. **검수 완료 후 go2k_v4** — 검수된 라벨 포함하여 재학습
4. **video_indoor SAHI 실시간 적용** — 모델 품질 향상 후 게이트 없이 운용
