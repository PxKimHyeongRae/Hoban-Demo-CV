# v16 학습 및 평가 보고서

## 작성일: 2026-02-17

---

## 1. 배경

### 데이터 오염 문제 해결
이전 모델(go2k_v2~v7)은 평가 데이터가 학습에 포함되는 leakage 문제가 있었음.
- go2k 604장 중 603장(99.8%)이 3k 데이터셋에 포함
- 모든 이전 모델의 eval 수치가 부풀려져 있었음

### v16 전략
- **Train**: 3k_finetune/train 2,564장 + v13 서브샘플 8,000장 = **10,564장**
- **Val**: 3k_finetune/val **641장** (leakage 0%)
- **라벨**: 3k 보수적 라벨 통일 (확실한 헬멧만 라벨링, 뭉개진 것은 제외)
- **augmentation**: YOLO 내장 (mosaic, mixup, copy_paste) - byte copy 제거

---

## 2. 학습 결과

| 항목 | 값 |
|------|-----|
| 모델 | YOLOv8m (v13_stage2 pretrained) |
| 해상도 | 640px |
| 배치 | 24 |
| 에폭 | 100 (best @ 86) |
| optimizer | SGD (lr0=0.005, cos_lr) |
| 학습 시간 | 4.0h |

### 학습 지표 (Best epoch 86)

| Metric | 값 |
|--------|-----|
| mAP50 | 0.852 |
| mAP50-95 | 0.561 |
| Precision | 0.833 |
| Recall | 0.753 |

---

## 3. SAHI 평가 결과

### 기본 설정 (1280x720 타일)

| 설정 | P | R | F1 |
|------|---|---|-----|
| Uniform conf=0.45 | 0.845 | 0.925 | 0.883 |
| Per-class c0=0.45, c1=0.40 | 0.845 | 0.928 | **0.884** |

### 클래스별 성능

| 클래스 | P | R | F1 |
|--------|---|---|-----|
| helmet_on (cls 0) | 0.847 | 0.944 | 0.893 |
| helmet_off (cls 1) | 0.806 | 0.734 | 0.768 |

---

## 4. SAHI 파라미터 파인튜닝

### Phase 1: 타일 크기

| 타일 크기 | F1 |
|-----------|-----|
| **1280x720** | **0.884** |
| 960x540 | 0.857 |
| 640x360 | 0.842 |
| 640x640 | 0.754 |
| 640x480 | 0.754 |

### Phase 2: Overlap

| Overlap | F1 |
|---------|-----|
| 0.05 ~ 0.35 | 0.884 (전부 동일) |

Overlap 값은 성능에 영향 없음.

### Phase 3: Postprocess

| 설정 | F1 |
|------|-----|
| **NMS/IOS@0.3** | **0.885** |
| NMS/IOS@0.4 | 0.884 |
| NMS/IOS@0.5 | 0.884 |
| NMS/IOU@0.4 | 0.884 |
| NMS/IOU@0.5 | 0.882 |
| NMM/IOS@0.4 | 0.882 |
| NMM/IOS@0.5 | 0.882 |

### Phase 4: perform_standard_pred

| 설정 | F1 |
|------|-----|
| **std_pred=True** | **0.885** |
| std_pred=False | 0.873 |

풀이미지 + 타일 병합이 타일만 사용하는 것보다 +0.012 향상.

---

## 5. 최적 설정 요약

```
모델: hoban_go3k_v16_640/weights/best.pt
SAHI 설정:
  slice: 1280x720
  overlap: 0.05 (최소값으로 충분)
  postprocess: NMS / IOS / threshold=0.3
  perform_standard_pred: True
  confidence: c0=0.45, c1=0.40

최종 F1 = 0.885 (Clean 641장, 1,072 bbox)
```

---

## 6. 이전 모델과의 비교

| 모델 | F1 | Eval Set | 비고 |
|------|-----|----------|------|
| **v16 단독** | **0.885** | **641장 (clean)** | **leakage 0%, 신뢰 가능** |
| v2+v3 WBF | 0.886 | 125장 (clean for v2/v3) | 2모델 앙상블 |
| v2 단독 | 0.862 | 125장 (clean for v2) | leakage 없음 |
| v3 단독 | 0.858 | 125장 (clean for v3) | leakage 없음 |
| v3+v5+v7 WBF | 0.904 | 125장 | **v5 leakage로 오염** |

v16은 단독 모델로 이전 2모델 앙상블(v2+v3) 수준에 도달.
평가 규모도 125장→641장으로 5배 확대되어 신뢰도가 높음.

---

## 7. 파일 구조

```
build_dataset_go3k_v16.py   # 데이터셋 빌드 스크립트
train_go3k_v16.py           # 학습 스크립트
eval_go3k_v16.py            # F1 평가 스크립트
eval_go3k_v16_sahi_sweep.py # SAHI 파라미터 sweep 스크립트
datasets_go3k_v16/          # 빌드된 데이터셋 (symlinks)
hoban_go3k_v16_640/         # 학습 결과
```
