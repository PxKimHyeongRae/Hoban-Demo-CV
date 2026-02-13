# Hoban v15 학습 전략 계획

## 현재까지의 실험 요약

| 실험 | Val mAP50 | CCTV 검출률 (1815장) | 속도 | 결론 |
|------|-----------|---------------------|------|------|
| v13 stage2 (AIHub, 1280px) | 0.945 | 6.7% (121장) | ~25ms | 도메인 갭 심각 |
| v13 stage2 + SAHI | - | 36.1% (656장) | ~330ms | 효과적이나 느림 |
| v13 stage2 + imgsz=1920 | - | 15.9% (288장) | ~80ms | 부분 개선 |
| v13 stage2 + imgsz=2560 | - | 13.3% (242장) | ~130ms | 오히려 하락 |
| v13 crop (3k, 640px) | 0.871 | 0% (0장) | - | 완전 실패 |
| v14 crop (60k, 640px) | 학습중 | TBD | - | v13 crop과 동일 접근 |
| Manual v1 (53장, head-only) | 0.641 | - | - | 데이터 부족 + 라벨 부적절 |

## 핵심 인사이트

1. **SAHI 36.1% 달성**: 모델의 학습된 feature는 CCTV에도 유효. 문제는 순수히 스케일/해상도
2. **Crop 학습 실패 (0%)**: 학습-추론 입력 분포 불일치가 근본 원인
3. **Tiled Training이 해법**: 학습 시에도 타일링하면 SAHI 추론과 분포 일치
4. **Domain Adaptation 필요**: AIHub vs CCTV 간 시점/배경/스케일 차이

## 실행 계획

### Phase 1: Tiled Training (v15)

**원리**: SAHI는 추론 시에만 타일링하지만, Tiled Training은 학습 시에도 타일링하여 모델이 타일 단위 입력에 최적화됨.

**v14 crop과의 핵심 차이**:
- v14 crop: bbox 중심으로 crop → 항상 객체가 있는 영역만 학습 → full-frame 추론 시 0%
- Tiled training: 그리드 기반 균일 분할 → 배경 타일 포함, SAHI 추론과 동일한 분포

**데이터 구성**:
- Source: datasets_v13 (train 36K + valid 9K, 1920x1080)
- 타일: 640x640, overlap 20%
- 이미지당 ~6 타일 → train ~216K tiles, valid ~54K tiles
- negative 타일 포함 (실제 SAHI 추론과 동일)

**학습 설정**:
- Base: yolo26m.pt (COCO pretrained)
- imgsz: 640
- Optimizer: AdamW, lr0=0.001
- P2 detection head 추가 고려 (소형 객체 +5-15%)

**추론 방식**:
- CCTV 이미지도 동일하게 640x640 타일로 분할 후 각 타일 추론
- 2x3=6 tiles @ ~15ms each = ~90ms/img (SAHI 330ms 대비 3.7x 빠름)

### Phase 2: SAHI 기반 Semi-supervised Domain Adaptation

**단계**:
1. v13 stage2 + SAHI로 CCTV 1815장 pseudo-label 생성 (conf >= 0.15)
2. 시각화된 결과와 YOLO 라벨 함께 저장
3. 수동 검수/보정 (CVAT 또는 직접 수정)
4. Phase 1 모델에 CCTV 라벨 데이터 혼합 fine-tuning

**기대 효과**: CCTV 도메인 특화 → 검출률 50-70%+

### Phase 3: 추론 최적화 (선택)

- TensorRT/ONNX 변환으로 타일당 추론 속도 15ms → 5ms
- 4-6 tiles × 5ms = 20-30ms/img → real-time 가능
- Motion-based selective tiling으로 추가 속도 개선

## 성공 기준

| 지표 | 현재 | 목표 |
|------|------|------|
| CCTV 검출률 (conf=0.25) | 6.7% | > 50% |
| 추론 속도 (1920x1080) | 25ms (미검출) / 330ms (SAHI) | < 100ms |
| Val mAP50 (타일 기준) | - | > 0.85 |

## 파일 구조

```
datasets_v15/           # Tiled training dataset
  train/images/         # 640x640 tiles
  train/labels/         # YOLO format labels
  valid/images/
  valid/labels/
  data.yaml

datasets_cctv_pseudo/   # Phase 2: CCTV pseudo-labels
  images/               # 원본 CCTV 이미지
  labels/               # SAHI pseudo-labels (YOLO format)
  visualize/            # 시각화 결과 (수동 검수용)

build_v15_tiled.py      # 타일 데이터셋 빌더
train_v15.py            # v15 학습 스크립트
generate_pseudo_labels.py  # SAHI pseudo-label 생성기
```
