# go2k 모델 개선 전략 (2026-02-15)

## 현재 상태

| 지표 | 값 | 비고 |
|------|-----|------|
| Validation mAP50 | 0.927 | go2k_v2, 640px 학습 |
| Validation mAP50-95 | 0.674 | bbox 정밀도 낮음 |
| SAHI 실전 F1 | 0.804 | go2k_manual 604장 GT 대비 |
| SAHI Precision | 0.725 | FP 578개 / 총 예측 2,095개 |
| SAHI Recall | 0.903 | FN 163개 / GT 1,680개 |
| 학습 해상도 | 640px | 추론은 SAHI 640x640 타일링 |
| CCTV 도메인 데이터 | 479장 (go2k train) | 8배 오버샘플링으로 보정 |

## 핵심 갭 분석: mAP50=0.927 vs SAHI F1=0.804

12.3%p 갭의 원인:

1. **학습-추론 형태 불일치**: 640px 전체 이미지로 학습 → SAHI 640x640 타일로 추론. 모델이 타일 경계 패턴을 학습하지 못함
2. **도메인 불일치**: v13 범용 데이터 8K가 go2k CCTV 도메인에서 FP 유발
3. **해상도 불일치**: 640px 학습에서 소형 객체가 수 픽셀에 불과

## 개선 전략 우선순위

### [1순위] 타일 학습 데이터 구축 (Sliced Training)

- **방법**: go2k_manual 672장을 SAHI와 동일하게 640x640 타일로 분할하여 학습 데이터 구축
- **효과**: 1장 → ~12타일 = ~8,000장. 학습-추론 입력 형태 일치
- **근거**: SAHI 원논문에서 "sliced fine-tuning + sliced inference 결합 시 모든 모델에서 상당한 개선" 보고
- **난이도**: 낮음 (yolo-tiling 라이브러리 또는 SAHI slice_image 함수 활용)
- **기대**: SAHI F1 0.804 → 0.83+ (타일 경계 FP 감소)

### [2순위] 학습 해상도 1280px 전환

- **방법**: `imgsz=1280, batch=4`로 학습 (RTX 4080 16GB에서 v13 stage2로 검증 완료)
- **효과**: 소형 객체 해상도 2배 향상
- **근거**: v13 stage2에서 이미 mAP50-95=0.727 달성 (현재 0.674 대비 +5.3%p)
- **트레이드오프**: 에폭당 학습 시간 ~4배 증가, batch=4
- **기대**: mAP50-95 0.674 → 0.70+

### [3순위] Pseudo-label 파이프라인 고도화 (현재 진행중)

현재 진행 중인 3,072장 pseudo-labeling + CVAT 검수는 올바른 방향. 보완 사항:

- **Hard Example Mining**: 균등 샘플링 대신, 저신뢰 탐지(conf 0.15~0.40) 이미지를 우선 라벨링
- **클래스별 임계값**: person_with_helmet conf=0.55, person_without_helmet conf=0.40 차등 적용
- **Iterative Refinement**: 1차 검수 → 재학습 → 개선 모델로 2차 pseudo-label → 더 적은 검수 필요

### [4순위] Copy-Paste 증강 활성화

- **방법**: `copy_paste=0.15` (현재 0.0으로 비활성)
- **효과**: 소형 객체 증강 + person_without_helmet 클래스 불균형(231 vs 1,449) 완화
- **난이도**: 코드 한 줄 변경

### [5순위] P2 소형객체 탐지 헤드 추가

- **방법**: YOLOv8 P3/P4/P5 → P2/P3/P4/P5 4-head로 확장
- **효과**: 160x160 feature map에서 아주 작은 객체 탐지 가능
- **근거**: VisDrone에서 P2 추가로 mAP50-95 +36.1% 보고 (SOD-YOLO)
- **주의**: SAHI와 효과 중복 가능 (SAHI가 이미 소형 객체를 확대)

### [6순위] TTA (Test-Time Augmentation)

- **방법**: `model.predict(augment=True)` - 수평반전 + 3스케일 앙상블
- **효과**: 오프라인 배치 처리에서만 실용적 (추론시간 2~3배 증가)
- **적용**: pseudo-label 생성 시 사용하여 라벨 품질 향상

### [7순위] 모델 아키텍처 변경

- YOLOv8m → YOLOv8l: batch=2 이하로 학습 불안정
- YOLOv11/YOLO26: 아직 초기 단계, 아키텍처 변경만으로 극적 개선 어려움
- **결론**: 현재 YOLOv8m에서 데이터/학습 전략 최적화가 우선

## 실행 계획

```
Phase 1 (완료)
  ├── [완료] go2k_manual 604장 → 640x640 타일 분할 학습 데이터 구축
  ├── [완료] imgsz=1280, batch=4, copy_paste=0.15 적용 (train_go2k_v3.py)
  ├── [완료] Full-Image Gate 개발 (FP 60% 감소, F1 0.804→0.880)
  ├── [완료] captures 프리즈 중복 제거 (43K장/38.8GB)
  ├── [완료] captures 3K gated pseudo-labeling (cam1:1219 + cam2:1781)
  ├── [완료] CVAT 패키징 (3,604장 = go2k 604 + captures 3K, 3개 task)
  └── [대기] go2k_v3 학습 시작

Phase 2 (진행 중)
  ├── [진행] CVAT pseudo-label 검수 (3,604장, 3개 task)
  ├── hard example mining으로 추가 라벨링 대상 선정
  └── 검수 완료분 추가하여 go2k_v4 학습

Phase 3 (Phase 1-2 결과 확인 후)
  ├── P2 detection head 실험
  ├── TTA pseudo-label 적용
  └── Teacher-Student iterative refinement
```

## 다음 학습 설정 (go2k_v3 제안)

```python
# 변경 전 (go2k_v2)
imgsz=640, batch=16, copy_paste=0.0

# 변경 후 (go2k_v3)
imgsz=1280          # 소형 객체 해상도 2배
batch=4             # RTX 4080 16GB 검증 완료
copy_paste=0.15     # 소형 객체 증강
multi_scale=0.3     # 960~1600 범위 다중 스케일
close_mosaic=10     # 마지막 10에폭 mosaic off
```

학습 데이터:
```
datasets_go2k_v3/
  train/
    images/
      - v13 서브샘플 8,000장
      - go2k_manual 타일 ~5,700장 (479장 x 12타일)
      - [Phase 2] pseudo-label 검수 완료분
  valid/
    images/
      - v13 서브샘플 1,000장
      - go2k_manual 타일 ~1,440장 (120장 x 12타일)
```

## 성공 기준

| 개선 | 목표 | 측정 |
|------|------|------|
| 타일 학습 | SAHI F1 >= 0.83 | eval_go2k_v2.py 동일 설정 |
| 1280px 학습 | mAP50-95 >= 0.70 | validation 기준 |
| Copy-paste | person_without_helmet Recall >= 0.85 | 클래스별 평가 |
| Pseudo-label 확장 | SAHI Precision >= 0.78 | go2k_manual GT 대비 |
| **최종 목표** | **SAHI F1 >= 0.85, P >= 0.80, R >= 0.90** | **통합 평가** |
