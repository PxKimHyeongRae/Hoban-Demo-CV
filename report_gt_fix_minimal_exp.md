# GT 수정 + 최소 데이터 실험 결과

> 날짜: 2026-02-23

---

## 1. GT 수정 (Val Set 검수)

### 작업 내용
- v19 모델의 SAHI 추론 결과와 기존 GT를 비교하여 오류 이미지 56장 식별
- CVAT에 업로드하여 수동 검수 완료
- 검수 결과를 val 라벨에 적용

### GT 변경 통계
| 항목 | 수량 |
|------|------|
| 검수 이미지 | 56장 / 605장 |
| bbox 추가 (미라벨링) | +67개 |
| 클래스 수정 (ON↔OFF) | 8개 파일 |
| bbox 삭제 | 0개 |

### v19 재평가 결과 (GT 수정 전후)

| 메트릭 | 수정 전 | 수정 후 | 변화 |
|--------|---------|---------|------|
| F1 (기존) | 0.922 | **0.938** | **+1.6%p** |
| F1 (ignore-aware) | 0.958 | **0.981** | **+2.3%p** |
| Precision | 0.968 | **0.986** | +1.8%p |
| Recall | 0.948 | **0.976** | +2.8%p |
| FP (ignore-aware) | 15 | **13** | -2 |
| FN (ignore-aware) | 22 | **22** | 동일 |

### Per-class (ignore-aware, 수정 후)
| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| helmet_on | 0.991 | 0.976 | **0.983** |
| helmet_off | 0.964 | 0.976 | **0.970** |

**결론: 모델 변경 없이 GT 수정만으로 F1 0.958 → 0.981 달성 (+2.3%p)**

---

## 2. 최소 데이터 실험

### 실험 설계
v23 현장 데이터(4,038장, 100% cam1+cam2)에서 단계별 서브셋 구성.
helmet_off(소수 클래스) 이미지를 우선 포함하여 판별력 극대화.

### 데이터셋 구성

| Tier | Train | ON bbox | OFF bbox | OFF 이미지 비율 | Neg |
|------|-------|---------|----------|----------------|-----|
| S | 500장 | 295 | 391 | 60% | 50 |
| **M** | **1,000장** | **684** | **661** | **50%** | **150** |
| L | 2,000장 | 2,534 | 1,245 | 47% | 306 |
| XL | 4,038장 | 4,710 | 1,245 | 23% | 1,064 |

공통 설정:
- Model: yolo26m.pt (COCO pretrained)
- Optimizer: SGD (lr0=0.005)
- imgsz: 1280, batch: 6
- Val: 517장 (수정된 GT)

### M 티어 결과 (1,000장)

| 메트릭 | M 티어 (1,000장) | v19 (10,852장) | 차이 |
|--------|-----------------|----------------|------|
| F1 (기존) | 0.905 | **0.938** | -3.3%p |
| F1 (ignore-aware) | 0.942 | **0.981** | -3.9%p |
| FP (ignore-aware) | 46 | 13 | +33 |
| FN (ignore-aware) | 61 | 22 | +39 |

### Per-class (ignore-aware)

| 클래스 | M 티어 | v19 | 차이 |
|--------|--------|-----|------|
| helmet_on F1 | 0.950 | 0.983 | -3.3%p |
| helmet_off F1 | 0.902 | 0.970 | **-6.8%p** |

### 분석
- 1,000장으로 F1=0.942 (ignore-aware)는 준수한 성능
- v19 대비 helmet_off에서 -6.8%p 차이 (Recall 0.872 vs 0.976)
- FP/FN 모두 v19 대비 3배 이상 → 데이터 양이 여전히 부족
- L 티어(2,000장) 실험으로 스케일링 효과 확인 필요

---

## 3. 다음 단계

- [ ] L 티어(2,000장) 학습 + 평가
- [ ] 결과에 따라 XL(4,038장) 또는 v19 데이터 기반 최종 결정
- [ ] 수정된 GT 기준 최종 모델 선정
- [ ] v19 → 배포 모델 교체 검토

---

## 4. 관련 파일

```
# 스크립트
train_minimal_data_exp.py         # 최소 데이터 실험 (S/M/L/XL 티어)
prepare_gt_fix_cvat.py            # GT 오류 검출 → CVAT 패키지
visualize_gt_errors.py            # GT 오류 시각화
apply_cvat_gt_fix.py              # CVAT 검수 결과 → val 라벨 적용

# 데이터셋
datasets_minimal_s/               # S 티어 (500장)
datasets_minimal_m/               # M 티어 (1,000장)
datasets_minimal_l/               # L 티어 (2,000장)
datasets_minimal_xl/              # XL 티어 (4,038장)

# GT 수정
cvat_gt_fix/cvat_gt_fix_manual/   # CVAT 검수 결과 (annotations.xml)
cvat_gt_fix/val_labels_backup/    # 원본 라벨 백업
cvat_gt_fix/visualized/           # 오류 시각화 이미지

# 모델
hoban_minimal_m/weights/best.pt   # M 티어 학습 결과

# 보고서
report_strategic_analysis.md      # 종합 전략 분석
report_gt_fix_minimal_exp.md      # 이 문서
```
