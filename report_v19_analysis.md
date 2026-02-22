# v19 종합 분석 보고서 (2026-02-22)

## 1. 모델 성능 비교 (605장 평가셋)

### 1.1 평가셋 정리
- 기존 729장 → **중복 124장 제거** → 605장 (517 val + 88 extra)
- cam1 정적 장면 동일 이미지(md5 hash) 제거
- v16/v19/v23 val 디렉토리의 broken symlink 248개씩 정리

### 1.2 기존 평가 (후처리 파이프라인 적용)

| Model | SAHI F1 | P | R | TP | FP | FN |
|-------|---------|------|------|------|------|------|
| **v19** | **0.922** | 0.892 | 0.953 | 1,092 | 132 | 54 |
| v21-l | 0.915 | 0.877 | 0.955 | 1,095 | 153 | 51 |
| v23 | 0.912 | 0.893 | 0.932 | 1,068 | 128 | 78 |
| v17 | 0.910 | 0.879 | 0.943 | 1,081 | 149 | 65 |

### 1.3 Ignore-aware 평가 (area < 0.00020 무시)

GT의 23% (264/1,146)가 ~20px 이하 극소 객체 → 사람도 판단 불가 → ignore 처리.

| Model | 기존 F1 | Ignore F1 | 변화 | Best Conf |
|-------|---------|-----------|------|-----------|
| **v19** | 0.922 | **0.958** | +3.6%p | c0=0.30, c1=0.25 |
| v21-l | 0.915 | 0.949 | +3.4%p | c0=0.35, c1=0.30 |
| v17 | 0.910 | 0.947 | +3.7%p | c0=0.40, c1=0.10 |
| v23 | 0.912 | 0.941 | +2.9%p | c0=0.50, c1=0.10 |

**v19 ignore-aware 상세**:
- helmet_on: P=0.940, R=0.974, F1=0.957
- helmet_off: P=0.945, R=0.981, F1=0.963

---

## 2. FP/FN 오류 분석

### 2.1 전체 FP 분석 (기존 평가, 132개)

| 패턴 | 수량 | 비중 | 위험도 |
|------|------|------|--------|
| BG→ON (배경→헬멧착용) | 103 | 78% | 무해 |
| BG→OFF (배경→미착용) | 17 | 13% | false alarm |
| ON→OFF (착용→미착용 혼동) | 7 | 5% | false alarm |
| OFF→ON (미착용→착용 혼동) | 5 | 4% | 놓침 |

- FP 94%가 tiny (<0.1% area, ~20px)
- FP의 40-50%는 GT 미라벨 (실제로는 정확한 탐지)
- FP 52%가 conf >= 0.7 → 단순 threshold로 해결 불가

### 2.2 Ignore-aware 후 남은 위험 오류 (15개)

| 유형 | 수량 | 위험도 | 패턴 |
|------|------|--------|------|
| ON→OFF 혼동 | 5 | false alarm | 어두운/역광 장면에서 헬멧이 모자처럼 보임 |
| BG→OFF | 4 | false alarm | 2-3개 GT 미라벨 가능 |
| OFF→ON 혼동 | 3 | 놓침 | 밝은 머리카락이 헬멧으로 오인 |
| FN OFF | 3 | 놓침 | 위 3건과 동일 이미지 (3장에서 6개 오류) |

**핵심 발견**: missed danger 6개가 **3장의 이미지**에 집중. 모두 밝은 머리카락/두건이 헬멧처럼 보이는 패턴.

---

## 3. mAP50-95 천장 (0.72) 원인

| 원인 | 설명 |
|------|------|
| Val GT 87-96% tiny | 19×35px에서 2px 오차 → IoU 0.73으로 급락 |
| Train-Val 크기 불일치 | Train median 0.42% vs Val median 0.04% (10배 차이) |
| YOLO P3 구조 한계 | stride=8에서 19px 객체 = 2.4 feature cells |
| GT annotation 불확실성 | ±2-3px 오차가 이론적 mAP50-95 상한 결정 |

**결론**: 현재 mAP50-95=0.72는 tiny object 물리적 한계에 근접. 의미 있는 개선은 P2 layer 추가 또는 GT re-annotation 필요.

---

## 4. Ignore Threshold 근거

Train/Val bbox area 분포 비교:

| Area 범위 | Train 비중 | Val 비중 | 설명 |
|-----------|-----------|---------|------|
| < 0.00015 (~18px) | 3.7% | 13.4% | |
| **< 0.00020 (~20px)** | **6.2%** | **23.0%** | **채택** |
| < 0.00025 (~23px) | 8.5% | 31.2% | |

- Train에서 6.2%만 해당 → 모델이 거의 학습하지 못한 영역
- Val에서 23.0% 해당 → FP/FN을 불공정하게 부풀림
- COCO 스타일 "ignore region" 적용

---

## 5. 개선 방안 (우선순위)

### 즉시 가능
1. **v19 배포**: v17(F1=0.910) → v19(F1=0.922) 교체, temporal smoothing이 false alarm 대부분 처리
2. **GT 보완**: BG→OFF 중 GT 미라벨 2-3건 추가

### 학습 데이터 보강 (다음 버전)
3. **밝은 머리 helmet_off 추가**: OFF→ON 3건 + FN OFF 3건 직접 해결
4. **어두운 장면 helmet_on 추가**: ON→OFF 혼동 감소
5. **오류 이미지 동일 장면 프레임 추가**: 3장의 문제 장면에서 추가 프레임 수집

### 장기 (ROI 낮음)
6. 2-stage 검증 모델 (FP 판별 분류기)
7. P2 layer 추가 (mAP50-95 개선용)

---

## 6. 핵심 결론

1. **v19가 최고 성능**: F1=0.922 (기존) / **0.958 (ignore-aware)**
2. **F1 천장의 주 원인은 GT 품질**: ~20px 이하 객체가 FP/FN 78개를 부풀림
3. **실제 위험 오류 15개**: 605장 중 2.5% 미만
4. **missed danger 6개는 3장에 집중**: 밝은 머리 helmet_off 데이터 보강으로 해결 가능
5. **모델 자체는 충분**: v21-l(26M)/v23(현장only) 모두 v19 미달, 데이터 품질이 핵심

---

## 7. 파일 구조

```
analysis/                          # 분석 결과 이미지 (gitignore)
  fp_crops_initial/                # 초기 19개 FP 크롭
  fp_crops_all/                    # 전체 132개 FP 크롭
  fp_crops_critical/               # 위험 FP 크롭 (카테고리별)
  remaining_errors/                # ignore-aware 후 남은 오류 크롭
    fp_on2off/ fp_off2on/          # 클래스 혼동
    fp_bg_off/ fp_bg_on/           # 배경 오탐
    fn_on/ fn_off/                 # 미탐지
  v23_vis/                         # v23 GT+Pred bbox 시각화 (605장)

# 스크립트
eval_ignore_aware.py               # Ignore-aware SAHI F1 평가 (--compare)
analysis_remaining_errors.py       # 남은 오류 크롭 분석
visualize_v23_results.py           # GT+Pred bbox 시각화

# 분석 문서
report_v19_analysis.md             # 이 문서
plan_f1_breakthrough.md            # F1 돌파 전략
plan_map50_95_breakthrough.md      # mAP50-95 분석
plan_next_steps.md                 # 실험 요약 + 다음 단계
```
