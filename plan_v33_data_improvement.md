# v33 데이터셋 고도화 계획

> 작성: 2026-02-27
> 목적: v32(SAHI F1=0.958) 기반으로 미탐/오탐 추가 감소

## 1. 현재 상태 (v32)

| 메트릭 | 값 | 비고 |
|--------|-----|------|
| SAHI F1 (helmet) | **0.958** | 역대 최고 |
| cam3 fallen FP @conf=0.60 | **8/171** (4.7%) | v29 대비 93.7% 감소 |
| cam3 fallen FP @conf=0.80 | **0/171** | FP 완전 제거, recall 미확인 |
| Fallen val recall @conf=0.25 | **57.3%** (86/150) | 낮음 - 개선 필요 |
| mAP50 | 0.853 | |

### v32 데이터셋 구성
| 카테고리 | 이미지 | bbox | 소스 |
|---------|--------|------|------|
| Helmet (class 0+1) | 2,000 | 3,779 | L-tier (datasets_minimal_l) |
| Fallen (class 2) | 2,000 | 2,509 | unified_safety_all (fastdup dedup) |
| Negative | 960 (+306 빈 라벨) | 0 | neg_1k_manual |
| **합계** | **4,960** | **6,288** | |

## 2. 근본 원인 분석 (5개 에이전트 종합)

### 2-1. cam3 잔여 FP 8건 패턴

| 클러스터 | 건수 | AR | area | cy | 특징 |
|----------|------|-----|------|-----|------|
| A: 소형 정방형 | 2 | 0.98 | 0.003 | 0.44 | 조도 변화/그림자 패턴 |
| B: 세로 좁은형 | 6 | 0.23 | 0.033 | 0.53 | **실제 fallen과 유사한 형태** |

핵심: 제거된 FP는 화면 상단(cy=0.28) 정방형. 잔여 FP는 바닥 근처(cy=0.51) 세로형 → 가장 혼동하기 쉬운 형태만 남음.

### 2-2. 학습-배포 분포 격차 (Critical)

| 이슈 | Train | Val/배포 | 격차 |
|------|-------|---------|------|
| **Fallen tiny (area<1%)** | **2.1% (52개)** | **25.3%** | **12배** |
| Fallen small (1~5%) | 30.9% (775개) | - | |
| Fallen cy 위치 | 0.644 (하단) | 0.559 (중앙) | -0.085 |
| 해상도 | 40%가 640x640 | 100% 1920x1080 | 도메인 갭 |

### 2-3. Tiny bbox 부족의 근본 한계

가용 풀 5,783장에서도 tiny(<1%) = **63장(1.7%)** 뿐.
→ 선별만으로는 해결 불가, **mosaic+scale로 small→tiny 자동 변환**이 핵심.

Mosaic 효과:
```
원본 5%  → mosaic 후 1.25%  (small → tiny 진입)
원본 2%  → mosaic 후 0.50%  (ultra-tiny)
원본 10% → mosaic 후 2.50%  (small)
+ scale=0.9 적용 시 추가 축소
```

**결론: small(1~5%) bbox를 많이 넣는 것 = tiny 학습 강화**

## 3. v33 데이터셋 구성

### 변경 요약

| 카테고리 | v32 | v33 | 변경사항 |
|---------|-----|-----|---------|
| Helmet | 2,000 | **3,000** | v19에서 +1,000 추가 |
| Fallen | 2,000 | **3,000** | small/tiny 우선 +1,000 추가 |
| Negative | 960 | **960** | 동일 |
| **합계** | **4,960** | **6,960** | |

### 3-1. Helmet 3,000장

- 기존 L-tier 2,000장 유지
- v19 train에서 추가 1,000장 선별 (L-tier/val 중복 제외)
- 소스: `/home/lay/hoban/datasets_go3k_v19/train/` (9,172장 추가 가용)
- 선별 기준: bbox 포함 이미지, val 이미지 제외

### 3-2. Fallen 3,000장 (Option B: small/tiny 우선)

- 기존 v32 큐레이션 2,000장 유지
- 미사용 풀 3,783장에서 `min_area` 오름차순 상위 1,000장 추가
- 소스: `/home/lay/hoban/.omc/fastdup_v32_full/deduped_keep.json`

추가 1,000장 선별 전략:
```python
# 미사용 풀에서 bbox area가 작은 순서로 1,000장 선별
# → small(1~5%) bbox가 대량 추가되어 mosaic 시 tiny 학습 극대화
unused_sorted = sorted(unused_pool, key=lambda r: r['min_area'])
v33_add_1000 = unused_sorted[:1000]
```

예상 분포 변화:
| 구간 | v32 (2k) | v33 (3k) | 변화 |
|------|----------|----------|------|
| tiny (<1%) | 52 (2.1%) | ~235 (5.4%) | +3.3%p |
| small (1~5%) | 775 (30.9%) | ~1,738 (39.7%) | +8.8%p |
| medium (5~15%) | 1,381 (55.0%) | ~1,706 (39.0%) | -16%p |
| large (>15%) | 301 (12.0%) | ~695 (15.9%) | +3.9%p |

### 3-3. Negative 960장 (동일)

- 소스: `datasets/cvat/neg_1k_manual/` (cam1: 742, cam2: 218)
- 실제 현장 CCTV, CVAT 검증 완료

### 3-4. Val (동일)

- 617장 (helmet 517 + fallen 100)
- v32와 동일

## 4. 학습 설정 변경

### Augmentation 최적화

| 파라미터 | v32 | v33 | 이유 |
|----------|-----|-----|------|
| scale | 0.7 | **0.9** | tiny 객체 생성 극대화 |
| multi_scale | 0.0 | **0.25** | 960~1600px 랜덤 변동, 스케일 다양성 |
| close_mosaic | 0 | **15** | mosaic tiny 학습 후 마지막 15ep 안정화 |
| erasing | 0.15 | **0.3** | occlusion 시뮬레이션 |
| perspective | 0.0 | **0.0005** | CCTV 부감 시뮬레이션 |
| epochs | 150 | **150** | 동일 |
| patience | 35 | **35** | 동일 |
| batch | 4 | **4** | 동일 |
| optimizer | SGD | SGD | 동일 |
| lr0 | 0.005 | 0.005 | 동일 |

### v33 학습 설정 전문

```python
model.train(
    data=data_yaml,
    epochs=150, imgsz=1280, batch=4, device="0",
    optimizer="SGD", lr0=0.005, lrf=0.01,
    momentum=0.937, warmup_epochs=3.0, weight_decay=0.0005, cos_lr=True,
    # Augmentation (tiny 최적화)
    mosaic=1.0, mixup=0.1, copy_paste=0.4,
    scale=0.9,          # 변경 (0.7→0.9)
    multi_scale=0.25,   # 신규 추가
    close_mosaic=15,    # 변경 (0→15)
    erasing=0.3,        # 변경 (0.15→0.3)
    perspective=0.0005, # 신규 추가
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    translate=0.1, degrees=5.0, fliplr=0.5,
    patience=35, amp=True, workers=4, seed=42,
)
```

## 5. 실행 계획

### Step 1: v33 스크립트 작성
- `train_go3k_v33.py` 생성
- v32 기반 + helmet 3k + fallen 3k(small/tiny 우선) + negative 960

### Step 2: 데이터셋 빌드
```bash
conda run -n llm python train_go3k_v33.py --prepare
```

### Step 3: 학습
```bash
conda run -n llm python train_go3k_v33.py --train --batch 4
```

### Step 4: 평가
1. SAHI F1 (helmet): 0.958 유지 확인
2. cam3 FP @conf=0.60: 8건 → 감소 목표
3. Fallen val recall: 57.3% → 개선 목표
4. v32 vs v33 per-class 비교

## 6. 추후 검토 (v33 결과 확인 후)

| 순위 | 작업 | 목표 | 조건 |
|------|------|------|------|
| 1 | SAHI 슬라이스 파인튜닝 | tiny→medium 변환 (+12~14%p AP) | 학습시간 3~9배 허용 시 |
| 2 | Small Object Copy-Paste 합성 | tiny fallen 직접 생성 | v33으로 부족 시 |
| 3 | Cleanlab 라벨 QA | 오염 라벨 제거 | 라벨 노이즈 의심 시 |
| 4 | 후처리 fallen 필터 | FP 즉시 제거 | FP 잔여 시 |
| 5 | 2-Stage Fallen Detection | FP 원천 차단 | 근본 해결 필요 시 |

## 7. 참조

### 분석 보고서
- cam3 FP 분석: `.omc/scientist/reports/20260227_062217_v32_fp_analysis.md`
- 분포 분석: `.omc/scientist/reports/20260227_062315_v32_distribution_analysis.md`
- fallen 전략 비교: `.omc/scientist/reports/20260227_065654_v32_fallen_analysis.md`

### 분석 차트
- `.omc/scientist/figures/v32_fp_analysis.png`
- `.omc/scientist/figures/v32_fallen_analysis.png`
- `.omc/scientist/figures/v33_strategy_comparison.png`

### 참고 자료
- SAHI 슬라이스 파인튜닝: arxiv.org/abs/2202.06934 (ICIP 2022, +12.7~14.5% AP)
- Simple Copy-Paste: CVPR 2021
- CACP (Context-Aware Copy-Paste): CVPR 2025 Workshop
- yolo-tiling: github.com/Jordan-Pierce/yolo-tiling
- Cleanlab: github.com/cleanlab/cleanlab
- SSDA-YOLO: arxiv.org/abs/2211.02213
