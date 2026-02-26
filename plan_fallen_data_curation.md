# Fallen 데이터 큐레이션 & 근본 개선 계획

> 작성: 2026-02-26
> 목적: cam3 실내 Fallen FP 근본 원인 제거

## 1. 문제 요약

v29 모델이 cam3 실내 CCTV에서 **천장 형광등, 비닐봉지 등 정적 물체**를 fallen(conf=0.75)으로 감지.
하루 127건 FP 발생. v17(2-class)은 동일 장면에서 0건.

## 2. 근본 원인 (4개 에이전트 분석 종합)

### 원인 1: fallen.v2i 데이터 오염 (Primary)
- fallen.v2i가 전체 fallen 라벨의 33.3% (971/2,915) 차지
- **29.6%가 이미지 상단 (cy < 0.3)** — 다른 소스는 1.3~3.2%
- 오버헤드 CCTV 앵글 → 사람이 작은 비정형 blob으로 보임
- 이 패턴이 실내 비닐봉지/형광등과 시각적으로 일치

### 원인 2: v29에서 negative 이미지 제거
- v24: 2,500장 negative 포함 → v29: 0장
- 모델이 "배경"을 학습할 기회 상실

### 원인 3: 학습-배포 도메인 갭
- 학습: Roboflow 640x640 (건설현장/야외)
- 배포: 1920x1080 실내 CCTV
- 실내 장면 학습 데이터 0건

### 원인 4: 클래스 불균형
- v29 = helmet 2,000 + fallen 2,000 (50:50)
- fallen 비율이 과도 → 모델이 fallen에 과민반응

## 3. 데이터 품질 감사 결과

### 소스별 품질
| 소스 | 라벨 수 | 제거 | 제거율 | 권장 |
|------|---------|------|--------|------|
| **fallen.v2i** | 971 | 361 | **37.2%** | Heavy Filter |
| fall.v1i | 720 | 92 | 12.8% | Filter (대형만) |
| Fall.v1i | 605 | 73 | 12.1% | Filter (대형만) |
| **fall.v4i** | 535 | 17 | **3.2%** | Keep All |
| Fall.v3i | 44 | 5 | 11.4% | Filter |
| fall.v2i | 40 | 3 | 7.5% | Filter |
| **합계** | **2,915** | **551** | **18.9%** | |

### 문제 유형별 분류
| 문제 유형 | 수량 | 주요 소스 |
|-----------|------|-----------|
| v2i 상단 tiny blob (area<1%, cy<0.35) | 277 | fallen.v2i 100% |
| Tiny top blob (area<0.5%, cy<0.3) | 176 | fallen.v2i 98% |
| 과도하게 큰 라벨 (area>25%) | 152 | fall.v1i+Fall.v1i 93% |
| **cam3 FP 프로파일 일치** | **66** | **fallen.v2i 70%** |
| 천장 위치 (cy<0.15) | 62 | fallen.v2i 94% |
| 엣지 라벨 | 44 | fallen.v2i 96% |
| 극소 (area<0.1%) | 17 | fallen.v2i 94% |
| 극단 AR (>4.0 or <0.2) | 11 | Mixed |

### 큐레이션 결과
| 항목 | Before | After |
|------|--------|-------|
| Fallen 라벨 | 2,915 | **2,364** (-18.9%) |
| 파일 완전 삭제 | - | 240 |
| 파일 재작성 (일부 라벨 제거) | - | 119 |
| 파일 유지 | - | 1,641 |

## 4. 실행 계획

### Phase 1: 데이터 큐레이션 (즉시)

**Step 1-1: Fallen 2,000장 — unified_safety_all에서 고품질만 선별**

소스: `/data/unified_safety_all/train/` (class 4 = fallen → class 2로 리맵)

소스별 전략:
| 소스 | 품질 통과 | 통과율 | 전략 |
|------|----------|--------|------|
| **fall.v4i** | **3,665장** | **98.9%** | **Primary — 최우선 사용** |
| fall.v1i | 3,426장 | 67.5% | Secondary — 보충용 |
| Fall.v1i | 3,202장 | 64.8% | Secondary — 보충용 |
| Fall.v3i | 806장 | 78.3% | Optional |
| fall.v2i | 402장 | 41.2% | Optional |
| ~~fallen.v2i~~ | ~~12,485장~~ | ~~91.5%~~ | **완전 배제 — FP 원인** |

품질 필터 (모든 소스에 적용):
- area < 0.1% → 제거 (극소)
- cy < 0.15 → 제거 (천장 위치)
- area < 0.5% AND cy < 0.3 → 제거 (tiny top blob)
- area > 25% → 제거 (과대)
- AR > 4.0 or AR < 0.2 → 제거 (극단 종횡비)
- cx < 0.02 or cx > 0.98 → 제거 (엣지)
- Roboflow 중복 제거 (.rf. 해시 기반 dedup)

선별 순서: fall.v4i 우선 → fall.v1i/Fall.v1i 보충 → 2,000장 달성

**Step 1-2: Helmet 최고 품질 2,000장**
- 소스: L-tier (`datasets_minimal_l/train/images`)
- 기존 v29 동일 구성

**Step 1-3: Negative 2,000장**
- 소스: `/data/aihub_data/helmet_negative_10k_v3/` (9,991장, 검증 완료)
- 2,000장 랜덤 샘플링, 빈 라벨 파일 생성
- ※ 실내 negative 미사용 (테스트 전용 환경)

### Phase 2: v32 학습 (큐레이션 후)

**데이터셋 구성:**
| 카테고리 | 수량 | 비율 |
|---------|------|------|
| Helmet (L-tier) | 2,000 | 33.3% |
| Fallen (큐레이션+보충) | 2,000 | 33.3% |
| Negative (건설현장) | 2,000 | 33.3% |
| **합계** | **6,000** | 100% |

**학습 설정 (v29 기반):**
```python
epochs=150, patience=35
optimizer="SGD", lr0=0.005, lrf=0.01
imgsz=1280, batch=4
close_mosaic=0
copy_paste=0.4, scale=0.7, mixup=0.1
```

**기대 효과:**
- cam3 실내 FP: 127건/일 → 대폭 감소 (오염 라벨 제거 + negative 학습)
- Fallen AP50: 0.841 유지 또는 소폭 변동 (고품질 라벨만 유지)
- Helmet SAHI F1: 0.955 유지 (helmet 데이터 동일)

### Phase 3: 검증

1. **v32 SAHI 평가**: 기존 val set (617장)
2. **cam3 FP 테스트**: cam3 빈 프레임 50장에 추론 → fallen 0건 확인
3. **v29 vs v32 비교**: per-class AP50, SAHI F1

### Phase 4 (추후): 2-Stage Fallen Detection

현재 아키텍처의 한계:
- single-stage YOLO는 "사람 없는 곳에 fallen이 있을 수 없다"는 상식을 모름
- bbox 형태만으로 fallen을 판단 → 비슷한 형태의 물체에 취약

**장기 해결: Person-first 2-stage 접근**
```
Stage 1: Person Detection (기존 cls 0,1 활용)
  → person bbox가 없으면 fallen 불가

Stage 2: Pose Classification
  → 감지된 person bbox 내에서 standing/fallen 판별
  → 옵션 A: YOLO의 keypoint detection (pose estimation)
  → 옵션 B: person crop → 별도 classifier (standing vs lying)
```

**장점:**
- 비닐봉지, 형광등 등 비인체 물체 FP 원천 차단
- 카메라/환경 변경 시 재학습 불필요
- fallen 정의가 "사람의 자세"로 명확해짐

**구현 비용:**
- YOLO-pose 활용 시: 추가 모델 없이 keypoint branch 추가
- 별도 classifier 시: lightweight ResNet/MobileNet fine-tune
- 추정 구현 기간: 1-2주

## 5. 우선순위 요약

| 순서 | 작업 | 기대 효과 | 소요 |
|------|------|----------|------|
| **1** | fallen.v2i 오염 라벨 제거 | FP 유발 패턴 제거 | 스크립트 자동화 |
| **2** | Negative 2,500장 복원 | 배경 인식 학습 | 즉시 |
| **3** | 실내 negative 200장 추가 | 실내 환경 인식 | 반자동 |
| **4** | v32 학습 + 검증 | 근본 해결 확인 | ~8시간 |
| **5** | (추후) 2-Stage 전환 | 영구 해결 | 1-2주 |

## 6. 참조 파일

- 문제 라벨 목록: `.omc/scientist/fallen_problematic_labels.json`
- 품질 감사 보고서: `.omc/scientist/reports/20260226_fallen_v29_quality_audit.md`
- FP 조사 보고서: `.omc/scientist/reports/20260226_101845_fallen_fp_investigation.md`
- 시각화:
  - `.omc/scientist/figures/fallen_v29_quality_overview.png`
  - `.omc/scientist/figures/fallen_v29_area_vs_cy_by_source.png`
  - `.omc/scientist/figures/root_cause_summary.png`
