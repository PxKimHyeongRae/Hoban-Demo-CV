# Hoban 프로젝트 개발 일지

## 프로젝트 개요

**목표**: 건설현장 안전 감지를 위한 YOLO 객체 탐지 모델 학습
**탐지 대상**: 헬멧 착용(0), 헬멧 미착용(1), 사람(2), 쓰러진 사람(3)
**모델**: yolo26m.pt (YOLO v26 Medium, 42MB)
**작업 기간**: 2026-02-06 ~ 2026-02-08 (3일)

### 환경 구성

| 환경 | 용도 | 스펙 |
|------|------|------|
| **로컬 (Windows)** | 서브셋 실험, 데이터 가공 | RTX 5070 Ti 16GB, conda `gostock`, Python 3.11 |
| **서버 (Linux)** | 풀 스케일 학습 | `lay@pluxity:~/hoban`, conda `llm` |
| **WSL (Ubuntu)** | fastdup 중복 분석 | `~/fastdup_env/bin/python`, Python 3.10 |

**환경별 주의사항**:
- Windows: multiprocessing 사용 시 `if __name__ == '__main__': freeze_support(); main()` 필수
- 서버: `torch.cuda.set_per_process_memory_fraction` 사용 금지 (CUDA forward compatibility 에러)
- fastdup: Linux 전용 → WSL에서만 실행 가능

---

## Day 1: 2026-02-06 — 원본 데이터 확보

### 20:25~03:16 — AIHub 데이터 다운로드

AIHub에서 "공사현장 안전장비 인식 데이터" (Dataset #163) 다운로드.

- 경로: `/data/aihub_data/105.공사현장_안전장비_인식_데이터/`
- 구성: 원천 이미지 zip 45개 + 라벨 zip 19개 (Training 10 + Validation 9)
- 공항시설 서측 데이터는 26개 `.part` 분할 파일로 제공됨
- 라벨 형식: **JSON** (YOLO txt가 아님 — 나중에 변환 필요)

> 이 시점에서는 아직 데이터 구조를 파악하지 못한 상태. 45개 클래스가 뒤섞여 있었고, 헬멧 관련 클래스 코드도 모르는 상태였다.

---

## Day 2: 2026-02-07 — 데이터 파이프라인 구축 (11시간 연속 작업)

### Phase 1: Person 데이터 문제 인식과 교체 (01:00~04:00)

#### 01:00~01:50 — 기존 데이터 분석

**문제 발견**: 기존 datasets에서 학습한 모델의 person/fallen 클래스 성능이 저조했다. mAP 0.9 목표 대비 크게 부족.

```
실행한 분석:
1. analyze_data.py → 기존 데이터의 클래스별 분포 확인
2. select_v5.py → 최소한의 고품질 데이터로 서브셋 실험 (v5)
3. clean_data.py / clean_and_select.py → fastdup 기반 정제 시도
```

**결과**: v5 서브셋에서도 person/fallen이 안 나왔고, 단순 정제로는 해결 불가 → **근본적으로 person 소스 데이터가 문제**라고 판단.

#### 01:50~02:10 — WiderPerson으로 교체 (v6)

**결정**: 기존 aihub person 데이터를 전량 제거하고, 보행자 탐지 전용 데이터셋인 **WiderPerson**으로 교체.

- WiderPerson: http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/
- bbox가 전신을 정확히 잡고 있어 YOLO 학습에 적합

```
스크립트: build_dataset_v6.py
구성: aihub 헬멧 + WiderPerson person + robo fallen
```

#### 02:10~03:40 — 로컬/서버 분리 전략 + COCO person 병렬 준비

**결정**: 로컬에서는 서브셋으로 빠른 실험, 서버에서는 풀 데이터 학습. 이 분리로 반복 실험 속도가 크게 향상.

```
v6 서브셋 실험 (로컬):
  스크립트: select_v6_subset.py + train_v6_sub.py
  구성: 클래스당 1,000장 (4,000 train / 1,000 valid)

병렬 준비:
  스크립트: download_coco_person.py
  COCO 2017 train에서 person 3만장 다운로드 → 비교 실험용
```

#### 03:37~04:00 — WiderPerson vs COCO 비교 실험 → COCO 채택 (v7)

v6(WiderPerson) 학습 결과를 확인한 뒤, 동일 조건에서 COCO person으로 교체한 v7을 실험.

```
v7 서브셋 실험:
  스크립트: build_dataset_v7.py + train_v7_sub.py
  구성: 클래스당 1,000장, 30에폭
```

**비교 결과 (서브셋 30에폭)**:

| 지표 | v6 (WiderPerson) | v7 (COCO) | 판정 |
|------|------------------|-----------|------|
| mAP50 | 0.782 | 0.779 | 비슷 |
| mAP50-95 | 0.462 | **0.476** | COCO 승 |
| Background FP율 | 58% | **47%** | COCO 승 |
| Person 정탐율 | 65% | **67%** | COCO 승 |

**결정: COCO person 채택**
**이유**: COCO가 FP(오탐)율이 11%p 낮았다. WiderPerson은 보행자 특화지만 배경 다양성이 부족해 background FP가 많이 발생. COCO는 일상 장면이 다양해 "사람이 아닌 것"도 잘 구분.

---

### Phase 2: 클래스 불균형 해결과 fastdup 환경 구축 (04:00~06:00)

#### 04:00~05:00 — bbox 기반 균형 샘플링 도입

**문제**: v7 풀 빌드에서 클래스당 10,000장 목표 시 **helmet_x(미착용)가 4,841 bbox밖에 없었다**.

**해결: 이미지 수가 아닌 bbox 수 기준 균형**

```
스크립트: build_dataset_v7_full.py
핵심 함수: select_by_bbox()
  → 셔플 후 목표 bbox 수에 도달할 때까지 이미지를 선택하는 그리디 방식
  → 이미지 1장에 bbox 여러 개 가능하므로, 이미지 수 기준은 부정확
```

```
결과 (v7 full):
  helmet_o: 10,000 bbox
  helmet_x: 4,841 bbox (전량 — 데이터 한계)
  person: 10,000 bbox
  fallen: 7,014 bbox
  Train: 12,829장 / Valid: 2,860장
```

#### 05:12~06:00 — fastdup 환경 구축 (WSL)

**문제**: fastdup이 Windows를 지원하지 않음.

**해결 과정**:
1. conda env 생성 시도 → 실패 (fastdup은 pip only)
2. WSL Ubuntu로 전환
3. Python 3.10 전용 venv 생성: `~/fastdup_env/bin/python`
4. `pip install fastdup==2.50` 성공

```bash
# WSL에서 실행하는 방법:
wsl -d Ubuntu -- bash -c "~/fastdup_env/bin/python /mnt/d/task/hoban/스크립트.py"
# WSL sudo 비밀번호: q
```

v7 전체 데이터에 대해 fastdup 실행 → 유사도 > 0.9 기준 중복 제거 완료.

---

### Phase 3: Fallen 데이터 다중 소스 통합 (06:00~08:00)

#### 06:00~07:00 — 데이터 전략 재수립

**문제**: v7까지 fallen(쓰러짐)은 roboflow **단일 소스**에서만 가져왔다. 단일 소스의 편향이 모델 일반화에 악영향을 줄 수 있었다.

**탐색**: `D:\task\dataset`에 8개 fall 관련 폴더 발견.

| # | 소스 | 이미지 수 | fallen bbox | 특이사항 |
|---|------|-----------|-------------|----------|
| 1 | fall detection ip camera.v3i | 6,745 | 4,285 | IP 카메라 영상 |
| 2 | Fall Detection.v4-aug3x | 9,438 | 9,441 | 3배 증강 데이터 |
| 3 | Fall.v1i | 5,445 | 6,212 | down 클래스만 사용 |
| 4 | fall.v1i (2) | 5,072 | 6,364 | |
| 5 | fall.v2i | 1,659 | 1,016 | 소규모 |
| 6 | Fall.v3i | 5,313 | 5,460 | |
| 7 | fall.v4i | 7,775 | 2,309 | |
| 8 | fallen.v2i | 14,607 | 30,971 | 최대 소스 |

**결정: 8개 소스 전량 수집 → fastdup 정제 (v8 전략)**

- 개별 소스를 눈으로 검증하는 건 비현실적
- 전량 수집 후 fastdup으로 기계적 정제가 효율적이고 신뢰성 있음
- fire_smoke 폴더, datasets_clean은 제외 (사용자 지시)

**동시 작업: helmet_30k JSON→YOLO 변환**

```
문제: aihub helmet_30k의 라벨이 JSON 형식 (YOLO txt 아님)
  - class "07" = 안전모 착용 → cls 0
  - class "08" = 안전모 미착용 → cls 1
  - box: [x1, y1, x2, y2] 픽셀좌표 → [cx, cy, w, h] 정규화좌표

스크립트: convert_helmet_30k.py
결과 → helmet_pool/ (29,975 imgs, cls0: 29,591 / cls1: 36,192 bbox)
```

> helmet_x(미착용) bbox가 36,192개로 확인됨 — v7의 4,841개 대비 **7.5배 증가**. datasets_merged가 아닌 helmet_30k 원본에서 직접 변환해야 전량 확보 가능.

#### 07:00~08:00 — v8 파이프라인 실행

```
실행 순서:
1. convert_helmet_30k.py  → helmet_pool/ (29,975 imgs)
2. collect_fallen_all.py  → fallen_pool 수집 (52,945 imgs, 77,250 bbox)
3. analyze_fallen_pool.py → WSL fastdup 분석
4. dedup_fallen_pool.py   → 중복/이상치 제거
5. build_dataset_v8.py    → 최종 빌드
6. train_v8_server.py     → 서버 학습 스크립트 작성
```

**fastdup 분석 결과 (fallen_pool)**:

| 항목 | 수치 |
|------|------|
| 분석 대상 | 52,945장 |
| 유사도 > 0.9 중복쌍 | **82,714쌍** |
| 이상치 | 4,326개 |
| 깨진 이미지 | 0개 |
| 제거 대상 | 29,284장 (**55%**) |
| 정제 후 | 23,661장 / 33,018 bbox |

> **교훈**: roboflow 출처 데이터는 소스 간 교차 중복이 매우 심함. 8개 소스의 55%가 중복이었다. 다중 소스 통합 시 fastdup 정제는 필수.

**datasets_v8 최종 구성**:

| 클래스 | bbox (train) | bbox (valid) | 소스 |
|--------|-------------|-------------|------|
| 0 (helmet_o) | 10,001 | 2,000 | helmet_pool |
| 1 (helmet_x) | 10,001 | 2,000 | helmet_pool |
| 2 (person) | 10,001 | 2,000 | coco_person |
| 3 (fallen) | 10,001 | 2,002 | fallen_pool (8소스 정제) |
| **합계** | **40,004** | **8,002** | **17,855 train / 3,562 valid 이미지** |

---

### Phase 4: "데이터 품질 > 데이터 양" — 핵심 전환점 (08:00~10:00)

#### 08:00~09:00 — v8 서브셋 실험과 문제 발견

```
실험 조건:
  스크립트: build_v8_sub.py + train_v8_sub.py
  서브셋: 3K bbox/cls, 30에폭, 로컬

결과: mAP50=0.801, mAP50-95=0.491
```

Confusion matrix에서 **3대 문제** 발견:

| 문제 | 수치 | 심각도 |
|------|------|--------|
| Person miss rate | **38%** (230/602 놓침) | 높음 |
| Background FP | **605건** | 높음 |
| Helmet o↔x 혼동 | **109건** | 중간 |

#### 09:00~09:50 — 근본 원인 분석

처음에는 "데이터 양을 늘리면 해결될 것"이라고 접근했으나, 사용자 피드백:

> *"데이터 양을 무조건 올린다고 답은 아니잖아"*

이 피드백이 **이번 프로젝트의 가장 중요한 전환점**이었다. 이후 각 문제의 근본 원인을 파고들었다.

**원인 1: Person miss 38% → COCO bbox의 48%가 탐지 불가 크기**

```
분석 스크립트: check_coco_quality.py

COCO person 195,377 bbox 크기 분포:
  area < 1% (탐지불가): 93,759개 (48.0%) ← 절반이 64px 이하
  area 1~2% (어려움):   25,165개 (12.9%)
  area 2~10% (가능):    48,369개 (24.8%)
  area > 10% (양호):    28,084개 (14.4%)
```

640px 이미지에서 area 1% = 약 64px. 이 크기는 YOLO가 탐지하기 거의 불가능한 수준이다. 모델이 "작은 사람도 학습"하려다 보니 정상 크기 사람도 놓치는 부작용이 발생.

**원인 2: Background FP 605건 → 네거티브 샘플 부재**

학습 데이터의 **모든 이미지에 객체가 존재**했다. 모델이 "아무것도 없는 이미지"를 학습한 적이 없어 배경에서도 무언가를 찾으려 함.

**원인 3: Helmet o↔x 혼동 109건 → HEAD-ONLY bbox가 너무 작음**

```
분석 스크립트: check_helmet_quality.py + visualize_bbox.py

helmet_o: 69.2%가 area < 0.5% (640px에서 ~45px)
helmet_x: 81.9%가 area < 0.5% (640px에서 ~30px)
```

bbox가 "사람 전체"가 아니라 "머리/헬멧 부분만" 잡고 있었다. aihub 데이터가 CCTV 원거리 촬영이라 머리가 매우 작게 찍힘. **30px짜리 bbox에서 헬멧 유무 구분은 사람 눈으로도 어려움.**

> **핵심 교훈**: 데이터 품질 > 데이터 양. 탐지 불가능한 작은 bbox를 아무리 많이 넣어봐야 모델을 혼란시킬 뿐이다.

---

### Phase 5: 품질 필터링과 v9 — 효과 입증 (09:50~12:30)

#### 09:50~10:30 — area 기반 필터링 전략 수립

각 데이터 소스별 "탐지 가능한 최소 크기" 기준 설정:

| 데이터 | 필터 기준 | 근거 |
|--------|-----------|------|
| COCO person | area >= **2%** | 640px에서 ~90px, 안정적 탐지 가능 |
| Fallen pool | **0.5%** <= area <= **70%** | 극단적 크기만 제거 |
| Helmet (사용자 제공) | area >= **2%** (착용) / **1%** (미착용) | 미착용은 데이터 부족해 기준 완화 |
| Negative samples | 빈 라벨 이미지 **1K장** 추가 | FP 감소 |

**필터링 전/후 비교**:

| 데이터 | 필터 전 | 필터 후 | 제거율 |
|--------|---------|---------|--------|
| COCO person | 195,377 bbox | 76,453 bbox | **61% 제거** |
| Fallen pool | 33,018 bbox | 27,105 bbox | 18% 제거 |

```
필터링 스크립트:
  filter_coco_person.py  → dataset/coco_person_filtered/
  filter_fallen_pool.py  → dataset/fallen_pool_filtered/
```

**헬멧 데이터 처리**:
- 사용자가 aihub에서 area 기준 상위 추출한 `helmet_21k` 제공
- JSON→YOLO 변환: `convert_helmet_21k.py`
- 상위 0.1% 이상치 확인: `check_helmet_21k.py` → 클로즈업 샷으로 허용 가능

#### 10:44~11:50 — v9 서브셋 실험: 품질 필터링의 효과 입증

```
실험 조건:
  스크립트: build_dataset_v9.py + train_v9_sub.py
  서브셋: 3K bbox/cls, 30에폭, 로컬 (v8_sub와 동일 조건)
```

**v8 vs v9 비교 (동일 양, 동일 에폭)**:

| 지표 | v8_sub (필터 전) | v9_sub (필터 후) | 변화 |
|------|-----------------|-----------------|------|
| mAP50 | 0.801 | **0.868** | **+6.7%p** |
| mAP50-95 | 0.491 | **0.593** | **+10.2%p** |
| Precision | 0.768 | **0.814** | +4.6%p |
| Recall | 0.820 | **0.823** | +0.3%p |

**같은 양(3K bbox/cls), 같은 에폭(30)인데 필터링만으로 mAP50-95가 10%p 상승.**

이것이 "데이터 양 < 데이터 품질"을 숫자로 증명한 결과.

#### 12:00~12:30 — v9 풀 스케일업 + 서버 준비

품질 필터링 효과가 입증되었으므로, 양을 30K bbox/cls로 확대.

```
사용자가 helmet_60k 제공 (aihub에서 area 상위 30K씩 추출, 이상치 >=20% 제거)
스크립트: convert_helmet_60k.py → JSON→YOLO 변환
스크립트: build_dataset_v9_full.py → 최종 빌드
스크립트: train_v9_server.py → 서버 학습 코드
```

**datasets_v9 (풀) 최종 구성**:

| 클래스 | bbox | 소스 | 품질 보증 |
|--------|------|------|----------|
| helmet_o (0) | **30,000** | helmet_60k (area 3.5~20%) | area 필터링 |
| helmet_x (1) | **30,000** | helmet_60k (area 0.4~20%) | area 필터링 |
| person (2) | **30,000** | COCO 2017 (area >= 2%) | 필터링 |
| fallen (3) | **27,105** | 8소스 (0.5~70%) | fastdup + area 필터링 |
| negative | **1,000장** | aihub background | 빈 라벨 |

서버 학습 설정: 300에폭, batch=24, patience=30, AdamW, cos_lr

**디스크 정리**: ~72GB 삭제 (불필요 데이터, 레거시 폴더)

---

## Day 2.5: 2026-02-07 (별도 세션) — AIHub 헬멧 원본 데이터 추출

> 이 작업은 Day 2의 메인 라인과 병렬로 서버에서 진행된 작업이다. 상세 내용은 `devlog_dataset.md` 참조.

### 13:30~19:41 — 원본 487GB에서 헬멧 서브셋 추출

**목표**: AIHub 공사현장 안전장비 인식 데이터(45개 클래스)에서 헬멧 착용/미착용만 분리.

**주요 과정**:

1. **클래스 코드 파악** (13:30~13:50)
   - PDF 2개에서 전체 클래스 분류 체계표 발견
   - Class 07 = 안전모 착용, Class 08 = 안전모 미착용
   - `poppler-utils`의 `pdftotext`로 텍스트 추출 (PyPDF2는 실패)

2. **전체 라벨 스캔** (13:50~14:00)
   - 19개 라벨 zip 전수 스캔
   - 착용 1,354,120건 / 미착용 246,960건 → **원본부터 약 5:1 불균형**
   - 일부 장소(스튜디오, 주상복합)에는 헬멧 데이터 0건

3. **전체 추출** (13:57~14:47)
   - `.part` 26개 → zip 병합 (25.2GB)
   - 원천 이미지 zip 45개 파일 목록 인덱싱 (1,920,022개)
   - class 07/08 필터링 → 753,685개 파일 (487GB)
   - 스크립트: `extract_helmet_data.py`

4. **서브셋 생성** (14:47~19:41)
   - `helmet_30k/` (20GB): 1:1 균형, area 필터 없음 — 프로토타이핑용
   - `helmet_21k/` (13GB): 착용 2% + 미착용 1% area 필터, 배경 1K 포함
   - `helmet_60k/` (36GB): area 큰 순서 top 30K씩 — 최고 품질 bbox
   - 범용 스크립트: `extract_helmet_filtered.py`

---

## Day 3: 2026-02-08 — 서버 학습 결과 분석과 v10 준비

### v9 서버 학습 결과

```
학습 조건: 300에폭 (patience=30), batch=24, AdamW, cos_lr
데이터: datasets_v9 (30K bbox/cls, 품질 필터링)
총 소요: 68에폭 (약 10시간, epoch당 ~9분)
```

**최고 성능 (epoch 38)**:

| 지표 | v9_sub (3K, 30ep) | v9 서버 (30K, 68ep) | 변화 |
|------|-------------------|---------------------|------|
| mAP50 | 0.868 | **0.940** | +7.2%p |
| mAP50-95 | 0.593 | **0.702** | +10.9%p |
| Precision | 0.814 | **0.894** | +8.0%p |
| Recall | 0.823 | **0.885** | +6.2%p |

**심각한 문제: epoch 38 이후 과적합 붕괴 (overfitting collapse)**

```
학습 곡선 분석:
  epoch  1~38: mAP50 꾸준히 상승 (0.703 → 0.940)
  epoch 38:    최고점 (mAP50=0.940, mAP50-95=0.702)
  epoch 39~68: val/cls_loss 폭발적 증가 (1.08 → 2.33)
               mAP50 급락 (0.940 → 0.759)
               Recall 급락 (0.886 → 0.628)
               Precision은 유지 (0.890 → 0.923) ← 남은 탐지는 정확하지만 대부분 놓침

원인 추정:
  - cls_loss만 폭발 (box_loss/dfl_loss는 안정)
  → 분류 과적합: 특정 클래스 패턴을 외우기 시작
  - close_mosaic=10 (마지막 10에폭 mosaic off)이 epoch 58에서 발동
  → 이미 붕괴 진행 중이라 회복 불가
```

> patience=30이 best epoch(38) + 30 = epoch 68에서 정확히 종료시킴. 정상 작동.

### 근본 원인 분석: Helmet 이미지에 Person 라벨 부재

**IMPROVEMENT_PLAN.md에서 분석한 핵심 문제**:

```
현재 helmet_60k 데이터 (전체의 ~50%):
  - cls 0 (helmet_o): 머리 bbox만 존재 ✅
  - cls 1 (helmet_x): 머리 bbox만 존재 ✅
  - cls 2 (person): ❌ 없음 ← 문제!
  - cls 3 (fallen): ❌ 없음

실제 이미지에는 사람 전신이 보이지만, 머리 bbox만 라벨링됨.
→ 모델이 "사람 몸통 = background"로 잘못 학습
→ person miss rate 21% (121건)의 주요 원인
```

### 해결책: Auto-labeling 파이프라인 구축

```
스크립트: autolabel_helmet_person.py

파이프라인:
1. COCO pre-trained YOLO 로드 (person 탐지 검증됨)
2. helmet_60k 이미지 60,000장에 대해 person 추론
3. confidence >= 0.5, area >= 2% 필터링
4. helmet bbox가 person bbox 안에 포함되는지 IoA(Intersection over Area) >= 0.3 검증
5. 기존 helmet label에 person label (cls 2) 추가
6. 결과: dataset/helmet_60k_labeled/
7. 시각 검증: autolabel_check/ (랜덤 100장)
```

### v10 데이터셋 준비

```
스크립트: build_dataset_v10.py

v9 대비 개선점:
1. helmet 이미지에 person auto-label 추가 (helmet_60k_labeled)
2. negative samples 1K → 10K+ (helmet_negative_10k 추가)
3. 클래스당 30K bbox 균형 유지
```

### v10 학습 설정 (과적합 대응)

```
스크립트: train_v10_server.py

v9 대비 변경:
  patience: 30 → 50 (여유)
  warmup_epochs: 3 → 5
  mixup: 0.1 → 0.15 (증강 강화)
  copy_paste: 0 → 0.1 (증강 추가)
  degrees: 5.0 → 10.0 (회전 증강 강화)
  translate: 0.1 → 0.15

의도: augmentation 강화로 과적합 지연 → 더 오래, 더 잘 학습
```

---

## 전체 버전 히스토리 요약

| 버전 | 핵심 변경 | mAP50 | mAP50-95 | 실험 규모 |
|------|----------|-------|----------|----------|
| v5 | 최소 데이터 실험 | - | - | 서브셋 |
| v6 | WiderPerson → person 교체 | 0.782 | 0.462 | 1K/cls, 30ep |
| v7 | COCO person 교체 (FP↓) | 0.779 | 0.476 | 1K/cls, 30ep |
| v7 full | bbox 균형 (10K/cls) | - | - | 서버 (미완) |
| v8 | 8소스 fallen + fastdup 정제 | 0.801 | 0.491 | 3K/cls, 30ep |
| **v9 sub** | **품질 필터링 도입** | **0.868** | **0.593** | 3K/cls, 30ep |
| **v9 서버** | **30K/cls 스케일업** | **0.940** | **0.702** | 30K/cls, 68ep |
| v10 | auto-label + neg 10K (준비) | - | - | 서버 대기 |

---

## 핵심 의사결정 트리

```
[시작] Person/Fallen 성능 저조
  │
  ├─ Person 소스 교체
  │   ├─ aihub → WiderPerson (v6): FP 58%
  │   └─ aihub → COCO (v7): FP 47% ← 채택
  │
  ├─ Fallen 다양성 확보
  │   ├─ 단일 소스 (robo) → 8개 소스 통합 (v8)
  │   └─ fastdup 정제: 55% 중복 제거
  │
  ├─ 클래스 균형
  │   └─ 이미지 수 → bbox 수 기준 균형 (select_by_bbox)
  │
  ├─ 품질 필터링 (전환점)
  │   ├─ COCO person: area >= 2% (61% bbox 제거)
  │   ├─ Fallen: 0.5% <= area <= 70%
  │   ├─ Helmet: area >= 2%/1% (착용/미착용)
  │   └─ 결과: 같은 양에서 mAP50-95 +10.2%p
  │
  ├─ 스케일업
  │   └─ 3K → 30K bbox/cls → mAP50 0.940 달성
  │
  └─ 과적합 대응 (v10)
      ├─ auto-labeling: helmet 이미지에 person bbox 추가
      ├─ negative samples: 1K → 10K+
      └─ augmentation 강화
```

---

## 기술적 교훈 정리

| # | 교훈 | 근거 | 적용 |
|---|------|------|------|
| 1 | **데이터 품질 > 데이터 양** | 필터링만으로 mAP50-95 +10.2%p | area 기반 필터링 |
| 2 | **bbox area 필터링 필수** | COCO의 48%가 탐지 불가 크기 | area >= 2% 기준 |
| 3 | **Negative samples 필요** | FP 605건 → 빈 배경 미학습이 원인 | 1K~10K 빈 이미지 추가 |
| 4 | **다중 소스 = 중복 주의** | fall 8소스 55%가 중복 | fastdup 사전 정제 |
| 5 | **서브셋 → 풀 스케일업** | 3K로 검증 후 30K로 확장 | 로컬/서버 분리 전략 |
| 6 | **라벨 포맷 사전 확인** | aihub JSON ≠ YOLO txt | 변환 스크립트 필수 |
| 7 | **분류 과적합 주의** | epoch 38 이후 cls_loss 폭발 | patience + augmentation |
| 8 | **미라벨 영역 = 잘못된 학습** | helmet 이미지의 person 미라벨 → person miss | auto-labeling |

---

## 파일 참조

### 데이터 빌드 파이프라인

| 순서 | 스크립트 | 입력 | 출력 | 환경 |
|------|---------|------|------|------|
| 1 | `convert_helmet_30k.py` | helmet_30k (JSON) | helmet_pool/ (YOLO) | Windows |
| 2 | `collect_fallen_all.py` | fall* 8개 폴더 | fallen_pool/ | Windows |
| 3 | `analyze_fallen_pool.py` | fallen_pool/ | fastdup_fallen/ | **WSL** |
| 4 | `dedup_fallen_pool.py` | fastdup 결과 | fallen_pool 정제 | Windows |
| 5 | `filter_coco_person.py` | coco_person/ | coco_person_filtered/ | Windows |
| 6 | `filter_fallen_pool.py` | fallen_pool/ | fallen_pool_filtered/ | Windows |
| 7 | `convert_helmet_60k.py` | helmet_60k (JSON) | helmet_60k_yolo/ (YOLO) | Windows |
| 8 | `autolabel_helmet_person.py` | helmet_60k_yolo/ | helmet_60k_labeled/ | Windows |
| 9 | `build_dataset_v9_full.py` | 필터링된 소스들 | datasets_v9/ | Windows |
| 10 | `build_dataset_v10.py` | labeled 소스들 | datasets_v10/ | Windows |

### 학습 스크립트

| 스크립트 | 데이터셋 | 에폭 | 환경 | 비고 |
|---------|---------|------|------|------|
| `train_v6_sub.py` | v6 서브셋 | 30 | 로컬 | WiderPerson 실험 |
| `train_v7_sub.py` | v7 서브셋 | 30 | 로컬 | COCO person 실험 |
| `train_v8_sub.py` | v8 서브셋 | 30 | 로컬 | multi-source fallen |
| `train_v9_sub.py` | v9 서브셋 | 30 | 로컬 | 품질 필터링 검증 |
| `train_v9_server.py` | v9 풀 | 300 | 서버 | **best@38: mAP50=0.940** |
| `train_v10_server.py` | v10 풀 | 300 | 서버 | augmentation 강화 |

### 분석 스크립트

| 스크립트 | 용도 |
|---------|------|
| `check_coco_quality.py` | COCO person bbox 크기 분포 분석 |
| `check_helmet_quality.py` | helmet bbox 크기 분포 분석 |
| `check_fallen_quality.py` | fallen_pool 품질 검증 |
| `check_helmet_21k.py` | helmet_21k 이상치 확인 |
| `visualize_bbox.py` | bbox 시각화 저장 |

### 문서

| 파일 | 내용 |
|------|------|
| `PROJECT_CONTEXT.md` | 프로젝트 전체 컨텍스트 (데이터, 환경, 버전) |
| `PLAN_V8.md` | v8 데이터셋 빌드 계획서 |
| `DATASET_STATUS.md` | 데이터셋 현황 및 디스크 사용량 |
| `DATA_QUALITY_ISSUES.md` | v8 품질 이슈 및 해결 방안 |
| `IMPROVEMENT_PLAN.md` | v10 핵심 개선 과제 (auto-labeling) |
| `devlog_dataset.md` | AIHub 헬멧 원본 추출 상세 일지 |
| `DEVLOG.md` | 본 문서 — 전체 개발 일지 |

---

## 다음 단계

1. **v10 서버 학습**: auto-label + negative 10K 데이터로 학습
2. **v10 결과 평가**: person miss rate 개선 확인 (목표: 21% → 10% 이하)
3. **과적합 모니터링**: cls_loss 추이 확인, 붕괴 시점 비교
4. **데이터 추가 확장 여력**: 각 pool에 미사용 데이터 충분 (helmet_o 최대 ~29K까지)
5. **최종 목표**: mAP50 0.95+, mAP50-95 0.75+
