# 최적 품질 데이터셋 구축 계획 (v8)

## Context
현재 datasets_v7의 fallen 데이터는 robo 단일 소스에서만 가져왔다. `D:\task\dataset`에 8개 fall 폴더에 다양한 fallen 데이터가 분산되어 있으나 품질이 검증되지 않았다. 모든 fallen 소스를 수집하고 fastdup로 정제하여 최적 품질의 v8 데이터셋을 구축한다.

---

## 데이터 소스

### 1. Helmet (cls 0: 착용, cls 1: 미착용)
**경로**: `D:\task\hoban\dataset\helmet_30k`

| Split | 착용 (wearing) | 미착용 (not_wearing) |
|-------|---------------|---------------------|
| training | 14,241장 / 14,241 labels | 14,239장 / 14,239 labels |
| validation | 745장 / 745 labels | 750장 / 750 labels |

- **라벨 형식**: JSON (aihub 원본) → **YOLO txt 변환 필요**
- `box: [x1, y1, x2, y2]` 픽셀 좌표 → `class x_center y_center width height` 정규화
- class "07" (wearing) → cls 0, class "08" (not_wearing) → cls 1
- `image.resolution: [width, height]` 로 정규화

### 2. Person (cls 2)
**경로**: `D:\task\hoban\dataset\coco_person`

| 항목 | 값 |
|------|---|
| 이미지 | 29,778장 |
| bbox | 195,377개 (cls 2) |
| 라벨 형식 | **YOLO txt** (변환 불필요) |

### 3. Fallen (cls 3) - 8개 소스에서 수집
**경로**: `D:\task\dataset\fall*` (8개 폴더)

| # | 폴더명 | Train imgs | Fallen bbox | 클래스 매핑 | 비고 |
|---|--------|-----------|-------------|------------|------|
| 1 | fall detection ip camera.v3i | 6,745 | 4,285 | cls 0(fall)→fallen | IP카메라 |
| 2 | Fall Detection.v4-aug3x | 9,438 | 9,441 | cls 0→fallen | 3x 증강, fastdup 판단 |
| 3 | Fall.v1i | 5,445 | 6,212 | cls 1(down)→fallen | cls 0(10-)→제외 |
| 4 | fall.v1i (2) | 5,072 | 6,364 | cls 0→fallen | 단일 클래스 |
| 5 | fall.v2i | 1,659 | 1,016 | cls 0(fall)→fallen | 소규모 |
| 6 | Fall.v3i | 5,313 | 5,460 | cls 0,1→fallen | |
| 7 | fall.v4i | 7,775 | 2,309 | cls 0(falling)→fallen | |
| 8 | fallen.v2i | 14,607 | 30,971 | cls 1(2_fallen)→fallen | 최대 소스 |

- **라벨 형식**: YOLO txt (변환 불필요, 클래스 번호만 재매핑)
- **예상 총 fallen bbox**: ~66,000+ (train+valid+test 전부 수집)
- **제외**: Fall.v1i의 cls 0 "10-" (의미 불명)

---

## 실행 계획

### Step 1: helmet_30k JSON→YOLO 변환 (`convert_helmet_30k.py`)
helmet_30k의 JSON 라벨을 YOLO txt로 변환하여 통합 폴더에 저장

```
D:\task\hoban\helmet_pool\
  images/   ← helmet_30k의 모든 이미지
  labels/   ← YOLO txt 변환 (cls 0: wearing, cls 1: not_wearing)
```

**변환 로직**:
```
JSON box [x1, y1, x2, y2] (pixel)
→ x_center = (x1 + x2) / 2 / width
→ y_center = (y1 + y2) / 2 / height
→ w = (x2 - x1) / width
→ h = (y2 - y1) / height
→ "cls x_center y_center w h"
```

### Step 2: fallen 데이터 통합 수집 (`collect_fallen_all.py`)
8개 fall 폴더에서 fallen bbox만 추출하여 하나의 pool로 수집

```
D:\task\hoban\fallen_pool\
  images/   ← prefix로 소스 구분 (fall1_*, fall2_*, ..., fall8_*)
  labels/   ← 통일된 class=0 (fallen만)
```

- 각 소스의 train + valid + test 모두 수집 (데이터 최대화)
- fallen 클래스만 추출, 모두 class 0으로 통일

### Step 3: fastdup 분석 (`analyze_fallen_pool.py`)
WSL에서 fallen_pool 전체에 대해 fastdup 실행

```bash
wsl -d Ubuntu -- bash -c "~/fastdup_env/bin/python /mnt/d/task/hoban/analyze_fallen_pool.py"
```

**분석 항목**:
1. 유사도 > 0.9 중복 쌍 (소스 간 교차 중복 포함)
2. 이상치(outliers) → 깨진/무관 이미지
3. 이미지 통계 → 해상도, 유효성

### Step 4: 중복/이상치 제거 (`dedup_fallen_pool.py`)
- 유사도 > 0.9: 그리디 제거 (우선순위: fallen.v2i > 나머지)
- 이상치: fastdup outlier score 기반 제거
- 빈 라벨 제거

### Step 5: v8 데이터셋 빌드 (`build_dataset_v8.py`)

**데이터 소스 및 목표**:
| 클래스 | 소스 | 경로 | 목표 bbox |
|--------|------|------|-----------|
| 0 (helmet_o) | helmet_30k (변환 후) | `helmet_pool/` | 10,000 |
| 1 (helmet_x) | helmet_30k (변환 후) | `helmet_pool/` | 최대한 (~14,239) |
| 2 (person) | coco_person | `dataset/coco_person/` | 10,000 |
| 3 (fallen) | fallen_pool (정제 후) | `fallen_pool/` | 10,000 |

**빌드 로직**: `build_dataset_v7_full.py`의 `select_by_bbox()` 패턴 재사용
**출력**: `datasets_v8/` (train/valid split, data.yaml 포함)

### Step 6: 최종 fastdup 검증
v8 빌드 후 전체 train에 대해 fastdup 재실행하여 교차 소스 중복 최종 확인

### Step 7: 서버 학습
`train_v8_server.py` (v7과 동일 하이퍼파라미터, data.yaml만 변경)

---

## 핵심 차이점 (v7 → v8)
| 항목 | v7 | v8 |
|------|----|----|
| helmet 소스 | datasets_merged (aihub YOLO) | helmet_30k (JSON→YOLO 변환) |
| helmet_x bbox | ~4,500 | ~14,239 (**3배 증가**) |
| person 소스 | coco_person (삭제됨, 재다운로드 필요) | dataset/coco_person (이미 존재) |
| fallen 소스 | robo 단일 소스 (~12K bbox) | 8개 소스 통합 (~66K bbox, fastdup 정제) |
| 품질 관리 | fastdup 사후 적용 | fastdup 사전 정제 후 빌드 |

## 파일 참조
| 파일 | 용도 |
|------|------|
| `build_dataset_v7_full.py` | `select_by_bbox()` 함수 패턴 재사용 |
| `dedup_v7.py` | fastdup 결과 기반 중복 제거 패턴 재사용 |

## 신규 작성 파일
| 파일 | 용도 |
|------|------|
| `convert_helmet_30k.py` | helmet_30k JSON→YOLO 변환 (Windows) |
| `collect_fallen_all.py` | 8개 소스에서 fallen 수집/통합 (Windows) |
| `analyze_fallen_pool.py` | fastdup 분석 (WSL) |
| `dedup_fallen_pool.py` | 중복/이상치 제거 (Windows) |
| `build_dataset_v8.py` | 최종 v8 빌드 (Windows) |
| `train_v8_server.py` | 서버 학습 스크립트 |

## 검증
1. **convert 후**: helmet_pool 이미지/라벨 수, 클래스별 bbox 통계
2. **collect 후**: 소스별 수집량, 총 이미지/bbox 통계
3. **fastdup 후**: 중복 쌍 수, 소스 간 교차 중복률, 이상치 수
4. **dedup 후**: 제거량, 남은 이미지/bbox
5. **v8 빌드 후**: 클래스별 bbox 분포 확인
6. **학습 후**: mAP50, mAP50-95로 v7 대비 비교

## 실행 순서
```
1. convert_helmet_30k.py     (Windows, ~2분) - JSON→YOLO 변환
2. collect_fallen_all.py     (Windows, ~5분) - 8개 소스에서 fallen 수집
3. analyze_fallen_pool.py    (WSL fastdup, ~20분) - 품질 분석
4. dedup_fallen_pool.py      (Windows, ~2분) - 정제
5. build_dataset_v8.py       (Windows, ~5분) - 최종 빌드
6. fastdup 최종 검증         (WSL, ~10분) - 교차 중복 확인
7. 서버 전송 + 학습
```
