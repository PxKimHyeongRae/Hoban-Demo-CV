# Hoban 데이터셋 현황 (2025-02-07)

## 1. 프로젝트 개요

YOLO 기반 건설현장 안전 감지 모델 학습용 데이터셋

| 항목 | 내용 |
|------|------|
| 모델 | yolo26m.pt (42MB) |
| 클래스 | 4개 (helmet_o, helmet_x, person, fallen) |
| 현재 데이터셋 | **datasets_v8** |
| 학습 스크립트 | train_v8_server.py |
| 상태 | **서버 학습 대기 중** |

---

## 2. datasets_v8 (학습용 데이터셋) - 8.00 GB

최종 빌드된 학습 데이터셋. 클래스당 bbox 균형 맞춤.

### 클래스 분포

| 클래스 | 이름 | Train bbox | Valid bbox |
|--------|------|-----------|-----------|
| 0 | person_with_helmet | 10,000 | 2,000 |
| 1 | person_without_helmet | 10,000 | 2,000 |
| 2 | person | 10,004 | 2,000 |
| 3 | fallen | 10,000 | 2,002 |
| **합계** | | **40,004** | **8,002** |

### 이미지 수

| Split | 이미지 | 라벨 |
|-------|--------|------|
| Train | 17,855 | 17,855 |
| Valid | 3,562 | 3,562 |
| **합계** | **21,417** | **21,417** |

### 데이터 소스별 구성

| 클래스 | 소스 | 파일 prefix |
|--------|------|------------|
| 0, 1 (helmet) | helmet_pool (aihub helmet_30k 변환) | (원본 stem 그대로) |
| 2 (person) | dataset/coco_person (COCO 2017) | `coco_` |
| 3 (fallen) | fallen_pool (8개 소스 통합+정제) | `fallen_` |

### data.yaml
```yaml
path: datasets_v8
train: train/images
val: valid/images
nc: 4
names:
  0: person_with_helmet
  1: person_without_helmet
  2: person
  3: fallen
```

---

## 3. 데이터 Pool (원본 저장소)

datasets_v8을 빌드할 때 사용한 원본 데이터 풀.

### helmet_pool - 19.75 GB

aihub helmet_30k의 JSON 라벨을 YOLO 형식으로 변환한 결과.

| 항목 | 수량 |
|------|------|
| 이미지 | 29,975장 |
| 총 bbox | 65,783개 |
| cls 0 (helmet_o) | 29,591 bbox |
| cls 1 (helmet_x) | 36,192 bbox |

- 변환 스크립트: `convert_helmet_30k.py`
- 원본: `dataset/helmet_30k/` (aihub JSON 형식)

### fallen_pool - 1.33 GB

8개 fall detection 데이터셋을 통합 수집 후 fastdup로 중복 제거한 결과.

| 항목 | 수량 |
|------|------|
| 이미지 | 23,661장 |
| 총 bbox | 33,018개 |
| cls 0 (fallen) | 33,018 bbox |

**수집 → 정제 과정:**
- 수집: 52,945장 / 77,250 bbox (8개 소스)
- fastdup 분석: 82,714 중복쌍 (유사도 > 0.9)
- 제거: 29,284장 (55%)
- 정제 후: 23,661장 / 33,018 bbox

**8개 소스:**

| # | 소스 폴더 | prefix | 원본 위치 |
|---|----------|--------|----------|
| 1 | fall detection ip camera.v3i | fall1_ | D:\task\dataset |
| 2 | Fall Detection.v4-aug3x | fall2_ | D:\task\dataset |
| 3 | Fall.v1i | fall3_ | D:\task\dataset |
| 4 | fall.v1i (2) | fall4_ | D:\task\dataset |
| 5 | fall.v2i | fall5_ | D:\task\dataset |
| 6 | Fall.v3i | fall6_ | D:\task\dataset |
| 7 | fall.v4i | fall7_ | D:\task\dataset |
| 8 | fallen.v2i | fall8_ | D:\task\dataset |

- 정제 스크립트: `dedup_fallen_pool.py`
- 분석 스크립트: `analyze_fallen_pool.py` (WSL fastdup)
- 수집 스크립트: `collect_fallen_all.py`

### dataset/coco_person - 5.51 GB

COCO 2017 train에서 person 클래스만 추출한 데이터.

| 항목 | 수량 |
|------|------|
| 이미지 | 29,778장 |
| 총 bbox | 195,377개 |
| cls 2 (person) | 195,377 bbox |

---

## 4. 분석 결과 폴더

### fastdup_fallen - 0.67 GB

fallen_pool에 대한 fastdup 분석 결과.

| 분석 항목 | 결과 |
|-----------|------|
| 유사도 > 0.9 쌍 | 82,714개 |
| 이상치 | 4,326개 |
| 깨진 이미지 | 0개 |
| 총 분석 이미지 | 52,945장 |

---

## 5. 기존/레거시 폴더

현재 사용하지 않지만 남아있는 폴더.

| 폴더 | 크기 | 파일 수 | 내용 | 삭제 가능 여부 |
|------|------|---------|------|--------------|
| datasets_clean | 16.94 GB | 90,337 | aihub+robo 정제 데이터 | 검토 필요 |
| WiderPerson | 0.96 GB | 22,392 | 보행자 데이터 (v6에서 사용, v8 미사용) | 삭제 가능 |
| hoban_yolo26m5 | 0.09 GB | 24 | 이전 버전 학습 결과 | 삭제 가능 |
| runs | 4.73 GB | 124 | ultralytics 학습 로그 | 삭제 가능 |
| dataset/helmet_30k | 19.78 GB | 59,952 | aihub 헬멧 원본 (JSON) | helmet_pool 있으면 삭제 가능 |
| dataset/datasets_merged | 20.68 GB | 115,047 | aihub+robo 병합 데이터 | datasets_clean 있으면 삭제 가능 |

**레거시 합계: ~63.18 GB**

---

## 6. 전체 디스크 사용량 요약

| 구분 | 폴더 | 크기 |
|------|------|------|
| **활성** | datasets_v8 | 8.00 GB |
| **활성** | helmet_pool | 19.75 GB |
| **활성** | fallen_pool | 1.33 GB |
| **활성** | dataset/coco_person | 5.51 GB |
| **활성** | fastdup_fallen | 0.67 GB |
| **활성 소계** | | **35.26 GB** |
| 레거시 | datasets_clean | 16.94 GB |
| 레거시 | WiderPerson | 0.96 GB |
| 레거시 | hoban_yolo26m5 | 0.09 GB |
| 레거시 | runs | 4.73 GB |
| 레거시 | dataset/helmet_30k | 19.78 GB |
| 레거시 | dataset/datasets_merged | 20.68 GB |
| **레거시 소계** | | **63.18 GB** |
| **전체 합계** | | **~98.44 GB** |

---

## 7. 빌드 파이프라인 스크립트

| 순서 | 스크립트 | 용도 |
|------|---------|------|
| 1 | `convert_helmet_30k.py` | helmet_30k JSON → YOLO 변환 → helmet_pool |
| 2 | `collect_fallen_all.py` | 8개 fall 소스 통합 수집 → fallen_pool |
| 3 | `analyze_fallen_pool.py` | fallen_pool fastdup 분석 (WSL 실행) |
| 4 | `dedup_fallen_pool.py` | fastdup 결과 기반 중복/이상치 제거 |
| 5 | `build_dataset_v8.py` | 최종 datasets_v8 빌드 (bbox 균형) |
| 6 | `train_v8_server.py` | 서버 학습 (100 epochs, AdamW, batch=-1) |

---

## 8. 다음 단계

1. **서버 학습**: datasets_v8 + train_v8_server.py + yolo26m.pt → 서버 전송 후 학습
2. **레거시 정리**: 63GB 레거시 폴더 삭제 검토 (디스크 확보)
3. **학습 결과 평가**: mAP50, mAP50-95로 이전 버전 대비 비교
