# Hoban 프로젝트 컨텍스트

## 프로젝트 개요
YOLO 기반 객체 탐지 모델 학습 프로젝트
- **목표**: 건설현장 안전 감지 (헬멧, 사람, 쓰러짐)
- **모델**: yolo26m (YOLO v8/26 Medium) - `yolo26m.pt` (42MB)
- **4개 클래스**:
  - 0: person_with_helmet (헬멧 착용)
  - 1: person_without_helmet (헬멧 미착용)
  - 2: person (일반 사람)
  - 3: fallen (쓰러진 사람)

## 환경
- **로컬**: Windows, RTX 5070 Ti 16GB, conda `gostock` env
  - Python 3.11.14, torch 2.9.1+cu128, ultralytics 8.4.12
  - 실행: `powershell.exe -Command "& 'C:\Users\lay4\anaconda3\envs\gostock\python.exe' 'script.py'"`
- **서버**: Linux, conda `llm` env, `lay@pluxity:~/hoban`
  - `torch.cuda.set_per_process_memory_fraction` 사용 금지 (CUDA forward compatibility 에러 발생)
  - `batch=-1` (auto), `workers=8`, 상대경로 사용
- **WSL**: Ubuntu, fastdup 전용 venv
  - `~/fastdup_env/bin/python` (Python 3.10)
  - fastdup 2.50 설치됨
  - 실행: `wsl -d Ubuntu -- bash -c "~/fastdup_env/bin/python /mnt/d/task/hoban/script.py"`
  - WSL sudo 비밀번호: `q`

## 데이터 소스 (v8 현재)
| 소스 | 내용 | 클래스 | 경로 |
|------|------|--------|------|
| **helmet_30k** | aihub 헬멧 30K (JSON→YOLO 변환) | 0 (helmet_o), 1 (helmet_x) | `helmet_pool/` |
| **COCO person** | COCO 2017 train person | 2 (person) | `dataset/coco_person/` |
| **fallen 8소스** | D:\task\dataset fall* 8개 폴더 | 3 (fallen) | `fallen_pool/` (정제 후) |

### helmet_pool (helmet_30k 변환 결과)
- 변환 스크립트: `convert_helmet_30k.py`
- 29,975 이미지, 65,783 bbox (cls0: 29,591 / cls1: 36,192)
- aihub JSON → YOLO txt 변환 (class "07"→0, "08"→1)

### fallen_pool (8개 소스 통합 + fastdup 정제)
| # | 소스 | prefix | 클래스 매핑 |
|---|------|--------|------------|
| 1 | fall detection ip camera.v3i | fall1_ | cls 0→fallen |
| 2 | Fall Detection.v4-aug3x | fall2_ | cls 0→fallen |
| 3 | Fall.v1i | fall3_ | cls 1(down)→fallen |
| 4 | fall.v1i (2) | fall4_ | cls 0→fallen |
| 5 | fall.v2i | fall5_ | cls 0→fallen |
| 6 | Fall.v3i | fall6_ | cls 0,1→fallen |
| 7 | fall.v4i | fall7_ | cls 0→fallen |
| 8 | fallen.v2i | fall8_ | cls 1→fallen |

- 수집: 52,945 이미지, 77,250 fallen bbox
- fastdup 분석: 82,714 pairs > 0.9 유사도, 4,326 outliers
- 정제 후: 23,661 이미지, 33,018 bbox (29,284장 제거, 55%)

## 데이터셋 버전 이력

### v6: aihub + WiderPerson + robo fallen
- 빌드: `build_dataset_v6.py`
- WiderPerson을 person(class 2)으로 사용
- 풀: 42,017 train / 4,455 valid
- 서브셋: 클래스당 1,000장 (4,000 train / 1,000 valid)
- **결과** (v6_sub, 30 에폭): mAP50=0.782, mAP50-95=0.462

### v7_sub: aihub + COCO person + robo fallen (서브셋)
- 빌드: `build_dataset_v7.py`
- COCO person을 person(class 2)으로 사용 (WiderPerson 대체)
- 서브셋: 클래스당 1,000장 (4,000 train / 1,000 valid)
- **결과** (v7_sub, 30 에폭): mAP50=0.779, mAP50-95=0.476
- COCO가 person FP 58%→47%로 개선, mAP50-95 더 높음

### v7 (이전): aihub + COCO person + robo fallen (bbox 균형)
- 빌드: `build_dataset_v7_full.py`
- bbox 기준 클래스당 10,000개 균형
- Train: 12,829장 / Valid: 2,860장
- **삭제됨** (delete 폴더에서 30GB 정리 시 함께 삭제)

### v8 (현재): helmet_30k + COCO person + multi-source fallen (bbox 균형)
- 빌드: `build_dataset_v8.py`
- **bbox 기준 클래스당 10,000개 완벽 균형**
- 헬멧: helmet_30k (aihub JSON→YOLO 변환) → helmet_pool
- 사람: COCO 2017 person → dataset/coco_person
- 쓰러짐: 8개 fall 소스 통합 → fastdup 정제 → fallen_pool
- 현재 상태:
  - Train: 17,855장 (helmet_o: 10,001 / helmet_x: 10,001 / person: 10,001 / fallen: 10,001 bbox)
  - Valid: 3,562장 (각 클래스 2,000 bbox)
- data.yaml 경로: 상대경로 `datasets_v8` (서버 호환)
- **서버 학습 대기 중** - train_v8_server.py 준비 완료

## v6 vs v7 비교 (서브셋 30 에폭)
| 지표 | v6 (WiderPerson) | v7 (COCO) |
|------|-------------------|-----------|
| mAP50 | 0.782 | 0.779 |
| mAP50-95 | 0.462 | **0.476** |
| person FP | 58% | **47%** |
| person 정탐율 | 65% | **67%** |

**결론**: COCO person이 일반화 성능 더 좋음 → v7 채택

## fastdup 분석 결과 (v8 fallen_pool)
- fallen_pool 52,945장에 대해 실행 완료
- 유사도 > 0.9: 82,714쌍 (소스 간 교차 중복 다수)
- 이상치: 4,326개, 깨진 이미지: 0개
- `dedup_fallen_pool.py`로 29,284장 제거 (55%)
- 소스 우선순위: fall8(10) > fall4(7) > fall1(6) > fall3,6(5) > fall5,7(4) > fall2(1)
- 정제 후 소스별: fall4: 6,811 / fall2: 5,538 / fall8: 5,498 / fall6: 2,815 등

## 파일 구조

### 스크립트 (v8 파이프라인)
| 파일 | 용도 |
|------|------|
| `convert_helmet_30k.py` | helmet_30k JSON→YOLO 변환 → helmet_pool |
| `collect_fallen_all.py` | 8개 fall 소스 통합 수집 → fallen_pool |
| `analyze_fallen_pool.py` | fallen_pool fastdup 분석 (WSL) |
| `dedup_fallen_pool.py` | fastdup 결과 기반 fallen_pool 중복/이상치 제거 |
| `build_dataset_v8.py` | **v8 최종 빌드** (helmet_pool + coco_person + fallen_pool) |
| `train_v8_server.py` | **v8 서버 학습** (100에폭, batch=-1, AdamW) |

### 스크립트 (이전 버전, 참고용)
| 파일 | 용도 |
|------|------|
| `build_dataset_v7_full.py` | v7 빌드 (select_by_bbox 패턴 원본) |
| `download_coco_person.py` | COCO 2017 person 다운로드+YOLO 변환 |
| `train_v7_server.py` | v7 서버 학습 |

### 데이터 폴더 (현재 존재)
| 폴더 | 내용 |
|------|------|
| `dataset/` | 원본 데이터 (coco_person, datasets_merged) |
| `helmet_pool/` | helmet_30k 변환 결과 (29,975 imgs) |
| `fallen_pool/` | 8소스 통합 + fastdup 정제 (23,661 imgs) |
| `fastdup_fallen/` | fallen_pool fastdup 분석 결과 |
| `datasets_v8/` | **현재 데이터셋** (17,855 train / 3,562 valid) |

### 레거시 폴더 (현재 남아있음, 삭제 검토 가능)
| 폴더 | 크기 | 내용 |
|------|------|------|
| datasets_clean | 16.94 GB | aihub+robo 정제 데이터 |
| WiderPerson | 0.96 GB | 보행자 데이터 (v6 사용, v8 미사용) |
| hoban_yolo26m5 | 0.09 GB | 이전 학습 결과 |
| runs | 4.73 GB | ultralytics 학습 로그 |
| dataset/datasets_merged | 20.68 GB | aihub+robo 병합 데이터 |

### 삭제 완료된 폴더
- datasets_v5~v7, datasets_v7_sub, datasets_v6_sub (30GB)
- hoban_v6, hoban_v6_sub, hoban_v7_sub
- fastdup_results, fastdup_aihub, fastdup_robo, fastdup_coco
- datasets_v7.zip (8.5GB)
- dataset/helmet_30k (19.78GB) - helmet_pool로 변환 완료 후 삭제

## 다음 작업 (TODO)
1. **서버에서 v8 학습**: `datasets_v8/` + `train_v8_server.py` + `yolo26m.pt` 서버 전송 후 학습
2. **v8 학습 결과 평가**: mAP50, mAP50-95로 v7 서브셋(0.78) 대비 개선 확인
3. **오류 분석**: FP/FN 많은 케이스 분석 → 데이터 보강 방향 결정
4. **데이터 확장 (필요 시)**: TARGET_BBOX를 10K→25K로 올려서 재빌드 (pool에 여유 충분)
5. **레거시 정리 (선택)**: datasets_clean, WiderPerson, runs 등 ~43GB 삭제 가능
6. **성능 목표**: mAP 0.9 달성

## 데이터 확장 여력
현재 각 pool에서 사용하지 않은 bbox가 남아있어 추가 수집 없이 확장 가능.

| 클래스 | Pool 전체 | v8 사용 | 미사용 | 최대 확장 |
|--------|----------|---------|--------|----------|
| 0 (helmet_o) | 29,591 | 10,000 | 19,591 | ~29K (병목) |
| 1 (helmet_x) | 36,192 | 10,000 | 26,192 | ~36K |
| 2 (person) | 195,377 | 10,004 | 185,373 | ~195K |
| 3 (fallen) | 33,018 | 10,000 | 23,018 | ~33K |

→ 클래스당 최대 ~29K bbox까지 균형 확장 가능 (helmet_o가 병목)

## 작업 이력

### v6 (완료)
- aihub 헬멧 + WiderPerson 사람 + robo fallen
- 서브셋 30에폭: mAP50=0.782, mAP50-95=0.462

### v7 (완료, 데이터 삭제됨)
- aihub 헬멧 + COCO person + robo fallen (단일 소스)
- 서브셋 30에폭: mAP50=0.779, mAP50-95=0.476
- COCO person이 WiderPerson보다 FP 낮음 확인
- bbox 균형 풀버전 빌드 + fastdup 중복 제거
- 12,829 train / 2,860 valid

### v8 (현재, 학습 대기)
1. helmet_30k JSON→YOLO 변환 → helmet_pool (29,975장, 65,783 bbox)
2. 8개 fall 소스 통합 수집 → fallen_pool (52,945장, 77,250 bbox)
3. WSL fastdup 분석 → 82,714 중복쌍, 4,326 이상치
4. 중복 제거 → 29,284장 삭제 (55%), 23,661장/33,018 bbox 남음
5. datasets_v8 빌드 → 17,855 train / 3,562 valid (클래스당 10K bbox 균형)
6. train_v8_server.py 작성 완료
7. 디스크 정리: 구버전 30GB + helmet_30k 19.78GB 삭제

### v8 vs v7 개선점
| 항목 | v7 | v8 |
|------|----|----|
| helmet 소스 | aihub (datasets_merged) | helmet_30k 직접 변환 |
| helmet_x bbox | 4,496 (부족) | **10,000** (충분) |
| fallen 소스 | robo 단일 | **8개 소스 통합+정제** |
| fallen bbox | 7,014 | **10,000** |
| 클래스 균형 | 불균형 | **완벽 균형 (10K/cls)** |
| train 이미지 | 12,829 | **17,855** |

## YOLO 라벨 형식
```
class x_center y_center width height
```
- 값은 0~1 사이로 정규화
- 한 줄에 하나의 bbox

## 주요 주의사항
- Windows에서 multiprocessing 사용 시 반드시 `if __name__ == '__main__': freeze_support(); main()` 필요
- 서버에서 `torch.cuda.set_per_process_memory_fraction` 사용 금지
- helmet_30k의 labels/는 JSON 형식 (YOLO txt 아님) → convert_helmet_30k.py로 변환 필요
- fallen 데이터 8개 소스는 55% 중복률 → fastdup 정제 필수
- fire_smoke 폴더는 fallen 소스에서 제외 (사용자 지시)
- datasets_clean은 v8에서 사용하지 않음 (사용자 지시)
