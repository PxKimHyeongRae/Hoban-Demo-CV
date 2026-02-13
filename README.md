# Hoban - 건설현장 안전 탐지 (YOLO)

건설현장 CCTV 영상에서 헬멧 착용/미착용, 쓰러진 사람을 실시간 탐지하는 YOLO 기반 객체 탐지 프로젝트.

## 환경

| 항목 | 사양 |
|------|------|
| 모델 | YOLOv8/v26 Medium (yolo26m.pt, 42MB) |
| GPU | RTX 4080 16GB (Linux server) |
| Conda | `llm` |
| 데이터 | /data/aihub_data (AIHub #163) |

## 현재 상태

### 최고 성능 모델

| 모델 | 클래스 | mAP50 | mAP50-95 | 비고 |
|------|--------|-------|----------|------|
| v13 stage2 | 2 (helmet_o, helmet_x) | **0.945** | **0.727** | 1280px, curriculum learning |
| v9 @epoch38 | 4 (helmet_o/x, person, fallen) | 0.940 | 0.702 | 이후 과적합 붕괴 |
| go500 fine-tune | 2 (helmet_o, helmet_x) | 0.748 | 0.465 | CCTV 도메인 특화 |

### 진행 방향: GO500 + SAHI

CCTV 소형 객체 탐지를 위해 go500 fine-tune 모델 + SAHI 조합으로 진행 중.
자세한 내용은 [DEVLOG_GO500_SAHI.md](DEVLOG_GO500_SAHI.md) 참조.

- SAHI 오프라인 pseudo-labeling: **93.7% 탐지율** (기존 38.7%)
- CVAT 검수용 데이터 생성 완료 (1815장, 4분할)
- 실시간 적용은 속도 문제로 대안 검토 중

## 클래스 변천

| 버전 | 클래스 수 | 구성 |
|------|-----------|------|
| v6-v10 | 4 | helmet_o(0), helmet_x(1), person(2), fallen(3) |
| v11-v12 | 3 | helmet_o(0), helmet_x(1), fallen(2) |
| v13+ | 2 | person_with_helmet(0), person_without_helmet(1) |

## 프로젝트 구조

```
hoban/
├── README.md                    # 이 파일
├── DEVLOG.md                    # 메인 개발일지 (v6-v10)
├── DEVLOG_GO500_SAHI.md         # GO500 + SAHI 개발일지
│
├── train_v13.py                 # v13 curriculum learning (3-stage)
├── train_v14.py                 # v14 crop 학습
├── train_v15.py                 # v15 tiled 학습
├── train_go500_finetune.py      # go500 fine-tune
├── train_go500_scratch.py       # go500 from-scratch
├── train_v13_crop.py            # v13 crop 실험
├── train_manual_v1.py           # 수동 라벨링 데이터 학습
│
├── build_go500.py               # go500 데이터셋 빌드
├── build_v14_crop.py            # v14 crop 데이터셋 빌드
├── build_v15_tiled.py           # v15 tiled 데이터셋 빌드
├── build_crop_experiment.py     # crop 실험 데이터셋
├── generate_pseudo_labels.py    # SAHI pseudo-label 생성
├── visualize_bbox.py            # bbox 시각화
│
├── docs/                        # 참고 문서
│   ├── PROJECT_CONTEXT.md       # 프로젝트 전체 컨텍스트
│   ├── IMPROVEMENT_PLAN.md      # v10 개선 계획 (auto-labeling)
│   ├── PLAN_V15.md              # v15 tiled 학습 계획
│   ├── DATA_QUALITY_ISSUES.md   # 데이터 품질 분석
│   ├── DATASET_STATUS.md        # v8 데이터셋 현황
│   ├── DEVLOG_2026-02-07.md     # 2/7 상세 개발일지
│   ├── devlog_dataset.md        # 데이터셋 추출 일지
│   ├── PLAN_V8.md               # v8 빌드 계획
│   └── TIMELINE_2026-02-07.md   # 2/7 타임라인
│
├── archive/                     # 이전 버전 스크립트
│   ├── data_processing/         # v6-v10 데이터 처리 (23개)
│   └── training/                # v6-v12 학습 스크립트 (14개)
│
├── datasets_*/                  # 데이터셋 (gitignore)
├── hoban_*/                     # 학습 결과 (gitignore)
└── snapshot_detect/             # CCTV 스냅샷 (gitignore)
```

## 핵심 학습

1. **데이터 품질 > 수량**: 필터링만으로 mAP50-95 +10.2%p
2. **bbox area 필터링 필수**: <0.5% area는 640px에서 탐지 불가
3. **Negative 샘플**: 배경 FP 감소에 효과적
4. **SGD > AdamW**: 장기 학습 시 과적합 붕괴 방지
5. **SAHI**: 오프라인 소형 객체 탐지에 탁월, 실시간은 속도 제약

## 관련 프로젝트

- **video_indoor** (`~/video_indoor/`): 실시간 CCTV 탐지 서버 (Flask + PluxMTX + SAFERS)
- 현재 모델: hoban_v12 (3-class)
- SAHI 실시간 적용 검토 중

## 주요 데이터 소스

| 경로 | 설명 | 크기 |
|------|------|------|
| `/data/aihub_data/helmet_60k/` | AIHub 헬멧 데이터 (최고 품질) | 36GB |
| `/data/aihub_data/helmet_negative_10k_v3/` | YOLO 검증 negative | 4.7GB |
| `/data/helmet/` | 2-class 헬멧 데이터셋 | 50GB |
| `/data/unified_safety_all/` | 8-class 통합 안전 데이터 | 258GB |
