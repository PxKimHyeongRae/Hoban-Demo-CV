# YOLO 헬멧 탐지 프로젝트 히스토리

## 버전 계보도

```
[Phase 1: 4-클래스] helmet_o / helmet_x / person / fallen
v6 → v7 → v8 → v9(★) → v10
                  │
                  └─ 핵심: 데이터 품질 필터링 = +10.2%p

[Phase 2: 3-클래스] helmet_o / helmet_x / fallen  (person 제거)
v11 → v12  (성능 하락 → 방향 재검토)

[Phase 3: 2-클래스] person_with_helmet / person_without_helmet
v13 stage1 → stage2(★★) → stage3  (커리큘럼 학습)
  │
  └─ 최고 범용 성능: mAP50-95=0.946

[Phase 4: CCTV 도메인 특화]
go500 → go2k_finetune → go2k_v2(★) → v3 → v4 → v5 → v6 → v7
  │                        │
  │                        └─ v13 8K + go2k 479장 x8
  └─ 보조: manual_v1, 3k_finetune, v14(crop), v15(tile)

[Phase 5: 실험 최적화]
quick_baseline → a1(copy_paste) → a2(freeze) → a3(loss) → a4(aug)
  └─ Phase A: v2 하이퍼파라미터 최적성 확인
  └─ Phase B: WBF 앙상블 F1=0.932 (leakage 포함)
  └─ Clean Eval: 실제 F1=0.891
```

## 핵심 모델 성능 비교

### 범용 모델 (YOLO native val)

| 모델 | 클래스 | mAP50 | mAP50-95 | 비고 |
|------|--------|-------|----------|------|
| v9 @ep38 | 4 | 0.940 | 0.702 | 이후 과적합 붕괴 |
| **v13 stage2** | **2** | **0.895** | **0.946** | **최고 범용** |
| v15 | 2 | 0.912 | 0.909 | 타일 학습 |

### CCTV 특화 모델 (SAHI 평가, go2k_manual 604장)

| 모델 | mAP50 (val) | SAHI F1 (전체) | SAHI F1 (Clean 125장) |
|------|-------------|---------------|----------------------|
| go2k_v2 | 0.925 | 0.914 | 0.845 |
| go2k_v3 | 0.951 | 0.881 | - |
| go2k_v5 | 0.920 | 0.891 | - |
| **v2+v3+v5 WBF** | - | **0.932** | **0.891** |

### 실제 일반화 성능 (Clean Eval, leakage 없음)

| 설정 | F1 | Prec | Rec |
|------|-----|------|-----|
| v2 단독 | 0.845 | 0.834 | 0.857 |
| v2 per-class | 0.856 | 0.868 | 0.845 |
| **v2+v3+v5 WBF** | **0.891** | **0.913** | **0.869** |

## 핵심 발견사항

| 발견 | 증거 | 영향 |
|------|------|------|
| 데이터 품질 > 양 | v8→v9: 필터링만 +10.2%p | 모든 데이터에 area 필터 적용 |
| SGD > AdamW 장기 학습 | v9 과적합 붕괴 방지 | go2k v2+에서 SGD 채택 |
| Data leakage +4~9%p | Clean 125장 F1=0.891 vs 전체 0.932 | 모든 평가를 Clean 기준으로 |
| x1~x7 = byte copy | MD5 동일 확인 | 실제 augmentation 필요 |
| v13 bbox 12.8x 큼 | median area 0.005 vs 0.0004 | 도메인 필터링 필요 |
| FN 93%가 <30px | Clean eval bbox 크기 분석 | 소형 객체가 핵심 병목 |

## 파일 분류

### 현역 (사용 중)

| 파일 | 용도 |
|------|------|
| train_go2k_v2.py | 주력 학습 스크립트 |
| eval_clean.py | Clean eval (leakage-free) |
| eval_phase_b.py | WBF/NMS 앙상블 평가 |
| eval_v7_compare.py | v7 vs v2 비교 |
| detect_go2k_sahi.py | SAHI 탐지 (배포용) |
| build_dataset_go2k_v7.py | Clean split 데이터셋 빌드 |

### 참고용 (과거 실험)

| 파일 | 용도 |
|------|------|
| train_go2k_v3~v7.py | 각 버전 학습 |
| train_quick_*.py | Phase A 하이퍼 실험 |
| train_v13.py ~ train_v15.py | 초기 버전 학습 |
| build_dataset_go2k_v3.py | 타일 분할 데이터셋 |
| eval_go2k_sahi_sweep.py | SAHI 파라미터 sweep |

### 정리 대상 (삭제 가능)

| 파일 | 이유 |
|------|------|
| detect_batch2.log, train_go2k_v3.log | 로그 파일 |
| detect_captures_3k.log | 로그 파일 |
| eval_comprehensive.py | eval_phase_b.py로 대체됨 |
| eval_fullimage_gate.py, eval_gate_finetune.py | gate 실험 (미사용) |
| eval_tile_size.py | 타일 크기 실험 (완료) |
| temp/ | 임시 파일 |

### 정리 대상 디렉토리 (디스크 절약)

| 디렉토리 | 크기 | 이유 |
|----------|------|------|
| hoban_v7, hoban_v10~v12 | 각 ~170MB | 4/3-클래스 구식 |
| hoban_quick_a1~a4 | 각 ~170MB | Phase A 실험 완료 |
| hoban_go500_* | 각 ~170MB | go2k로 대체됨 |
| hoban_go2k_v4, v6 | 각 ~170MB | v2/v5가 best |
| datasets_v7~v15 | 수 GB | 구식 데이터셋 |

## 모델 경로 (현역)

```
/home/lay/hoban/hoban_v13_stage2/weights/best.pt   # 범용 최고
/home/lay/hoban/hoban_go2k_v2/weights/best.pt       # CCTV 주력 (640px)
/home/lay/hoban/hoban_go2k_v3/weights/best.pt       # CCTV 보조 (1280px)
/home/lay/hoban/hoban_go2k_v5/weights/best.pt       # CCTV 보조 (AdamW)
/home/lay/hoban/hoban_go2k_v7/weights/best.pt       # Clean 학습 실험
```

## 데이터셋 경로 (현역)

```
/home/lay/hoban/datasets/go2k_manual/               # 평가 GT (604장, 1,680 bbox)
/home/lay/hoban/datasets_go2k_v2/                   # 주력 학습 데이터
/home/lay/hoban/datasets_go2k_v7/                   # Clean split 데이터
/home/lay/hoban/datasets_go2k_quick/                 # Quick 실험 데이터
/data/aihub_data/helmet_60k/                         # 원본 최고 품질 데이터
/data/helmet/                                        # 2-클래스 헬멧 (76K)
/data/unified_safety_all/                            # 8-클래스 통합 (493K)
```
