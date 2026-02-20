<!-- OMC:START -->
<!-- OMC:END -->

# Hoban Project - CLAUDE.md

## Project Identity

YOLO v8/v26 기반 건설현장 안전 감지 시스템.
- **모델**: yolo26m.pt (YOLO26 Medium, 21.8M params)
- **클래스**: 2개 - person_with_helmet(0), person_without_helmet(1)
- **서버**: RTX 4080 16GB Linux (lay@pluxity), conda env: `llm`
- **로컬**: Windows RTX 5070 Ti 16GB, conda env: `gostock`
- **배포**: `/home/lay/video_indoor/app.py` (Flask + SocketIO 실시간 감지)

## Current Best Model

- **v19**: SAHI F1=0.928, mAP50=0.960 (v17 ft + helmet_off + neg)
- **v17 (배포중)**: SAHI F1=0.918, mAP50=0.958 (COCO pt + 1280px)
- 가중치: `hoban_go3k_v17/weights/best.pt`
- 후처리: cross_nms(IoU=0.3) → min_area(5e-5) → gate(conf=0.20, r=30) → per_class(ON≥0.40, OFF≥0.15)

## MUST Rules (반드시 지켜야 할 것)

### Server Safety
- **NEVER** `torch.cuda.set_per_process_memory_fraction()` - CUDA forward compatibility error 발생
- **NEVER** `conda run --no-banner` - 지원 안 됨, stdout 버퍼링 문제
- GPU 메모리 16GB 한계: 1280px에서 batch=6이 최대

### Data Integrity
- **NEVER** train/val 데이터 섞기 - data leakage는 F1을 4%p 이상 부풀림 (v2-v6에서 확인됨)
- 데이터셋 빌드 시 반드시 기존 val 이미지 제외 목록 확인
- 새 데이터 추가 시 `datasets_go3k_v16/valid/images`와 중복 체크 필수

### Training
- **SGD 사용** (AdamW는 장기 학습 시 overfitting collapse 발생)
- seed=42 고정 (재현성)
- close_mosaic=10~15 (마지막 N epoch에서 mosaic 해제)
- patience=15~20 (early stopping)

### Evaluation
- **SAHI 필수**: CCTV 소형 객체는 SAHI 없이 절대 정확하게 평가 불가
- SAHI 설정: 1280x720 tiles, overlap=0.15, NMS/IOS, perform_standard_pred=True
- **반드시 per-class 메트릭 보고** (helmet_on과 helmet_off 특성이 다름)
- 평가셋: 3k val 641장 + verified helmet_off 88장 = 729장 combined

### Code Style
- 학습 스크립트 네이밍: `train_<dataset>_<version>.py`
- 평가 스크립트 네이밍: `eval_<dataset_or_version>_<variant>.py`
- 모델 출력 디렉터리: `hoban_<dataset>_<version>/weights/{best,last}.pt`
- 데이터셋 디렉터리: `datasets_<name>/` with `{train,valid}/{images,labels}/`
- data.yaml: `path`, `train`, `val`, `nc`, `names` 필수

## MUST NOT Rules (절대 하지 말 것)

- eval 없이 모델 배포하지 않기
- val set에 있는 이미지를 train에 넣지 않기
- 앙상블에 기대하지 않기 (v17+v16+v13 NMS/WBF 모두 개선 없음 확인됨)
- v13 모델을 SAHI에 사용하지 않기 (탐지 수 극히 적음: 479 vs v17 1746)
- 후처리 파라미터를 추론 시간에 튜닝하지 않기 (offline eval에서 미리 최적화)

## Key Paths

```
# 프로젝트 루트
/home/lay/hoban/

# 현재 최고 모델
/home/lay/hoban/hoban_go3k_v17/weights/best.pt

# 학습 데이터 (v16 base = 10,564 train / 641 val)
/home/lay/hoban/datasets_go3k_v16/data.yaml

# 평가셋 (GT)
/home/lay/hoban/datasets/3k_finetune/val/{images,labels}/          # 641장
/home/lay/hoban/datasets/cvat_helmet_off/valid_helmet_off_137/     # 88장 extra

# 배포 앱
/home/lay/video_indoor/app.py

# 로컬 Windows 접근
Z:/home/lay/hoban/
Z:/home/lay/video_indoor/
```

## Training Hyperparameters (Standard)

```python
# v17 기준 (현재 최적 설정)
optimizer="SGD", lr0=0.005, lrf=0.01, momentum=0.937
warmup_epochs=3.0, weight_decay=0.0005, cos_lr=True
imgsz=1280, batch=6, device="0"

# Augmentation
mosaic=1.0, mixup=0.1, copy_paste=0.15
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
scale=0.5, translate=0.1, degrees=5.0
fliplr=0.5, erasing=0.15, close_mosaic=10

# Fine-tune (v18)
lr0=0.001 (1/5), warmup_epochs=1.0, patience=15
```

## Post-Processing Pipeline (Production)

```python
# 1. Cross-class NMS: ON+OFF 동시 탐지 시 높은 conf 유지
cross_class_nms(iou_threshold=0.3)

# 2. Min area filter: 너무 작은 bbox 제거
min_area >= 5e-05 (이미지 면적 대비)

# 3. Full-image Gate: SAHI에서만 탐지되고 전체 이미지에서 안 보이면 제거
gate_conf=0.20, radius=30px

# 4. Per-class confidence
helmet_on >= 0.40, helmet_off >= 0.15
```

## video_indoor Deployment Config

```python
MODEL_PATH = "/home/lay/hoban/hoban_go3k_v17/weights/best.pt"
CLASS_CONFIDENCE_THRESHOLD = {0: 0.40, 1: 0.15}
CONSECUTIVE_FRAMES_REQUIRED = 15   # 이벤트 트리거 전 연속 프레임
DETECTION_HOLD_FRAMES = 5          # 누락 시 유지 프레임
TRACK_MAX_AGE = 60                 # 미감지 시 트랙 삭제 (2초)
BBOX_IOU_THRESHOLD = 0.3          # IoU 트래커 임계값
# 클래스 안정성: 15프레임 중 12프레임 이상 같은 클래스여야 이벤트 발생
```

## Version History (Key Milestones)

| Version | F1 (SAHI) | Key Change | Date |
|---------|-----------|------------|------|
| v9 | - | 4-class, mAP50=0.940, overfitting collapse | Feb 8 |
| v13 | - | 2-class curriculum learning, mAP50=0.945 | Feb 13 |
| v16 | 0.885 | Clean 3k dataset, CCTV only | Feb 17 |
| v17 | 0.918 | COCO pt + 1280px (+3.4%p) | Feb 18 |
| v19 | **0.928** | v17 ft + helmet_off + neg 10,852 | Feb 19 |
| v20 | 0.915 | v17 ft + 12,470 (데이터↑ but F1↓) | Feb 20 |
| v21-l | TBD | yolo26l COCO pt (26.3M params) | Feb 20 |

## Critical Learnings

1. **Data quality > quantity**: 필터링만으로 +10.2%p mAP50-95
2. **Data leakage 치명적**: train/eval 겹침 79.3% → F1 4.1%p 부풀림
3. **SAHI 필수**: tile 1280x720으로 F1 +10.8%p (0.804→0.912)
4. **Gate 가장 효과적**: 후처리 중 FP -40개 감소 (+0.8%p)
5. **앙상블 효과 없음**: v17 단독이 모든 조합보다 동등 이상
6. **COCO pretraining**: domain-specific보다 COCO base가 +2%p 우수
7. **bbox area**: 640px에서 면적 <0.5%는 탐지 불가
8. **오류 94% tiny**: FP/FN 모두 <0.1% 면적 (~30px) 소형 객체에 집중
9. **후처리 천장**: min_area/conf/SAHI타일 sweep 모두 F1=0.927 이상 불가
10. **데이터↑ ≠ 성능↑**: v20(12,470)이 v19(10,852)보다 SAHI F1 하락

## Script Quick Reference

```bash
# 학습
python train_go3k_v17.py                    # v17 학습 (yolo26m COCO pt)
python train_go3k_v19.py --prepare          # v19 데이터셋 준비
python train_go3k_v19.py                    # v19 학습 (v17 ft)
python train_go3k_v21_l.py                  # v21-l 학습 (yolo26l COCO pt)
python train_go3k_v21_l.py --resume         # 이어서 학습

# 평가
python eval_go3k_v18.py                     # SAHI F1 평가
python eval_go3k_v18.py --model path.pt     # 다른 모델 평가
python eval_v17_postprocess.py              # 7-Phase 후처리 실험

# 분석
python analysis_v19_errors.py               # FP/FN 오류 시각화
python exp_min_area_sweep.py                # min_area + SAHI 타일 sweep
python exp_tiny_gt_verify.py                # tiny GT 크롭 검증

# 데이터 수집 (로컬)
python extract_data_v17.py                  # helmet_off + neg 수집
python extract_data_v17.py --mode helmet_off --server  # 서버에서 helmet_off만
```
