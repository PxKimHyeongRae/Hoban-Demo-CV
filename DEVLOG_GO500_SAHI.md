# GO500 + SAHI 방향 정리

> 2026-02-13 작성

## 배경

현장 CCTV 영상에서 기존 모델(v13 stage2)의 소형 객체 탐지 성능이 부족한 문제를 해결하기 위해,
실제 CCTV 스냅샷 500장(go500)으로 fine-tune한 모델과 SAHI를 조합하는 방향으로 진행.

## GO500 데이터셋

- **소스**: `/home/lay/hoban/snapshot_detect/` (현장 CCTV 스냅샷 1815장)에서 수동 라벨링 500장
- **클래스**: 2-class (`person_with_helmet`=0, `person_without_helmet`=1)
- **학습 데이터**: `/home/lay/hoban/datasets/` (go500 YOLO format)

## 모델 학습 결과

### Fine-tune (v13 stage2 → go500)

| 항목 | 값 |
|------|-----|
| 베이스 모델 | v13 stage2 best.pt |
| 학습률 | lr0=0.0002 |
| Best epoch | 76/100 |
| **mAP50** | **0.748** |
| **mAP50-95** | **0.465** |
| Precision | 0.637 |
| Recall | 0.702 |

- 경로: `/home/lay/hoban/hoban_go500_finetune/weights/best.pt`

### From-scratch (yolo26m.pt → go500)

| 항목 | 값 |
|------|-----|
| 베이스 모델 | yolo26m.pt (COCO pretrained) |
| 학습률 | lr0=0.001 |
| Best epoch | 33/100 (early stop 66) |
| **mAP50** | **0.768** |
| **mAP50-95** | **0.386** |
| Precision | 0.616 |
| Recall | 0.864 |

- 경로: `/home/lay/hoban/hoban_go500_scratch/weights/best.pt`

### 비교

| 지표 | Fine-tune | Scratch |
|------|-----------|---------|
| mAP50 | 0.748 | **0.768** |
| mAP50-95 | **0.465** | 0.386 |
| Precision | **0.637** | 0.616 |
| Recall | 0.702 | **0.864** |

- Fine-tune이 mAP50-95와 Precision에서 우세 → **실제 배포용으로 fine-tune 채택**
- Scratch는 Recall이 높지만 FP도 많음

## SAHI 적용 결과 (오프라인)

go500 fine-tune 모델 + SAHI로 CCTV 스냅샷 1815장 pseudo-labeling:

| 항목 | SAHI + go500 fine-tune | 기존 v13 모델 |
|------|----------------------|--------------|
| 탐지 이미지 | **1700/1815 (93.7%)** | 703/1815 (38.7%) |
| 총 bbox | **5375** | 960 |
| helmet_o | 4739 | - |
| helmet_x | 636 | - |

- **탐지율 2.4배**, **bbox 수 5.6배 증가**
- SAHI 설정: slice=640x640, overlap=0.2, conf=0.15

## CVAT 업로드 데이터

### 이미지 (4분할)

| 파일 | 이미지 | 크기 |
|------|--------|------|
| `cvat_go500_images_part1_1-500.zip` | 500장 | 482MB |
| `cvat_go500_images_part2_501-1000.zip` | 500장 | 448MB |
| `cvat_go500_images_part3_1001-1500.zip` | 500장 | 448MB |
| `cvat_go500_images_part4_1501-1815.zip` | 315장 | 283MB |

### 어노테이션 (YOLO 1.1 형식, 4분할)

| 파일 | 라벨 | bbox | 크기 |
|------|------|------|------|
| `cvat_go500_part1_1-500.zip` | 498 | 1619 | 151KB |
| `cvat_go500_part2_501-1000.zip` | 459 | 1050 | 143KB |
| `cvat_go500_part3_1001-1500.zip` | 480 | 2037 | 157KB |
| `cvat_go500_part4_1501-1815.zip` | 263 | 669 | 89KB |

**업로드 순서**: Task 생성 → 이미지 zip 업로드 → YOLO 1.1 어노테이션 import

## SAHI 실시간 적용 (video_indoor)

### 시도 결과: 실패

2-카메라 RTSP 실시간 스트리밍에 SAHI 직접 적용 시도 → **30초 RTSP 타임아웃 발생**

- 원인: 640x640 슬라이스 ~12개 × 2카메라 = 프레임당 ~24회 추론, GPU 경합
- per-camera 모델 + CUDA 워밍업도 해결 안 됨

### 권장 대안

1. **N프레임마다 SAHI** - 일반 프레임은 기존 YOLO, 매 N번째만 SAHI
2. **고해상도 직접 추론** - `imgsz=1280`으로 YOLO 직접 추론 (SAHI보다 빠름)
3. **비동기 SAHI** - 별도 스레드에서 SAHI, 결과를 메인 스트림에 오버레이
4. **슬라이스 수 감소** - slice 크기 증가 or overlap 감소

### video_indoor 현재 상태

- 모델: `hoban_v12/weights/best.pt` (3-class) - 원본 유지
- `pluxmtx_adapter.py`: RTSP TCP 전송 모드 추가 (mediamtx 재시작 후 UDP 불안정 대응)
- 서버 정상 가동 중

## 다음 단계

1. **CVAT에서 pseudo-label 검수/수정** → 고품질 CCTV 학습 데이터 확보
2. **검수 완료 데이터로 go500 v2 모델 학습** → 정확도 추가 향상
3. **video_indoor SAHI 실시간 적용** → 위 권장 대안 중 택1
4. **클래스 매핑 통일** → go500(2-class) vs hoban_v12(3-class) 정리 필요
