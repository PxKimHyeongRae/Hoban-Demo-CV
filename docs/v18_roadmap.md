# v18 로드맵: v17 배포 + Temporal Smoothing + Fine-tune + 데이터 수집

## 현재 상태

- **최고 모델**: v17 (SAHI F1=0.918, mAP50=0.961)
- **video_indoor**: go500_finetune 모델 사용 중 (구형)
- **후처리 한계**: 추론 시간 후처리만으로는 +0.9%p 한계 (F1 0.909→0.918)
- **병목**: 학습 데이터의 helmet_off 부족 + 배경 hard negative 부재

## 작업 순서

```
[서버] ① Temporal Smoothing + v17 배포 (video_indoor)
  ↓
[서버] ② 앙상블 테스트 (v17+v16+v13)
  ↓ (동시 진행)
[서버] ③ Fine-tune v18 준비 + 학습    ←──── [로컬] ④ 데이터 수집 (helmet_off + hard negative)
  ↓
[서버] ⑤ v18 평가 + video_indoor 재배포
```

---

## ① video_indoor에 v17 + Temporal Smoothing 적용

### 1-1. 모델 교체
- **현재**: `hoban_go500_finetune/weights/best.pt` (app.py line 108)
- **변경**: `hoban_go3k_v17/weights/best.pt`
- v17은 1280px 학습, SLICE_LEVEL=2 (1280x1280)와 호환

### 1-2. Temporal Smoothing 개선
**현재 구현**:
- 15프레임 연속 탐지 후 이벤트 발생
- 탐지 누락 시 5프레임 유지
- IoU 기반 tracking

**추가 개선**:

| 기능 | 설명 | 효과 |
|------|------|------|
| **Track-level confidence 누적** | 각 track_id의 최근 N프레임 confidence 평균 | 단발성 고conf 오탐 방지 |
| **클래스 안정성 검증** | 15프레임 중 12프레임 이상 같은 클래스여야 유효 | ON↔OFF 전환 오류 제거 |
| **Cross-class NMS** | 동일 인물 ON+OFF 동시 탐지 시 높은 conf만 유지 | 중복 이벤트 제거 |
| **Per-class confidence** | helmet_on≥0.40, helmet_off≥0.15 | 최적 임계값 적용 |

### 1-3. 수정 파일
- `/home/lay/video_indoor/app.py`

### 1-4. 검증
- 서버 재시작 → live 스트림에서 오탐 빈도 확인
- `/logs` API로 이벤트 로그 모니터링

---

## ② 앙상블 테스트

### 2-1. 사용 모델
| 모델 | 경로 | 특징 |
|------|------|------|
| v17 | `hoban_go3k_v17/weights/best.pt` | 1280px, COCO pt, F1=0.918 |
| v16 | `hoban_go3k_v16_640/weights/best.pt` | 640px, v13 pt, F1=0.885 |
| v13 | `hoban_v13_stage2/weights/best.pt` | curriculum learning |

### 2-2. 테스트 조합
```
v17+v16 NMS (IoU=0.4, 0.5)
v17+v13 NMS
v17+v16+v13 NMS
v17+v16 WBF (ensemble_boxes 설치 필요)
```

### 2-3. 구현
- `eval_v17_ensemble.py` (신규)
- 평가: 729장 combined set (3k val + verified helmet_off)
- 소요: ~15분

---

## ③ Fine-tune v18

### 3-1. 데이터 구성
| 소스 | 수량 | 용도 |
|------|------|------|
| 3k clean (기존) | 2,567 train + 644 val | 기본 학습 데이터 |
| CVAT verified helmet_off | 137장 | helmet_off 강화 (train에 추가) |
| ④ 추가 helmet_off | ~400장 (목표) | helmet_off 추가 강화 |
| ④ hard negatives | ~200~300장 (목표) | 배경 오탐 제거용 (빈 라벨) |

### 3-2. 학습 설정
- **시작 가중치**: v17 best.pt (transfer learning)
- **epochs**: 30~50 (짧은 fine-tune)
- **lr0**: 0.001 (v17의 1/5)
- **imgsz**: 1280
- **나머지**: v17과 동일

### 3-3. 파일
- `train_go3k_v18.py` (신규)
- `datasets_go3k_v18/data.yaml` (신규)

---

## ④ 로컬 데이터 수집 (helmet_off + hard negative 동시)

### 4-1. helmet_off 수집
**스크립트**: `extract_helmet_off_v17.py` (기존 수정)
- v17 모델 사용 (v16 대비 오탐 23% 감소)
- captures cam1+cam2에서 SAHI 추론
- 3k train/val + 이전 결과 제외
- target: 400장, 3분 간격 + burst 스킵
- CVAT 패키징 → 검수 후 train에 추가

### 4-2. hard negative 수집
**스크립트**: `extract_hard_negatives.py` (신규)
- v17 모델로 captures 전체 스캔
- 수집 기준: **사람 없는 이미지에서 오탐된 케이스**
  - 풀이미지 추론에서 탐지 없음 + SAHI에서 탐지 있음 (gate에 걸리는 케이스)
  - bbox 면적이 극히 작거나 비정상 종횡비
- target: 200~300장
- 빈 라벨(.txt)과 함께 저장
- CVAT 패키징 → 검수 → train에 빈 라벨로 추가

### 4-3. 실행 환경
- 로컬 PC (Z:/ 경로), `--server` 없이 실행
- GPU(CUDA) 또는 CPU

---

## 소요시간 예상

| 작업 | 소요 | 위치 | 병렬 |
|------|------|------|------|
| ① v17 배포 + temporal smoothing | 30분 | 서버 | - |
| ② 앙상블 테스트 | 15분 | 서버 | - |
| ③ v18 데이터 준비 | 30분 | 서버 | ④와 동시 |
| ③ v18 학습 (50ep) | 6~8시간 | 서버 GPU | ④와 동시 |
| ④ helmet_off 수집 | 1~2시간 | 로컬 | ③과 동시 |
| ④ hard negative 수집 | 1~2시간 | 로컬 | ③과 동시 |
| ④ CVAT 검수 | 수동 작업 | 로컬 | - |
| ⑤ v18 평가 + 재배포 | 30분 | 서버 | - |

**총 예상: 1~2일 (학습 시간 포함, ③④ 병렬 진행)**
