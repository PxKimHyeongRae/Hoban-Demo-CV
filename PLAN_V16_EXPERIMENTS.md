# v16 실험 계획: F1 0.885 돌파를 위한 체계적 탐색

## 현재 상태
- v16 baseline: F1=0.885 (640px, v13_s2 pretrained, 3k+v13 8K, 100ep)
- Eval: 3k val 641장, 1,072 bbox (clean)
- SAHI: 1280x720, NMS/IOS@0.3, c0=0.45/c1=0.40

## 실험 전략

### Quick Train (20 epochs) → 후보 선별 → Full Train (100 epochs)
- 20 epoch 학습으로 상대적 순위 파악 (~10-20분/run)
- 상위 2-3개만 100 epoch full training
- Eval은 항상 전체 641장 (SAHI, ~1.5분)

---

## Phase A: 학습 변수 실험 (Quick 20ep)

| # | 실험명 | 변경점 | 가설 |
|---|--------|--------|------|
| A1 | no_v13 | v13 제거, CCTV 2,564장만 | 도메인 일치 극대화 |
| A2 | v13_4k | v13 4K로 축소 (CCTV 39%) | 도메인 균형 개선 |
| A3 | v13_filtered | v13 bbox area 필터 (소형만) | v13 중 CCTV와 유사한 것만 |
| A4 | imgsz_1280 | 1280px 학습 | 소형 객체 해상도 강화 |
| A5 | coco_pretrained | COCO pt에서 시작 | v13_s2 편향 제거 |
| A6 | heavy_aug | copy_paste=0.3, mixup=0.2, erasing=0.2 | 더 강한 증강 |
| A7 | light_aug | mosaic=0.5, mixup=0, copy_paste=0 | 과증강 방지 |

## Phase B: 앙상블 실험 (학습 불필요)

| # | 실험명 | 조합 | 가설 |
|---|--------|------|------|
| B1 | v16+v2 | WBF | v2는 다른 데이터로 학습 → 다양성 |
| B2 | v16+v3 | WBF | v3는 1280px 학습 → 보완적 |
| B3 | v16+v5 | WBF | v5는 가장 많은 3k 데이터 사용 |
| B4 | v16+v2+v3 | WBF | 3모델 앙상블 |
| B5 | v16+best_A | WBF | Phase A 최고 모델과 앙상블 |

## Phase C: Full Training (100ep)

- Phase A+B에서 F1 상위 2-3개 설정으로 100 epoch 학습
- 최종 SAHI 파인튜닝 포함

---

## 예상 소요 시간

| Phase | 항목 수 | 시간/항목 | 총 예상 |
|-------|---------|----------|---------|
| A (quick train + eval) | 7 | ~15분 | ~2시간 |
| B (eval only) | 5 | ~3분 | ~15분 |
| C (full train) | 2-3 | ~4시간 | ~8-12시간 |

**Phase A+B: ~2.5시간으로 방향 확정**
