#!/usr/bin/env python3
"""
SAHI 기반 CCTV pseudo-label 생성기

- v13 stage2 best.pt + SAHI로 snapshot_detect 이미지에 자동 라벨링
- YOLO format 라벨 + 시각화 이미지 저장
- 수동 검수/보정용 데이터 생성

실행 (서버):
  python generate_pseudo_labels.py
  python generate_pseudo_labels.py --conf 0.15 --slice 640
"""

import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/lay/hoban/hoban_v13_stage2/weights/best.pt")
    parser.add_argument("--source", default="/home/lay/hoban/snapshot_detect")
    parser.add_argument("--output", default="/home/lay/hoban/datasets_cctv_pseudo")
    parser.add_argument("--conf", type=float, default=0.15, help="confidence threshold (낮을수록 많이 검출)")
    parser.add_argument("--slice", type=int, default=640, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    args = parser.parse_args()

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from PIL import Image, ImageDraw, ImageFont
    import shutil

    # 출력 디렉토리
    img_out = os.path.join(args.output, "images")
    lbl_out = os.path.join(args.output, "labels")
    vis_out = os.path.join(args.output, "visualize")

    for d in [img_out, lbl_out, vis_out]:
        os.makedirs(d, exist_ok=True)

    # 모델 로드
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.model,
        confidence_threshold=args.conf,
        device="cuda:0",
    )

    # 클래스 이름
    class_names = {0: "person_with_helmet", 1: "person_without_helmet"}

    # 이미지 목록
    src_images = sorted([
        f for f in os.listdir(args.source)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print("=" * 60)
    print("SAHI Pseudo-Label Generator")
    print(f"  Model: {args.model}")
    print(f"  Source: {args.source} ({len(src_images)} images)")
    print(f"  Output: {args.output}")
    print(f"  Conf: {args.conf}, Slice: {args.slice}, Overlap: {args.overlap}")
    print("=" * 60)

    stats = {"total_images": 0, "with_detection": 0, "total_boxes": 0, "cls": {0: 0, 1: 0}}

    for i, img_name in enumerate(src_images):
        img_path = os.path.join(args.source, img_name)
        stem = Path(img_name).stem

        # SAHI 추론
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=args.slice,
            slice_width=args.slice,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
        )

        img = Image.open(img_path)
        img_w, img_h = img.size
        stats["total_images"] += 1

        # 라벨 변환
        bboxes = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox  # BoundingBox object
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            cls = pred.category.id
            conf = pred.score.value

            # YOLO normalized format
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # clamp
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = min(w, 1.0)
            h = min(h, 1.0)

            if w > 0.005 and h > 0.005:
                bboxes.append((cls, cx, cy, w, h, conf))
                stats["cls"][cls] = stats["cls"].get(cls, 0) + 1

        stats["total_boxes"] += len(bboxes)
        if bboxes:
            stats["with_detection"] += 1

        # 이미지 복사 (심볼릭 링크)
        dst_img = os.path.join(img_out, img_name)
        if not os.path.exists(dst_img):
            os.symlink(img_path, dst_img)

        # YOLO 라벨 저장 (conf 없이)
        lbl_path = os.path.join(lbl_out, stem + ".txt")
        with open(lbl_path, 'w') as f:
            for cls, cx, cy, w, h, conf in bboxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # 시각화 저장
        draw = ImageDraw.Draw(img)
        colors = {0: (0, 255, 0), 1: (255, 0, 0)}  # green=helmet, red=no_helmet
        for cls, cx, cy, w, h, conf in bboxes:
            px1 = int((cx - w / 2) * img_w)
            py1 = int((cy - h / 2) * img_h)
            px2 = int((cx + w / 2) * img_w)
            py2 = int((cy + h / 2) * img_h)
            color = colors.get(cls, (255, 255, 0))
            draw.rectangle([px1, py1, px2, py2], outline=color, width=2)
            label = f"{class_names[cls]} {conf:.2f}"
            draw.text((px1, max(0, py1 - 12)), label, fill=color)
        img.save(os.path.join(vis_out, img_name), quality=90)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(src_images)} processed... ({stats['with_detection']} with detections)")

    # data.yaml 생성
    data_yaml = (
        f"path: {args.output}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: person_with_helmet\n"
        f"  1: person_without_helmet\n"
    )
    with open(os.path.join(args.output, "data.yaml"), 'w') as f:
        f.write(data_yaml)

    print(f"\n{'='*60}")
    print("Pseudo-Label Generation Complete")
    print(f"  Total images: {stats['total_images']}")
    print(f"  With detections: {stats['with_detection']} ({stats['with_detection']/max(1,stats['total_images'])*100:.1f}%)")
    print(f"  Total boxes: {stats['total_boxes']}")
    print(f"  helmet_o: {stats['cls'][0]}, helmet_x: {stats['cls'][1]}")
    print(f"\n  Labels: {lbl_out}/")
    print(f"  Visualize: {vis_out}/  ← 여기서 수동 검수")
    print(f"  data.yaml: {os.path.join(args.output, 'data.yaml')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
