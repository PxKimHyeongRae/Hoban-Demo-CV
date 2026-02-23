#!/usr/bin/env python3
"""CVAT XML 검수 결과 → val 라벨 업데이트

CVAT에서 검수 완료된 annotations.xml을 파싱하여:
1. CVAT XML → YOLO format 변환
2. 원본 val 라벨 백업
3. 검수된 56장의 라벨만 교체
4. 변경 통계 출력

사용법:
  python apply_cvat_gt_fix.py                    # 변환 + 적용
  python apply_cvat_gt_fix.py --dry-run          # 미리보기 (변경 안 함)
"""
import argparse
import os
import shutil
import xml.etree.ElementTree as ET

HOBAN = "/home/lay/hoban"
CVAT_XML = f"{HOBAN}/cvat_gt_fix/cvat_gt_fix_manual/annotations.xml"

# val 라벨 경로 (원본)
VAL_LBL = f"{HOBAN}/datasets/3k_finetune/val/labels"
BACKUP_DIR = f"{HOBAN}/cvat_gt_fix/val_labels_backup"

CLASS_MAP = {"person_with_helmet": 0, "person_without_helmet": 1}


def parse_cvat_xml(xml_path):
    """CVAT XML → {filename: [(cls, cx, cy, w, h), ...]}"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {}
    for img in root.findall("image"):
        fname = img.get("name")
        img_w = float(img.get("width"))
        img_h = float(img.get("height"))

        boxes = []
        for box in img.findall("box"):
            label = box.get("label")
            cls = CLASS_MAP.get(label)
            if cls is None:
                continue

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            # pixel → YOLO normalized
            cx = ((xtl + xbr) / 2) / img_w
            cy = ((ytl + ybr) / 2) / img_h
            w = (xbr - xtl) / img_w
            h = (ybr - ytl) / img_h

            boxes.append((cls, cx, cy, w, h))

        result[fname] = boxes

    return result


def load_yolo_labels(lbl_path):
    """YOLO 라벨 파일 읽기"""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4])))
    return boxes


def boxes_to_yolo(boxes):
    """YOLO format string"""
    lines = []
    for cls, cx, cy, w, h in boxes:
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CVAT GT fix → val labels update")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    print("=" * 60)
    print("  CVAT GT Fix → Val Labels Update")
    print("=" * 60)

    # 1. Parse CVAT XML
    print(f"\nCVAT XML: {CVAT_XML}")
    cvat_labels = parse_cvat_xml(CVAT_XML)
    print(f"  Images: {len(cvat_labels)}")
    total_new = sum(len(v) for v in cvat_labels.values())
    print(f"  Total boxes: {total_new}")

    # 2. Compare with original
    added, removed, changed, unchanged = 0, 0, 0, 0
    details = []

    for fname, new_boxes in sorted(cvat_labels.items()):
        lbl_name = fname.replace(".jpg", ".txt")
        lbl_path = os.path.join(VAL_LBL, lbl_name)
        old_boxes = load_yolo_labels(lbl_path)

        old_count = len(old_boxes)
        new_count = len(new_boxes)
        diff = new_count - old_count

        if diff > 0:
            added += diff
            details.append(f"  + {fname}: {old_count} → {new_count} (+{diff})")
        elif diff < 0:
            removed += abs(diff)
            details.append(f"  - {fname}: {old_count} → {new_count} ({diff})")
        elif old_count == new_count:
            # Check if content changed
            old_str = boxes_to_yolo(old_boxes)
            new_str = boxes_to_yolo(new_boxes)
            if old_str != new_str:
                changed += 1
                details.append(f"  ~ {fname}: {old_count} boxes (modified)")
            else:
                unchanged += 1

    print(f"\n  Changes:")
    print(f"    Boxes added:    +{added}")
    print(f"    Boxes removed:  -{removed}")
    print(f"    Files modified: {changed}")
    print(f"    Files unchanged: {unchanged}")

    if details:
        print(f"\n  Details:")
        for d in details:
            print(d)

    if args.dry_run:
        print(f"\n  [DRY RUN] No changes applied.")
        return

    # 3. Backup original labels
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        backed = 0
        for fname in cvat_labels:
            lbl_name = fname.replace(".jpg", ".txt")
            src = os.path.join(VAL_LBL, lbl_name)
            if os.path.exists(src):
                # resolve symlink
                real_src = os.path.realpath(src)
                shutil.copy2(real_src, os.path.join(BACKUP_DIR, lbl_name))
                backed += 1
        print(f"\n  Backup: {backed} files → {BACKUP_DIR}")
    else:
        print(f"\n  Backup already exists: {BACKUP_DIR}")

    # 4. Apply new labels
    applied = 0
    for fname, new_boxes in sorted(cvat_labels.items()):
        lbl_name = fname.replace(".jpg", ".txt")
        lbl_path = os.path.join(VAL_LBL, lbl_name)

        # Remove symlink if exists, write real file
        if os.path.islink(lbl_path):
            os.unlink(lbl_path)

        with open(lbl_path, "w") as f:
            f.write(boxes_to_yolo(new_boxes) + "\n")
        applied += 1

    print(f"  Applied: {applied} label files updated")

    print(f"\n{'='*60}")
    print(f"  Done! Val labels updated with CVAT review.")
    print(f"  Backup: {BACKUP_DIR}")
    print(f"  To revert: copy backup files back to {VAL_LBL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
