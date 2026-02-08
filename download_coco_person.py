"""
COCO 2017에서 person 클래스 이미지 3만장 다운로드
1. annotations 다운로드
2. person 이미지 필터링 (category_id=1)
3. 이미지 병렬 다운로드
4. YOLO 포맷 변환 (person → class 2)
"""
import json
import os
import zipfile
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

OUT = Path(r"D:\task\hoban\coco_person")
ANNO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNO_ZIP = OUT / "annotations_trainval2017.zip"
MAX_IMAGES = 30000
WORKERS = 16

def download_file(url, dest):
    if dest.exists():
        print(f"  이미 존재: {dest.name}")
        return
    print(f"  다운로드 중: {url}")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  완료: {dest.name}")

def download_image(args):
    url, save_path = args
    try:
        if save_path.exists():
            return True
        urllib.request.urlretrieve(url, str(save_path))
        return True
    except:
        return False

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    (OUT / "labels").mkdir(exist_ok=True)

    # 1. 어노테이션 다운로드
    print("=== 1. COCO 어노테이션 다운로드 ===")
    download_file(ANNO_URL, ANNO_ZIP)

    # 2. 압축 해제
    anno_json = OUT / "annotations" / "instances_train2017.json"
    if not anno_json.exists():
        print("  압축 해제 중...")
        with zipfile.ZipFile(str(ANNO_ZIP), 'r') as z:
            z.extract("annotations/instances_train2017.json", str(OUT))
        print("  완료")
    else:
        print("  이미 해제됨")

    # 3. person 어노테이션 파싱
    print("\n=== 2. person 어노테이션 파싱 ===")
    with open(anno_json, 'r') as f:
        coco = json.load(f)

    # person category_id = 1
    person_cat_id = 1

    # 이미지 정보 인덱스
    img_info = {img['id']: img for img in coco['images']}

    # person bbox를 이미지별로 그룹화
    img_annos = defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id'] == person_cat_id and not ann.get('iscrowd', 0):
            img_annos[ann['image_id']].append(ann)

    print(f"  person 이미지: {len(img_annos)}개")

    # bbox 수 기준 정렬 (bbox 많은 이미지 우선 → 더 유용)
    sorted_imgs = sorted(img_annos.keys(), key=lambda x: len(img_annos[x]), reverse=True)
    selected = sorted_imgs[:MAX_IMAGES]
    print(f"  선택: {len(selected)}개")

    total_bbox = sum(len(img_annos[i]) for i in selected)
    print(f"  총 person bbox: {total_bbox}개")

    # 4. YOLO 라벨 생성
    print("\n=== 3. YOLO 라벨 생성 ===")
    download_list = []
    label_count = 0

    for img_id in selected:
        info = img_info[img_id]
        img_w = info['width']
        img_h = info['height']
        file_name = info['file_name']

        yolo_lines = []
        for ann in img_annos[img_id]:
            x, y, w, h = ann['bbox']  # COCO: x_min, y_min, width, height (absolute)

            # 유효성 검사
            if w <= 0 or h <= 0:
                continue

            # YOLO 변환: center_x, center_y, width, height (normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # 범위 클리핑
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            # 너무 작은 bbox 제외
            if nw * nh < 0.0005:
                continue

            yolo_lines.append(f"2 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not yolo_lines:
            continue

        stem = Path(file_name).stem
        label_path = OUT / "labels" / f"{stem}.txt"
        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_lines) + "\n")
        label_count += 1

        img_url = f"http://images.cocodataset.org/train2017/{file_name}"
        img_path = OUT / "images" / file_name
        download_list.append((img_url, img_path))

    print(f"  라벨 생성: {label_count}개")

    # 5. 이미지 병렬 다운로드
    already = sum(1 for _, p in download_list if p.exists())
    to_download = [(u, p) for u, p in download_list if not p.exists()]
    print(f"\n=== 4. 이미지 다운로드 ===")
    print(f"  이미 존재: {already}개, 다운로드 필요: {len(to_download)}개")

    if to_download:
        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(download_image, args): args for args in to_download}
            for future in as_completed(futures):
                if future.result():
                    done += 1
                else:
                    failed += 1
                total = done + failed
                if total % 500 == 0:
                    print(f"  진행: {total}/{len(to_download)} (성공={done}, 실패={failed})")

        print(f"  완료: 성공={done}, 실패={failed}")

    # 6. 최종 확인
    final_imgs = len(list((OUT / "images").glob("*.jpg")))
    final_lbls = len(list((OUT / "labels").glob("*.txt")))
    print(f"\n=== 최종 결과 ===")
    print(f"  이미지: {final_imgs}개")
    print(f"  라벨: {final_lbls}개")
    print("done!")

if __name__ == "__main__":
    main()
