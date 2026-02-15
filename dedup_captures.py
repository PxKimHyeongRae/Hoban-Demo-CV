#!/usr/bin/env python3
"""
captures 원본 폴더에서 프리즈 중복(연속 동일 프레임) 제거
연속 파일 크기 비교 → md5 확인 → 첫 장만 유지, 나머지 삭제

실행: python dedup_captures.py [--dry-run]
"""
import os
import hashlib
import argparse
import time

CAP_DIR = "/home/lay/video_indoor/static/captures"
CAMS = ["cam1", "cam2"]

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="삭제하지 않고 목록만 출력")
args = parser.parse_args()


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


total_removed = 0
total_saved_bytes = 0

for cam in CAMS:
    cam_dir = os.path.join(CAP_DIR, cam)
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".jpg"))
    print(f"\n{'='*60}")
    print(f"{cam}: {len(files)}장 스캔")

    prev_size = None
    prev_md5 = None
    prev_path = None
    dup_count = 0
    dup_start = None
    removed = 0
    saved_bytes = 0

    for i, fname in enumerate(files):
        path = os.path.join(cam_dir, fname)
        size = os.path.getsize(path)

        if size == prev_size and prev_path:
            # 크기 같으면 md5 비교
            md5 = file_md5(path)
            if prev_md5 is None:
                prev_md5 = file_md5(prev_path)

            if md5 == prev_md5:
                # 중복! 삭제 대상
                if dup_count == 0:
                    dup_start = fname
                dup_count += 1

                if not args.dry_run:
                    os.remove(path)
                removed += 1
                saved_bytes += size
                # prev 유지 (같은 해시)
                continue
            else:
                # 크기는 같지만 내용 다름
                if dup_count > 0:
                    print(f"  프리즈: {dup_start} ~ {files[i-1]} ({dup_count}장 중복)")
                    dup_count = 0
                prev_md5 = md5
        else:
            if dup_count > 0:
                print(f"  프리즈: {dup_start} ~ {files[i-1]} ({dup_count}장 중복)")
                dup_count = 0
            prev_md5 = None

        prev_size = size
        prev_path = path

        if (i + 1) % 20000 == 0:
            print(f"  진행: {i+1}/{len(files)}, 중복 {removed}장...")

    if dup_count > 0:
        print(f"  프리즈: {dup_start} ~ {files[-1]} ({dup_count}장 중복)")

    total_removed += removed
    total_saved_bytes += saved_bytes
    print(f"{cam}: {removed}장 {'삭제 예정' if args.dry_run else '삭제'} ({saved_bytes/1024/1024:.1f}MB)")

print(f"\n{'='*60}")
action = "삭제 예정" if args.dry_run else "삭제 완료"
print(f"총 {action}: {total_removed}장 ({total_saved_bytes/1024/1024/1024:.2f}GB)")

# 삭제 후 남은 수
if not args.dry_run:
    for cam in CAMS:
        cam_dir = os.path.join(CAP_DIR, cam)
        remaining = len([f for f in os.listdir(cam_dir) if f.endswith(".jpg")])
        print(f"  {cam} 남은 수: {remaining}장")
