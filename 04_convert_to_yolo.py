# -*- coding: utf-8 -*-
"""
04. YOLO 학습 데이터로 변환
이미지와 라벨 데이터를 학습시킬 수 있도록 변환 및 분류
- 이미지 데이터 : 압축 파일에서 추출된 이미지 파일을 분류 (train/val)
- 라벨 데이터 : COCO JSON를 YOLO txt로 변환 후 분류(train/val)

입력 : selected.csv      (02의 서브샘플링 및 분류(train/val) 목록)
       work/staging      (03이 서브샘플링한 이미지)
       라벨 json          (GT 라벨 데이터)
출력 : dataset/images/{train,val}/*.jpg
       dataset/labels/{train,val}/*.txt   (정상은 빈 txt = 네거티브)

변환 : bbox [x, y, w, h] 픽셀절대 -> YOLO 정규화 [cls cx cy w h]
       이미지는 '복사' (원본·추출본 이동/삭제 안 함)
"""
import sys, csv, json, shutil
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import config as C


def to_yolo(json_path):
    """라벨 json -> YOLO 라벨 줄 리스트 (정상/없으면 빈 리스트)."""
    clamp = lambda v: min(max(v, 0.0), 1.0)          # 0~1 범위로 자름
    try:
        j = json.load(open(json_path, encoding="utf-8"))
        W, H = j["image"]["width"], j["image"]["height"]
    except (OSError, KeyError, json.JSONDecodeError):
        return []
    out = []
    for a in j.get("annotations", []):
        cid = C.CLASS_MAP.get(a["categories_id"])     # fl->0, sm->1, none->제외(빈 라벨)
        if cid is None:
            continue
        x, y, w, h = a["bbox"]                        # 픽셀 절대좌표 [좌상x, 좌상y, 폭, 높이]
        out.append(f"{cid} {clamp((x + w/2)/W):.6f} {clamp((y + h/2)/H):.6f} "
                   f"{clamp(w/W):.6f} {clamp(h/H):.6f}")   # 중심좌표·크기를 0~1로 정규화
    return out


def main():
    # 03이 저장한 '압축 내부 경로 prefix'. PowerShell이 붙였을 수 있는 BOM은 utf-8-sig가 제거
    prefix = C.ARCHIVE_PREFIX.read_text(encoding="utf-8-sig").strip() \
             if C.ARCHIVE_PREFIX.exists() else C.ARCHIVE_PREFIX_CANDIDATES[0]

    # 출력 폴더 4개 생성: images·labels × train·val
    for sp in ("train", "val"):
        (C.DATASET / "images" / sp).mkdir(parents=True, exist_ok=True)
        (C.DATASET / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # 이미지 출처: 03 추출본(staging) 우선, 없으면 이미 풀려있던 TS 폴더에서
    use_staging = C.STAGING.exists()
    print(f"이미지 소스: {'staging(추출본)' if use_staging else 'TS(이미 풀린 분량)'}")

    ok = miss = boxes = 0                              # 변환성공 / 이미지누락 / 총 박스 수
    # selected.csv 한 줄 = 이미지 1장 (split·stem·img_rel·json_path 포함)
    with open(C.SELECTED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp, stem = r["split"], r["stem"]           # sp: train/val,  stem: 확장자 뗀 파일이름

            # 원본 이미지 위치 (staging 안 prefix/상대경로, 또는 TS 폴더)
            src = (C.STAGING / prefix / r["img_rel"]) if use_staging \
                  else (C.SRC_EXTRACTED / "화재현상" / "이미지" / r["img_rel"])
            if not src.exists():                       # 추출 안 된 이미지면 건너뜀
                miss += 1
                continue

            # 이미지를 split 폴더로 복사 (이름을 stem.jpg 로 평탄화, copy2 = 원본 보존)
            shutil.copy2(src, C.DATASET / "images" / sp / f"{stem}.jpg")

            # 라벨 json -> YOLO 줄 변환 후 같은 이름.txt 로 저장 (정상은 빈 파일 = 네거티브)
            lines = to_yolo(r["json_path"])
            (C.DATASET / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            ok += 1
            boxes += len(lines)
            if ok % 5000 == 0:                         # 5천 장마다 진행 표시
                print(f"  ...{ok:,} 변환", flush=True)

    # 요약 (누락이 많으면 03 추출/ prefix 문제)
    print(f"\n[완료] 변환 {ok:,}장, 박스 {boxes:,}개, 누락 {miss:,}장")
    if miss:
        print("  누락 많으면 03 추출 부족 또는 prefix 불일치 -> archive_prefix.txt 확인")
    print(f"dataset: {C.DATASET} -> 다음: python model.py train --models yolov8m")


if __name__ == "__main__":
    main()
