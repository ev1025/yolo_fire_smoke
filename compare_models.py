# -*- coding: utf-8 -*-
"""
Phase 2/3. 학습된 모델들 + 기존 baseline(YOLOv8m)을 한 표로 비교.

측정 항목 (모델당)
- 정확도 : mAP@50, mAP@50-95   (각 best.pt를 동일 val·동일 imgsz로 재평가 -> 공정 비교)
- 속도   : 추론 ms/img, FPS    (워밍업 후 단일 이미지 N회 측정)
- 크기   : 파라미터 수(M), 가중치 파일 MB

입력 : DATA(val 포함), 각 모델 best.pt
출력 : model_comparison.csv  + 콘솔 마크다운 표
실행 : python compare_models.py
"""
import csv, time, glob
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# ================= 설정 =================
DATA    = "data.yaml"
PROJECT = "runs"
IMGSZ   = 640                 # 비교용 추론 해상도 (train_all_models.py와 동일하게)
DEVICE  = 0                   # 측정용 단일 GPU (없으면 "cpu")
WARMUP, RUNS = 10, 50

# 비교 대상: 표시이름 -> best.pt 경로
CANDIDATES = {
    "YOLOv8n":      f"{PROJECT}/yolov8n/weights/best.pt",
    "YOLOv8s":      f"{PROJECT}/yolov8s/weights/best.pt",
    "YOLOv5n(u)":   f"{PROJECT}/yolov5nu/weights/best.pt",
    "YOLOv5s(u)":   f"{PROJECT}/yolov5su/weights/best.pt",
    "YOLOv5m(u)":   f"{PROJECT}/yolov5mu/weights/best.pt",
    "YOLOv8m(기존)": f"{PROJECT}/yolov8m_fire_smoke/weights/best.pt",   # baseline
}
# ========================================


def sample_image():
    """val 폴더에서 이미지 1장 로드(속도 측정용). 없으면 합성."""
    for p in glob.glob(f"{PROJECT}/../dataset/images/val/*.jpg")[:1]:
        import cv2
        img = cv2.imread(p)
        if img is not None:
            return img
    return (np.random.rand(720, 1280, 3) * 255).astype("uint8")


def measure(name, wpath, img):
    """한 모델: mAP 재평가 + 속도 + 크기."""
    wp = Path(wpath)
    if not wp.exists():
        print(f"  [없음] {name}: {wpath}")
        return None
    model = YOLO(str(wp))

    # 정확도 (동일 val/imgsz 재평가)
    m = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE, verbose=False, plots=False)
    map50, map5095 = float(m.box.map50), float(m.box.map)

    # 속도 (단일 이미지 워밍업 후 반복)
    for _ in range(WARMUP):
        model.predict(img, imgsz=IMGSZ, device=DEVICE, half=(DEVICE != "cpu"), verbose=False)
    t0 = time.time()
    for _ in range(RUNS):
        model.predict(img, imgsz=IMGSZ, device=DEVICE, half=(DEVICE != "cpu"), verbose=False)
    ms = (time.time() - t0) / RUNS * 1000
    fps = 1000.0 / ms if ms else 0.0

    # 크기
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    size_mb = wp.stat().st_size / 1e6

    row = {"model": name, "mAP50": round(map50, 4), "mAP50_95": round(map5095, 4),
           "ms_per_img": round(ms, 1), "fps": round(fps, 1),
           "params_M": round(params, 2), "size_MB": round(size_mb, 1)}
    print(f"  [완료] {name:14s} mAP50={row['mAP50']:.3f} mAP50-95={row['mAP50_95']:.3f} "
          f"{row['fps']:.0f}fps {row['params_M']:.1f}M")
    return row


def main():
    img = sample_image()
    print(f"비교 시작 (imgsz={IMGSZ}, device={DEVICE})\n")
    rows = [r for name, wp in CANDIDATES.items() if (r := measure(name, wp, img))]
    if not rows:
        print("측정된 모델 없음. train_all_models.py 먼저 실행.")
        return

    # CSV 저장
    cols = ["model", "mAP50", "mAP50_95", "ms_per_img", "fps", "params_M", "size_MB"]
    with open("model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    # 마크다운 표
    print("\n| 모델 | mAP@50 | mAP@50-95 | ms/img | FPS | 파라미터(M) | 크기(MB) |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x["mAP50_95"]):
        print(f"| {r['model']} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | "
              f"{r['ms_per_img']:.1f} | {r['fps']:.0f} | {r['params_M']:.1f} | {r['size_MB']:.1f} |")
    print("\n저장: model_comparison.csv  -> 다음: python select_best_model.py")


if __name__ == "__main__":
    main()
