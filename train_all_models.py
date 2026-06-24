# -*- coding: utf-8 -*-
"""
Phase 2. 여러 YOLO 모델을 동일 데이터(24k)·동일 설정으로 학습.

대상 : YOLOv8n/s, YOLOv5n/s/m (5개)
       * YOLOv8m은 제외 (이미 학습됨 = baseline, Phase 3 비교군)
       * YOLOv5는 ultralytics 네이티브 u-variant(yolov5nu/su/mu, 앵커프리)

입력 : DATA (24k data.yaml)
출력 : PROJECT/<model>/weights/best.pt  + results.csv (모델별)
실행 : python train_all_models.py                 # 5개 전체 순차
       python train_all_models.py yolov8n yolov8s # 일부만

설정은 아래 상수 블록에서 한 번에 조정. baseline과 동일 레시피(box=7.5, flipud=0).
"""
import os, sys, time
from pathlib import Path
from ultralytics import YOLO

# ================= 설정 (env로 머신별 조정) =================
DATA    = os.environ.get("DATA", "data.yaml")     # 24k 데이터셋 (절대경로 권장)
PROJECT = os.environ.get("PROJECT", "runs")       # 결과 저장 (data2 등 절대경로 권장)
IMGSZ   = int(os.environ.get("IMGSZ", "1280"))    # 학습·평가 해상도 (cfg2/배포와 동일하게 1280)
EPOCHS  = int(os.environ.get("EPOCHS", "100"))
BATCH   = int(os.environ.get("BATCH", "-1"))      # -1=AutoBatch(단일GPU 권장). 서버 DDP는 BATCH=32 등 고정값
DEVICE  = os.environ.get("DEVICE", "0")           # 서버 "2,3" / 로컬 "0"

# 학습할 모델: CLI 인자로 받거나(없으면) 5개 전체. yolov5는 ultralytics u-variant
MODELS = ["yolov8n", "yolov8s", "yolov5nu", "yolov5su", "yolov5mu"]

# baseline과 동일 레시피 (box=7.5 기본, 불꽃 도메인상 flipud=0)
COMMON = dict(
    box=7.5, flipud=0.0, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    mosaic=1.0, close_mosaic=10,
    patience=30, cos_lr=True,
)
# ========================================


def train_one(name):
    """모델 1개 학습. 이미 끝난 run(best.pt 존재)은 건너뜀(재실행 안전)."""
    out = Path(PROJECT) / name / "weights" / "best.pt"
    if out.exists():
        print(f"[건너뜀] {name}: 이미 학습됨 ({out})")
        return name, True, 0.0

    print(f"\n{'='*60}\n[학습 시작] {name}  imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH} dev={DEVICE}\n{'='*60}")
    t0 = time.time()
    try:
        YOLO(f"{name}.pt").train(
            data=DATA, project=PROJECT, name=name, exist_ok=True,
            imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, device=DEVICE,
            **COMMON,
        )
        dt = time.time() - t0
        print(f"[완료] {name}  소요 {dt/60:.1f}분  -> {out}")
        return name, True, dt
    except Exception as e:                       # 한 모델 실패해도 나머지 진행
        print(f"[실패] {name}: {e}")
        return name, False, time.time() - t0


def main():
    targets = sys.argv[1:] or MODELS
    print(f"학습 대상 {len(targets)}개: {targets}")
    results = [train_one(m) for m in targets]

    print(f"\n{'='*60}\n[전체 요약]")
    for name, ok, dt in results:
        print(f"  {name:12s} {'OK ' if ok else 'FAIL'}  {dt/60:6.1f}분")
    done = [n for n, ok, _ in results if ok]
    print(f"\n학습 완료 {len(done)}/{len(targets)}. 다음: python compare_models.py")


if __name__ == "__main__":
    main()
