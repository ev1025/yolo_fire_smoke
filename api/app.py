# -*- coding: utf-8 -*-
"""
화재/연기 탐지 FastAPI 서버.

엔드포인트
- GET  /            웹 데모 페이지
- GET  /health      모델 로드 상태
- POST /detect      이미지 1장 -> 박스+경보 JSON (+주석 이미지)
- POST /detect/video 영상 -> 프레임 샘플링 탐지 -> 요약+프레임별 JSON

가중치 경로 우선순위: 환경변수 WEIGHTS > work/runs/.../best.pt > yolov8m.pt
실행: api/ 에서  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from detector import FireSmokeDetector

ROOT = Path(__file__).resolve().parent


def _resolve_weights():
    cands = []
    if os.environ.get("WEIGHTS"):
        cands.append(Path(os.environ["WEIGHTS"]))
    cands += [
        ROOT.parent / "work" / "runs" / "yolov8m" / "weights" / "best.pt",
        ROOT.parent / "yolov8m.pt",   # best.pt 회수 전 구조 테스트용 폴백
    ]
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None


app = FastAPI(title="화재/연기 탐지 API", version="1.0")
WEIGHTS = _resolve_weights()
IMGSZ = int(os.environ.get("IMGSZ", "1280"))  # 학습값과 일치(1280)
detector = FireSmokeDetector(WEIGHTS, imgsz=IMGSZ) if WEIGHTS else None


@app.get("/health")
def health():
    return {
        "status": "ok" if detector else "no_model",
        "weights": WEIGHTS,
        "imgsz": IMGSZ,
        "classes": detector.names if detector else None,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.25), annotate: bool = Form(True)):
    if not detector:
        raise HTTPException(503, "모델 미로드 (가중치 없음)")
    data = await file.read()
    try:
        return detector.detect_image(data, conf=conf, annotate=annotate)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    frame_stride: int = Form(15),
    max_frames: int = Form(120),
    annotate_frames: bool = Form(False),
):
    if not detector:
        raise HTTPException(503, "모델 미로드 (가중치 없음)")
    data = await file.read()
    return detector.detect_video(
        data, conf=conf, frame_stride=frame_stride, max_frames=max_frames, annotate_frames=annotate_frames
    )


@app.get("/", response_class=HTMLResponse)
def demo():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")
