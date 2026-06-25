# 화재/연기 탐지 API (FastAPI)

YOLOv8m `best.pt` 추론 서버. 이미지·영상 탐지 + 박스/경보 JSON + 주석 이미지 + 웹 데모.

## 실행
```bash
# 의존성은 requirements.txt 에 포함 (fastapi, uvicorn, python-multipart)
# 가중치 자동 인식: work/runs/yolov8m/weights/best.pt
#   (또는 환경변수 WEIGHTS=<경로> 로 지정)
cd api
uvicorn app:app --host 0.0.0.0 --port 8000
```
브라우저에서 http://localhost:8000 접속 → 웹 데모(이미지/영상 업로드 → 경보·박스·주석이미지).

## 엔드포인트
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 웹 데모 페이지 |
| GET | `/health` | 모델 로드 상태, 가중치 경로, 클래스 |
| POST | `/detect` | 이미지 1장 → 박스 + 경보 JSON (+ 주석 이미지) |
| POST | `/detect/video` | 영상 → 프레임 샘플링 탐지 → 요약 + 프레임별 결과 |

## 파라미터
- `/detect`: `file`(이미지), `conf`(기본 0.25), `annotate`(기본 true)
- `/detect/video`: `file`(영상), `conf`, `frame_stride`(기본 15=30fps 기준 0.5초), `max_frames`(기본 120), `annotate_frames`(기본 false)

## 응답 예시 (/detect)
```json
{
  "detections": [
    {"class": "fire", "class_id": 0, "confidence": 0.946, "bbox_xyxy": [812.4, 233.1, 1042.7, 588.9]}
  ],
  "alarm": {"alarm": true, "status": "ALARM", "classes": ["fire"],
            "counts": {"fire": 1}, "max_confidence": 0.946},
  "image_size": {"width": 1920, "height": 1080},
  "annotated_image": "data:image/jpeg;base64,..."
}
```

경보 규칙: fire 또는 smoke 검출이 1건이라도 있으면 `alarm=true (ALARM)`, 없으면 `NORMAL`.

## 구성
- `app.py` 라우팅 / `detector.py` 추론 래퍼(이미지·영상) / `static/index.html` 웹 데모
- 클래스명은 모델(`best.pt`)에서 직접 읽음(하드코딩 아님)
