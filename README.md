# 화재/연기 YOLO 학습 파이프라인
**불꽃·연기를 검출**하는 YOLO 모델을 학습하는 파이프라인   
분할압축된 화재, 연기 데이터(800G) 중 일부만 선택추출하여 학습   
배포 환경(CPU/GPU)별 최적 모델을 골라 추론용 ONNX로 변환

**흐름**

```
데이터 준비 (01~04)
  [01] 라벨 색인       라벨 JSON 색인 → manifest (압축 미접근)
   │
  [02] 서브샘플·분할   클립당 12장 + 클립단위 train/val split
   │
  [03] 이미지 추출     7z로 선택한 jpg만 추출 (원본 불변)
   │
  [04] YOLO 변환       COCO JSON → YOLO txt + 이미지
   │
모델 (model.py)
  [05] 다종 학습·HPO   YOLOv8/v5 여러 체급 + Optuna 튜닝
   │
  [06] 비교·선정       mAP·속도·크기 → 배포환경별 최적
   │
  [07] ONNX 배포       CPU 추론용 dynamic ONNX
```

## 프로젝트 구조

```
yolo_fire_smoke/
├── config.py               # 중앙 설정 (경로·클래스·서브샘플 파라미터)
├── 01_build_manifest.py    # 라벨 색인 + 통계 (zip 미접근)
├── 02_subsample_split.py   # 클립당 12장 서브샘플 + 클립단위 train/val 분할
├── 03_extract_images.ps1   # 7z로 선택한 jpg만 추출 (분할압축, 원본 불변)
├── 04_convert_to_yolo.py   # COCO JSON -> YOLO txt + 이미지 복사
├── model.py                # 모델 학습·평가·비교·ONNX 변환
├── requirements.txt
├── api/                    # 추론 백엔드 (FastAPI, 로컬 데모)
│   ├── app.py              #   라우팅 (/detect, /detect/video, /health)
│   ├── detector.py         #   추론 래퍼 (이미지·영상)
│   ├── static/index.html   #   웹 데모 페이지
│   └── README.md
├── server/                 # GPU 서버 학습 스크립트
│   ├── setup.sh            #   venv + torch(cu128) + ultralytics
│   ├── train.sh            #   학습 (분리 실행)
│   ├── eval.sh             #   best.pt 평가
│   ├── data.yaml
│   └── README.md
├── work/                   # 산출물 (gitignore): .venv, runs/(가중치), manifest.csv, selected.csv
└── *.pt                    # 사전학습 체크포인트 (yolov8m/8n 등)
```

학습 데이터셋은 레포 밖 중앙 허브에 분리 (`config.DATA_HOME`가 가리킴):

```
LLM/DATA/datasets/fire_smoke_yolo/
├── dataset_24k/      # 기본 학습셋 (train 19,188 / val 4,770)
├── dataset_48k/      # 48k 실험셋
├── data.yaml         # → dataset_24k
└── data_48k.yaml     # → dataset_48k
```

## 1. 데이터셋 개요

| 항목 | 내용 |
|---|---|
| 정체 | 연기, 화재 발생 이미지 (불꽃·연기·정상 + 화재현장 객체) |
| 원천 압축 | `TS.z01~z08` 각 100GiB + `TS.zip` 3.9GB = **약 804GB 분할압축** |
| 라벨 | COCO 스타일 JSON 약 191만개 (압축 안 됨, 직접 접근) |
| 이미지 스펙 | 1920×1080, 30fps, **클립당 360프레임(=12초)** |



| 클래스 | 코드 | 라벨 수 | 클립 수 |
|---|---|---|---|
| 불꽃 | fl | 610,920 | 1,697 |
| 연기 | sm | 610,920 | 1,697 |
| 정상 | none | 305,280 | 848 |
| **합계** | | **1,527,120** | **4,242** |
- **클립 수** = 라벨 수 ÷ 360 (한 클립이 360프레임)
- **라벨 좌표**: 픽셀 절대값 `bbox = [x, y, w, h]`, 클래스 id `1=불꽃 · 2=연기 · 3=정상`
- **라벨 분포** (학습 타깃) : 불꽃·연기·정상
- **라벨 미사용** : `화재현장 주요객체`(소화기·소화전·표지판 등 15종, 약 38.5만장)는 현장 사물 검출용이라 제외

## 2. 데이터 준비 파이프라인 (01~04)
- **클립 단위 분할**: 같은 클립 프레임이 train/val에 섞이면 데이터 누수 → 성능 거짓 상승. 프레임이 아니라 **클립 단위**로 split.
- **선택추출**: 800GB·30fps라 프레임이 중복 → 압축을 다 풀지 않고 라벨 색인으로 필요한 jpg만.
- **2클래스(fire/smoke)**: 정상은 빈 라벨 네거티브로 써서 오탐 억제.
- **결과 데이터셋**: 23,958장 (train 19,188 / val 4,770, 박스 26,056)

| 단계 | 스크립트 | 산출 |
|---|---|---|
| 1 | `01_build_manifest.py` | 전체 라벨 색인 `manifest.csv` (zip 미접근) |
| 2 | `02_subsample_split.py` | 클립당 12장 서브샘플 + 클립단위 split `selected.csv` |
| 3 | `03_extract_images.ps1` | 선택 jpg만 7z 추출 (원본 불변) |
| 4 | `04_convert_to_yolo.py` | YOLO 레이아웃(images/labels × train/val) + `data.yaml` |

## 3. 산출물

| 위치 | 내용 |
|---|---|
| `LLM/DATA/datasets/fire_smoke_yolo/dataset_24k` | 학습 데이터셋 24k (train 19,188 / val 4,770) |
| `work/manifest.csv` / `work/selected.csv` | 라벨 색인 / 서브샘플·split 결과 |
| `work/runs/<model>/weights/best.pt` (+ `best.onnx`) | 학습 가중치 + 배포용 ONNX |
| `model_comparison.csv` | 모델 비교표 |

## 4. 평가지표

- 검출: mAP@50, mAP@50-95, 클래스별 AP/P/R
- 경보: 정상 이미지를 네거티브로 써서 오탐율(FAR), image-level P/R/F1
- 화재 특성상 recall(놓침 최소) 우선, conf 임계값으로 FAR과 trade-off 튜닝

## 5. 모델 벤치마크 (6종)

6종 동일 조건 학습 후 정확도·속도·크기 종합 비교.

- 데이터: 화재/연기 24,000장 · 1280px · 100 epoch
- 측정: GPU는 .pt(PyTorch), CPU는 ONNX(실제 배포 경로)
- 옵티마이저: ultralytics `auto` (6종 모두 MuSGD)

### 5.1 정확도 + 크기
모든 체급에서 YOLOv8 > YOLOv5(u)   
onnx는 fp32라 .pt의 약 2배 (fp16 변환 시 절반)

| 모델 | mAP@50 | mAP@50-95 | 파라미터 | .pt | onnx |
|---|---|---|---|---|---|
| YOLOv8m (기존) | 0.905 | **0.638** | 25.8M | 52MB | 104MB |
| YOLOv5m | 0.904 | 0.627 | 25.1M | 51MB | 101MB |
| YOLOv8s | 0.895 | 0.622 | 11.1M | 23MB | 45MB |
| YOLOv8n | 0.894 | 0.612 | 3.0M | 6MB | 12MB |
| YOLOv5s | 0.880 | 0.603 | 9.1M | 19MB | 37MB |
| YOLOv5n | 0.879 | 0.598 | 2.5M | 5MB | 10MB |



### 5.2 이미지 1장 추론 시간
프레임당 추론 지연(시간). 낮을수록 빠름.
| 모델 | GPU .pt | CPU 640px | CPU 1280px |
|---|---|---|---|
| YOLOv8m | 4.3ms | 65ms | 280ms |
| YOLOv5m | 3.6ms | 50ms | 204ms |
| YOLOv8s | 1.8ms | 27ms | 117ms |
| YOLOv8n | 0.8ms | 12ms | 45ms |
| YOLOv5s | 1.7ms | 22ms | 86ms |
| YOLOv5n | 0.7ms | 9ms | 34ms |

### 5.3 영상 분석 시간
10초 영상을 frame_stride=15로 약 16장 샘플 분석 (CPU onnx, 실제는 디코드·NMS로 +20~30%).
| 모델 | 640px | 1280px |
|---|---|---|
| YOLOv8m | 1.0s | 4.5s |
| YOLOv5m | 0.8s | 3.3s |
| YOLOv8s | 0.4s | 1.9s |
| YOLOv8n | 0.2s | 0.7s |
| YOLOv5s | 0.4s | 1.4s |
| YOLOv5n | 0.1s | 0.5s |

### 5.4 실시간 처리율
초당 처리 프레임. 라이브 30fps 기준: **✓** 30↑ / **△** 20~29 / **✗** <20. GPU(.pt)는 전 모델 235↑라 전부 ✓.
| 모델 | CPU 640px | CPU 1280px |
|---|---|---|
| YOLOv8m | 15fps ✗ | 3.6fps ✗ |
| YOLOv5m | 20fps ✗ | 4.9fps ✗ |
| YOLOv8s | 37fps ✓ | 8.5fps ✗ |
| YOLOv8n | 86fps ✓ | 22fps △ |
| YOLOv5s | 46fps ✓ | 12fps ✗ |
| YOLOv5n | 109fps ✓ | 30fps ✓ |

### 5.5 정리 + 배포 추천
- **GPU vs CPU**: CPU는 1280px 기준 모델 간 최대 6배 차이 → CPU 배포는 모델 선택이 결정적
- **이미지/영상(업로드 분석)**: 프레임 샘플링이라 CPU로도 충분 (적은 프레임만 추론)
- **실시간(라이브)**: CPU 30fps는 640px + 소형/nano만, 1280px 라이브는 nano만, GPU면 전부

| 배포 환경 | 추천 모델 | mAP@50-95 | 크기(.pt) | CPU 640px |
|---|---|---|---|---|
| CPU 균형 (현재 백엔드) | **YOLOv8s** | 0.622 | 23MB | 37fps |
| CPU 엣지/실시간 | **YOLOv8n** | 0.612 | 6MB | 86fps |
| GPU 최고정확도 | **YOLOv8m** | 0.638 | 52MB | 15fps |

## 6. 하이퍼파라미터 튜닝 (HPO)

CPU 배포 후보 **YOLOv8s·YOLOv8n** 대상. Optuna proxy 튜닝 → 채택값으로 본학습 1회.

### 6.1 방식
- 도구: **Optuna** (TPESampler + MedianPruner), 모델별 study 분리(8s/8n 동시 탐색)
- proxy 탐색: **데이터 15% · 30 epoch · 640px**로 빠르게 많은 조합 시도. 매 epoch fitness 보고 → 나쁜 trial 조기중단
- 탐색 대상: lr0/lrf/momentum/weight_decay/warmup/optimizer(SGD·AdamW)/box/cls/dfl/증강(hsv·degrees·translate·scale·shear·mosaic·mixup)
- 점수: `fitness = 0.1·mAP50 + 0.9·mAP50-95` (ultralytics 기본)

### 6.2 결과

- optimizer: SGD·AdamW 중 **SGD**
- 공통 신호: **box 가중치 ↑ (7.5 → 10~11)**, cls·dfl 소폭 ↑
- fitness 절대값이 낮은 건 proxy(15%·30ep·640px)라서. 본학습(전체·1280px·100ep)에서 회복

| 모델 | 총 trial | 완료/가지치기 | best fitness(proxy) | 비고 |
|---|---|---|---|---|
| YOLOv8s | 15 | 9 / 6 | 0.362 | 기본 조합 대비 개선 |
| YOLOv8n | 14 | 6 / 8 | 0.314 | 개선폭 미미(첫 조합이 최고) |



### 6.3 채택 파라미터 (본학습 적용)

- **lr0/lrf만 0.01로 오버라이드** (HPO값 ~0.0008 미사용): proxy(30ep)가 lr을 낮게 편향시켜 100ep 본학습엔 부적합
- box/cls/dfl/증강은 전이성이 좋아 HPO값 그대로 채택

| 파라미터 | YOLOv8s | YOLOv8n | 출처 |
|---|---|---|---|
| box | 10.26 | 11.06 | HPO |
| cls | 0.78 | 0.72 | HPO |
| dfl | 1.74 | 1.71 | HPO |
| mixup | 0.195 | 0.086 | HPO |
| optimizer | SGD | SGD | HPO |
| **lr0 / lrf** | **0.01 / 0.01** | **0.01 / 0.01** | **오버라이드** |



### 6.4 본학습 결과 (SGD 고정, 진행 중)

- **baseline은 `auto`(MuSGD), 튜닝 재학습은 `SGD` 고정** → 공정 비교 위해 옵티마이저 통일 권장.
- 설정: 1280px, 전체데이터, 100 epoch, patience=30, cos_lr, batch=32.
- 완료 후 ONNX 변환 → baseline 대비 최종 확정 → 백엔드 가중치 교체.

| 모델 | 옵티마이저 | mAP50-95 | baseline 대비 |
|---|---|---|---|
| YOLOv8s 튜닝 | SGD | ~0.629 (ep80/100) | baseline 0.622 **초과** |
| YOLOv8n 튜닝 | SGD | ~0.610 (ep90/100) | baseline 0.612 근접 |


