# -*- coding: utf-8 -*-
"""중앙 설정. 모든 스크립트가 이 파일을 import 한다."""
from pathlib import Path

# ===================== 원본 데이터 (D:) =====================
DATA_ROOT      = Path(r"D:\화재연기_원천데이터")   # 로컬 원본 데이터 폴더(환경에 맞게 수정)
SRC_ZIP        = DATA_ROOT / "01.원천데이터" / "TS.zip"   # 분할압축 진입점 (7z가 z01~z08 자동 인식)
SRC_EXTRACTED  = DATA_ROOT / "01.원천데이터" / "TS"        # 이미 풀린 분량 (불꽃 2개 장소)
# 라벨 루트: 폴더명에 '화재 현상'(공백 있음) 주의. 이미지 루트는 '화재현상'(공백 없음).
LABEL_ROOT     = DATA_ROOT / "02.라벨링데이터" / "TL" / "화재 현상" / "이미지"

# ===================== 작업 출력 =====================
# 중간 산출물·가중치·venv 는 work/ 에. 완성 데이터셋은 중앙 DATA/ 로 일원화(2026-06-25). 원본 D:는 읽기전용.
WORK           = Path(__file__).resolve().parent / "work"
MANIFEST_CSV   = WORK / "manifest.csv"      # 전체 라벨 색인 (zip 미접근)
SELECTED_CSV   = WORK / "selected.csv"      # 서브샘플 + train/val 결과
EXTRACT_LIST   = WORK / "extract_list.txt"  # 7z 선택추출용 (클래스폴더 기준 상대경로)
ARCHIVE_PREFIX = WORK / "archive_prefix.txt"  # 03이 탐지한 아카이브 내부 prefix 저장
STAGING        = WORK / "staging"           # 7z가 선택 추출하는 임시 위치
RUNS_DIR       = WORK / "runs"              # 학습 가중치

# 완성 데이터셋(images/labels × train/val)은 중앙 데이터 허브로 분리 (절대경로, 환경에 맞게 수정)
DATA_HOME      = Path(r"C:\Users\eg287\OneDrive\바탕 화면\project\LLM\DATA\datasets\fire_smoke_yolo")
DATASET        = DATA_HOME / "dataset_24k"  # 기본 학습셋(24k). 48k 실험은 dataset_48k
DATA_YAML      = DATA_HOME / "data.yaml"    # 기본 yaml(→dataset_24k). 48k는 data_48k.yaml

# ===================== 클래스 (추천: 2클래스) =====================
# JSON categories_id : 1=fl(불꽃), 2=sm(연기), 3=none(정상)
CLASS_MAP   = {1: 0, 2: 1}          # fl->0, sm->1.  none(3)은 박스 없음 = 네거티브
CLASS_NAMES = ["fire", "smoke"]      # YOLO 클래스 이름
# 클래스 폴더(한글) -> 약어
CLS_DIRS    = {"불꽃": "fl", "연기": "sm", "정상": "none"}

# ===================== 서브샘플링 / 분할 =====================
FRAMES_PER_CLIP = 12     # 클립당 360프레임 -> 12 = 약 4.8만장 (다양성 유지, 중복만 제거)
NORMAL_RATIO    = 0.15   # 정상(네거티브) 이미지 비율 상한 (전체 대비)
VAL_RATIO       = 0.20   # 검증 비율 (반드시 '클립' 단위로 분할)
SEED            = 42

# 아카이브 내부 경로 prefix 후보 (03이 probe로 자동 선택)
ARCHIVE_PREFIX_CANDIDATES = ["화재현상\\이미지", "TS\\화재현상\\이미지"]
