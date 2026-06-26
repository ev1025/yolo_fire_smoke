# -*- coding: utf-8 -*-
"""
hard-negative 마이닝: 미사용 정상프레임(약 30만장)에서 '현재 모델이 실제로 오탐(false positive)을
내는' 프레임만 골라 배경 네거티브로 학습에 투입한다. 무작위 정상보다 FAR 저감 효과가 크다.

근거: hard-negative mining(OHEM, Shrivastava CVPR2016) + 배경 이미지 투입(Ultralytics 권장).
배포가 순수 onnxruntime이므로 scoring도 onnxruntime으로(=배포와 동일 기준으로 오탐 마이닝).

파이프라인(서브커맨드):
  candidates : manifest의 정상 클립에서 후보 프레임 균등 샘플 -> work/hn/extract_list_hn.txt
               (val 정상클립 제외 = 누수 방지, 이미 쓰인 정상 stem 제외)
  (추출)     : hn_extract.ps1 실행 -> work/hn/staging 에 후보 jpg 선택추출 (원본 D: 불변)
  score      : 추출본을 현재 onnx로 추론 -> 오탐 심각도 점수 work/hn/scores.csv
               (정상 프레임이므로 '탐지=오탐'. 점수 = 최대 conf, 탐지 수)
  inject     : 점수 상위 K장을 dataset train 에 배경(빈 라벨)으로 복사 -> work/hn/injected.csv (되돌리기용)

사용:
  python hn_mine.py candidates --per-clip 16
  powershell -File hn_extract.ps1
  python hn_mine.py score  --weights work/onnx_final/yolov8m.onnx --conf 0.25
  python hn_mine.py inject --topk 3000
"""
import sys, csv, argparse, shutil
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import config as C

HN          = C.WORK / "hn"
CAND_CSV    = HN / "candidates.csv"
EXTRACT_HN  = HN / "extract_list_hn.txt"
STAGING_HN  = HN / "staging"
SCORES_CSV  = HN / "scores.csv"
INJECTED_CSV = HN / "injected.csv"
csv.field_size_limit(10**7)


def pick_even(items, k):
    """리스트에서 균등 간격 k개 (k 이하면 그대로). 02_subsample_split 과 동일 규칙."""
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _used_split():
    """selected.csv 에서 (val 정상 클립키 집합, 이미 쓰인 stem 집합) 반환."""
    val_clips, used_stems = set(), set()
    if not C.SELECTED_CSV.exists():
        return val_clips, used_stems
    with open(C.SELECTED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            used_stems.add(r["stem"])
            if r["cls_kr"] == "정상" and r["split"] == "val":
                val_clips.add((r["place_kr"], r["clip"]))
    return val_clips, used_stems


# ---------------- candidates ----------------
def cmd_candidates(a):
    val_clips, used_stems = _used_split()
    # manifest 의 정상 프레임을 클립별로 묶기
    clip_rows = defaultdict(list)
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cls_kr"] != "정상":
                continue
            clip_rows[(r["place_kr"], r["clip"])].append(r)

    cand, skip_val, skip_used = [], 0, 0
    for key, rows in sorted(clip_rows.items()):
        if key in val_clips:                       # val 정상클립은 제외(누수 방지)
            skip_val += 1
            continue
        rows = sorted((r for r in rows if r["frame"].lstrip("-").isdigit()),
                      key=lambda r: int(r["frame"]))
        for r in pick_even(rows, a.per_clip):
            if r["stem"] in used_stems:            # 이미 학습에 쓰인 프레임 제외
                skip_used += 1
                continue
            cand.append(r)

    HN.mkdir(parents=True, exist_ok=True)
    cols = ["place_kr", "clip", "frame", "stem", "json_path", "img_rel"]
    with open(CAND_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in cand:
            w.writerow([r[c] for c in cols])
    EXTRACT_HN.write_text("\n".join(sorted({r["img_rel"] for r in cand})) + "\n", encoding="utf-8")

    place_cnt = defaultdict(int)
    for r in cand:
        place_cnt[r["place_kr"]] += 1
    print(f"후보 {len(cand):,}장  (정상클립 {len(clip_rows):,}개 중 val제외 {skip_val}, 기사용 프레임 제외 {skip_used:,})")
    print("장소별 후보:")
    for p, n in sorted(place_cnt.items(), key=lambda x: -x[1]):
        print(f"   {p:14s} {n:>7,}")
    print(f"\n출력 -> {CAND_CSV}\n        {EXTRACT_HN}")
    print("다음: powershell -File hn_extract.ps1   (work/hn/staging 으로 선택추출)")


# ---------------- score (순수 onnxruntime) ----------------
def _make_session(weights):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(weights), providers=prov)
    dev = "gpu" if "CUDAExecutionProvider" in sess.get_providers() else "cpu"
    return sess, sess.get_inputs()[0].name, dev


def _letterbox(img_bgr, imgsz):
    import cv2
    import numpy as np
    h, w = img_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dh, dw = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(blob.transpose(2, 0, 1)[None])


def cmd_score(a):
    import cv2
    import numpy as np
    cand_path = Path(a.candidates) if a.candidates else CAND_CSV   # 파일럿은 부분집합 csv 지정 가능
    if not cand_path.exists():
        sys.exit(f"{cand_path} 없음. 먼저 candidates 실행.")
    sess, in_name, dev = _make_session(a.weights)
    print(f"onnx: {a.weights}  device={dev}  imgsz={a.imgsz}  conf={a.conf}")

    # 추출본 stem -> 실제 파일 경로 매핑 (staging 안 어디에 있든 stem.jpg 로 탐색)
    prefix_file = C.ARCHIVE_PREFIX
    prefix = prefix_file.read_text(encoding="utf-8-sig").strip() if prefix_file.exists() \
        else C.ARCHIVE_PREFIX_CANDIDATES[0]

    rows = list(csv.DictReader(open(cand_path, encoding="utf-8")))
    HN.mkdir(parents=True, exist_ok=True)
    out = open(SCORES_CSV, "w", newline="", encoding="utf-8")
    w = csv.writer(out); w.writerow(["stem", "place_kr", "clip", "img_rel", "n_det", "max_conf", "top_class"])
    done = miss = fired = 0
    for r in rows:
        src = STAGING_HN / prefix / r["img_rel"]
        if not src.exists():
            miss += 1; continue
        img = cv2.imread(str(src))
        if img is None:
            miss += 1; continue
        blob = _letterbox(img, a.imgsz)
        pred = sess.run(None, {in_name: blob})[0]          # [1, 4+nc, N]
        p = np.squeeze(pred, 0).T                           # [N, 4+nc]
        scores = p[:, 4:]
        cls = np.argmax(scores, axis=1)
        conf = scores[np.arange(len(scores)), cls]
        keep = conf >= a.conf
        n_det = int(keep.sum())
        if n_det:
            mx = conf[keep].max()
            top = int(cls[keep][np.argmax(conf[keep])])
            fired += 1
        else:
            mx, top = 0.0, -1
        w.writerow([r["stem"], r["place_kr"], r["clip"], r["img_rel"],
                    n_det, f"{float(mx):.4f}", top])
        done += 1
        if done % 1000 == 0:
            print(f"  ...{done:,}/{len(rows):,}  오탐발생 {fired:,}", flush=True)
    out.close()
    print(f"\n[완료] 채점 {done:,}장 (누락 {miss:,}). 오탐발생 {fired:,}장 ({fired/max(done,1)*100:.1f}%)")
    print(f"출력 -> {SCORES_CSV}\n다음: python hn_mine.py inject --topk 3000")


# ---------------- inject ----------------
def cmd_inject(a):
    if not SCORES_CSV.exists():
        sys.exit("scores.csv 없음. 먼저 score 실행.")
    prefix = C.ARCHIVE_PREFIX.read_text(encoding="utf-8-sig").strip() \
        if C.ARCHIVE_PREFIX.exists() else C.ARCHIVE_PREFIX_CANDIDATES[0]
    rows = [r for r in csv.DictReader(open(SCORES_CSV, encoding="utf-8")) if int(r["n_det"]) > 0]
    rows.sort(key=lambda r: float(r["max_conf"]), reverse=True)
    pick = rows[:a.topk]
    if not pick:
        sys.exit("오탐 프레임이 없음(주입할 hard-negative 없음). conf 낮춰 재채점 고려.")

    img_dir = C.DATASET / "images" / "train"
    lbl_dir = C.DATASET / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    inj = open(INJECTED_CSV, "w", newline="", encoding="utf-8")
    w = csv.writer(inj); w.writerow(["stem", "place_kr", "clip", "max_conf", "top_class"])
    ok = miss = 0
    for r in pick:
        src = STAGING_HN / prefix / r["img_rel"]
        if not src.exists():
            miss += 1; continue
        shutil.copy2(src, img_dir / f'{r["stem"]}.jpg')      # 원본 보존(copy2)
        (lbl_dir / f'{r["stem"]}.txt').write_text("", encoding="utf-8")  # 빈 라벨 = 배경 네거티브
        w.writerow([r["stem"], r["place_kr"], r["clip"], r["max_conf"], r["top_class"]])
        ok += 1
    inj.close()
    th = float(pick[-1]["max_conf"])
    print(f"[완료] hard-negative {ok:,}장 주입 (누락 {miss}). conf 컷 {th:.3f} 이상")
    print(f"되돌리기용 목록 -> {INJECTED_CSV}")
    print(f"dataset: {C.DATASET}  ->  python model.py train --models yolov8m  (재학습 후 FAR 비교)")


def main():
    ap = argparse.ArgumentParser(description="hard-negative 마이닝")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates"); c.add_argument("--per-clip", type=int, default=16, dest="per_clip")
    c.set_defaults(fn=cmd_candidates)
    s = sub.add_parser("score")
    s.add_argument("--weights", required=True); s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--conf", type=float, default=0.25)
    s.add_argument("--candidates", default=None)   # 미지정시 전체 candidates.csv, 파일럿은 부분집합 지정
    s.set_defaults(fn=cmd_score)
    j = sub.add_parser("inject"); j.add_argument("--topk", type=int, default=3000)
    j.set_defaults(fn=cmd_inject)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
