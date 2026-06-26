# ============================================================
# hard-negative 후보 jpg만 분할압축에서 선택추출. 03_extract_images.ps1 의 hn 버전.
#
# 입력 : work/hn/extract_list_hn.txt  (hn_mine.py candidates 가 고른 정상 jpg 목록)
#        TS.zip + z01~z08              (원본 분할압축, 읽기 전용)
# 출력 : work/hn/staging               (후보 jpg만 풀린 폴더)
#        archive_prefix.txt            (없으면 탐지해서 생성. score/inject 가 사용)
#
# 메모 : 'x'(추출)만 사용 -> 원본 D: 불변.
# ============================================================
$ErrorActionPreference = "Stop"

$ROOT      = Split-Path -Parent $MyInvocation.MyCommand.Path
$WORK      = Join-Path $ROOT "work"
$ZIP       = "D:\화재연기_원천데이터\01.원천데이터\TS.zip"
$LIST      = Join-Path $WORK "hn\extract_list_hn.txt"
$STAGING   = Join-Path $WORK "hn\staging"
$PREFIXOUT = Join-Path $WORK "archive_prefix.txt"
$FULL      = Join-Path $WORK "hn\extract_list_hn_full.txt"
$CANDS     = @("화재현상\이미지", "TS\화재현상\이미지")
$noBom     = New-Object System.Text.UTF8Encoding($false)

# 1) 7-Zip 찾기
$sz = @("C:\Program Files\7-Zip\7z.exe", "C:\Program Files (x86)\7-Zip\7z.exe",
        (Get-Command 7z -ErrorAction SilentlyContinue).Source) |
      Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $sz)               { Write-Host "[에러] 7-Zip 없음 -> winget install --id 7zip.7zip -e" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $LIST)) { Write-Host "[에러] $LIST 없음. hn_mine.py candidates 먼저 실행." -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path $STAGING | Out-Null
Write-Host "7-Zip: $sz"

# 2) prefix: 이미 있으면 재사용, 없으면 첫 파일로 탐지
if (Test-Path $PREFIXOUT) {
    $prefix = (Get-Content -LiteralPath $PREFIXOUT -Raw).Trim()
} else {
    $firstRel = Get-Content -LiteralPath $LIST -TotalCount 1
    $prefix = $CANDS | Where-Object {
        (& $sz l $ZIP -i!"$_\$firstRel" 2>&1 | Out-String) -match [regex]::Escape([IO.Path]::GetFileName($firstRel))
    } | Select-Object -First 1
    if (-not $prefix) { Write-Host "[에러] 압축 내부 prefix 탐지 실패." -ForegroundColor Red; exit 1 }
    [System.IO.File]::WriteAllText($PREFIXOUT, $prefix, $noBom)
}
Write-Host "prefix: $prefix"

# 3) prefix 붙인 추출 목록
$lines = Get-Content -LiteralPath $LIST | ForEach-Object { "$prefix\$_" }
[System.IO.File]::WriteAllLines($FULL, $lines, $noBom)
Write-Host "추출 대상 $($lines.Count) 개 -> $STAGING"

# 4) 선택 추출 (x = 추출, 원본 불변)
& $sz x $ZIP "-o$STAGING" "@$FULL" -y -bsp1
if ($LASTEXITCODE -ne 0) { Write-Host "[경고] 7z 종료코드 $LASTEXITCODE" -ForegroundColor Yellow }
Write-Host "[완료] 추출 끝 -> 다음: python hn_mine.py score --weights work/onnx_final/yolov8m.onnx"
