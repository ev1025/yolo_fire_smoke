# ============================================================
# 03. 선택한 jpg만 분할압축에서 추출.
#
# 입력 : extract_list.txt    (02가 고른 jpg 목록)
#        TS.zip + z01~z08     (원본 분할압축, 약 800GB. 읽기 전용)
# 출력 : work/staging         (목록의 jpg만 풀린 폴더)
#        archive_prefix.txt   (압축 내부 경로 prefix. 04가 사용)
#
# 메모 : 분할압축은 tar 불가 -> 7-Zip 필요. 'x'(추출)만 써서 원본은 불변.
# ============================================================
$ErrorActionPreference = "Stop"

$WORK      = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "work"
$ZIP       = "D:\화재연기_원천데이터\01.원천데이터\TS.zip"
$LIST      = Join-Path $WORK "extract_list.txt"
$STAGING   = Join-Path $WORK "staging"
$PREFIXOUT = Join-Path $WORK "archive_prefix.txt"
$FULL      = Join-Path $WORK "extract_list_full.txt"
$CANDS     = @("화재현상\이미지", "TS\화재현상\이미지")    # 압축 내부 경로 prefix 후보
$noBom     = New-Object System.Text.UTF8Encoding($false)   # Python utf-8 호환(BOM 금지)

# 1) 7-Zip 찾기 (Program Files 또는 PATH). 없으면 안내 후 종료
$sz = @("C:\Program Files\7-Zip\7z.exe", "C:\Program Files (x86)\7-Zip\7z.exe",
        (Get-Command 7z -ErrorAction SilentlyContinue).Source) |
      Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $sz)               { Write-Host "[에러] 7-Zip 없음 -> winget install --id 7zip.7zip -e" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $LIST)) { Write-Host "[에러] $LIST 없음. 02 먼저 실행." -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path $STAGING | Out-Null
Write-Host "7-Zip: $sz"

# 2) prefix 자동 탐지: 첫 파일이 어느 후보 경로로 압축 안에 들어있는지 확인
$firstRel = Get-Content -LiteralPath $LIST -TotalCount 1
$prefix = $CANDS | Where-Object {
    (& $sz l $ZIP -i!"$_\$firstRel" 2>&1 | Out-String) -match [regex]::Escape([IO.Path]::GetFileName($firstRel))
} | Select-Object -First 1
if (-not $prefix) { Write-Host "[에러] 압축 내부 prefix 탐지 실패. CANDS 확인 필요." -ForegroundColor Red; exit 1 }
Write-Host "탐지된 prefix: $prefix"
[System.IO.File]::WriteAllText($PREFIXOUT, $prefix, $noBom)

# 3) prefix 붙인 추출 목록 작성 (BOM 없이)
$lines = Get-Content -LiteralPath $LIST | ForEach-Object { "$prefix\$_" }
[System.IO.File]::WriteAllLines($FULL, $lines, $noBom)
Write-Host "추출 대상 $($lines.Count) 개 -> $STAGING"

# 4) 선택 추출 (x = 추출, 원본 불변)
& $sz x $ZIP "-o$STAGING" "@$FULL" -y -bsp1
if ($LASTEXITCODE -ne 0) { Write-Host "[경고] 7z 종료코드 $LASTEXITCODE" -ForegroundColor Yellow }
Write-Host "[완료] 추출 끝 -> 다음: python 04_convert_to_yolo.py"
