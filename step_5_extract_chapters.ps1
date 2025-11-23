Param(
    [Parameter(Mandatory=$true)]
    [string]$VideoFile,

    [string]$ChapterFilter
)

# Validate input
if (-not (Test-Path $VideoFile)) { throw "Input file not found: $VideoFile" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FFmpeg = Join-Path $ScriptDir "bin\ffmpeg.exe"
if (-not (Test-Path $FFmpeg)) { throw "ffmpeg not found: $FFmpeg" }

# Load filter chains
$VideoFilterChain = ((Get-Content (Join-Path $ScriptDir "filters_video.cfg") | Where-Object {$_ -notmatch '^\s*#' -and $_.Trim()} ) -join ',')
$AudioFilterChain = ((Get-Content (Join-Path $ScriptDir "filters_audio.cfg") | Where-Object {$_ -notmatch '^\s*#' -and $_.Trim()} ) -join ',')

Write-Host "Using video filter chain: $VideoFilterChain"
Write-Host "Using audio filter chain: $AudioFilterChain"

# Export chapters
$TempMeta = Join-Path $env:TEMP ("chapters_" + [guid]::NewGuid() + ".ffmetadata")
& $FFmpeg -nostdin -v error -i $VideoFile -f ffmetadata -y $TempMeta

# Read all lines
$lines = Get-Content $TempMeta -Encoding UTF8

# Initialize
$Start = $null
$End = $null
$Title = $null
$CreationTime = $null

function Process-Chapter {
    param([double]$StartNs, [double]$EndNs, [string]$Title, [string]$CreationTime)

    if (-not $StartNs -or -not $EndNs -or -not $Title) { return }
    if ($ChapterFilter -and $Title -ne $ChapterFilter) { return }

    $StartSec = [math]::Round($StartNs / 1e9, 3)
    $EndSec = [math]::Round($EndNs / 1e9, 3)
    $SafeTitle = ($Title -replace '[\/:*?"<>|]', '_')
    $OutFile = "$SafeTitle.mp4"

    Write-Host "Extracting chapter '$Title' -> $OutFile"
<#
  From Grok AI:
  FINAL x265 SETTINGS – GOLD STANDARD FOR VHS / VHS-C AFTER QTGMC (2025)

  These parameters are ONLY safe and ideal AFTER a proper QTGMC pass.
  Never use them on raw interlaced captures – they will preserve combing and jitter!

  -x265-params "no-sao=1:psy-rd=2.0:psy-rq=2.0:aq-mode=3:deblock=-2,-2"

  Parameter               What it does                                                          Why it belongs AFTER QTGMC (and nowhere else)
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  no-sao=1                Disables Sample Adaptive Offset (SAO)                                 QTGMC already removed all block/combing artefacts. SAO would now only
                                                                                       attack real tape grain → turns snow into plastic. Mandatory OFF for analog.

  psy-rd=2.0              Psychovisual RD optimisation – spend extra bits on visible detail    QTGMC recovered extremely fine luma detail and stabilised grain.
                                                                                       High psy-rd forces x265 to preserve that hard-won detail instead of flattening it.

  psy-rq=2.0              Psychovisual quantisation tuning                                      Protects the natural “breathing” motion of tape grain that QTGMC restored.
                                                                                       Essential for VHS texture to stay alive.

  aq-mode=3               Variance + edge-aware Adaptive Quantisation                           QTGMC output is full of correct, high-frequency noise (grain).
                                                                                       Mode 3 is the only mode that doesn’t murder grain with ugly pooling/worms.

  deblock=-2,-2           Very light positive deblocking                                        QTGMC + SourceMatch/Lossless can leave <0.1 % residual ringing on
                                                                                       insane frames. Tiny deblock cleans the last traces without touching real grain.

  This exact string is used on every single top-tier VHS/VHS-C release in 2025.
  It is the current universal best practice for final archival encodes.
#>
    & $FFmpeg -nostdin -v error -i $VideoFile `
        -ss $StartSec -to $EndSec `
        -pix_fmt yuv422p `
        -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 `
        -tag:v hvc1 `
        -vf "$VideoFilterChain" `
        -c:v libx265 -preset slower -crf 16 `
        -x265-params "no-sao=1:psy-rd=2.0:psy-rq=2.0:aq-mode=3:deblock=-2,-2" `
        -af "$AudioFilterChain" `
        -c:a aac -b:a 48k -ac 1 -ar 48000 `
        -movflags +faststart `
        -metadata "title=$Title" `
        -metadata "creation_time=$CreationTime" `
        -metadata "comment=Extracted chapter from $VideoFile (video_filter_chain=$VideoFilterChain, audio_filter_chain=$AudioFilterChain)" `
        -y $OutFile
}

# Iterate lines robustly
foreach ($line in $lines) {
    $line = $line.Trim()
    if ($line -eq "") { continue }

    if ($line -match '^\[CHAPTER\]') {
        # Process previous chapter
        Process-Chapter -StartNs $Start -EndNs $End -Title $Title -CreationTime $CreationTime
        $Start = $null; $End = $null; $Title = $null; $CreationTime = $null
        continue
    }

    if ($line -match '^START=(\d+)') { $Start = [double]$Matches[1]; continue }
    if ($line -match '^END=(\d+)') { $End = [double]$Matches[1]; continue }
    if ($line -match '^title=(.+)') { $Title = $Matches[1].Trim(); continue }
    if ($line -match '^creation_time=(.+)') { $CreationTime = $Matches[1].Trim(); continue }
}

# Process last chapter after loop
Process-Chapter -StartNs $Start -EndNs $End -Title $Title -CreationTime $CreationTime

# Cleanup temp file
Remove-Item $TempMeta -ErrorAction SilentlyContinue

Write-Host "Chapter extraction complete."
