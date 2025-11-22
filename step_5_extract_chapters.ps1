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

function Process-Chapter {
    param([double]$StartNs, [double]$EndNs, [string]$Title)

    if (-not $StartNs -or -not $EndNs -or -not $Title) { return }
    if ($ChapterFilter -and $Title -ne $ChapterFilter) { return }

    $StartSec = [math]::Round($StartNs / 1e9, 3)
    $EndSec = [math]::Round($EndNs / 1e9, 3)
    $SafeTitle = ($Title -replace '[\/:*?"<>|]', '_')
    $OutFile = "$SafeTitle.mp4"

    Write-Host "Extracting chapter '$Title' -> $OutFile"

    & $FFmpeg -nostdin -v error -i $VideoFile `
        -ss $StartSec -to $EndSec `
        -pix_fmt yuv420p `
        -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 `
        -tag:v hvc1 `
        -vf "$VideoFilterChain" `
        -c:v libx265 -preset slow -crf 20 -profile:v main `
        -af "$AudioFilterChain" `
        -c:a aac -b:a 41.1k -ac 1 -ar 44100 `
        -movflags +faststart `
        -metadata "title=$Title" `
        -metadata "comment=Extracted chapter from $VideoFile (video_filter_chain=$VideoFilterChain, audio_filter_chain=$AudioFilterChain)" `
        -y $OutFile
}

# Iterate lines robustly
foreach ($line in $lines) {
    $line = $line.Trim()
    if ($line -eq "") { continue }

    if ($line -match '^\[CHAPTER\]') {
        # Process previous chapter
        Process-Chapter -StartNs $Start -EndNs $End -Title $Title
        $Start = $null; $End = $null; $Title = $null
        continue
    }

    if ($line -match '^START=(\d+)') { $Start = [double]$Matches[1]; continue }
    if ($line -match '^END=(\d+)') { $End = [double]$Matches[1]; continue }
    if ($line -match '^title=(.+)') { $Title = $Matches[1].Trim(); continue }
}

# Process last chapter after loop
Process-Chapter -StartNs $Start -EndNs $End -Title $Title

# Cleanup temp file
Remove-Item $TempMeta -ErrorAction SilentlyContinue

Write-Host "Chapter extraction complete."
