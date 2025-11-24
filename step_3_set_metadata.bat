<#
.SYNOPSIS
    Batch extracts chapters from one or more *.mkv files.
    Correctly embeds the real source filename/path in output metadata.
#>

Param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$InputPath,

    [string]$ChapterFilter
)

# Resolve input (directory or single file)
if (Test-Path $InputPath -PathType Container) {
    $VideoFiles = Get-ChildItem -Path $InputPath -Filter "*.mkv" | Select-Object -ExpandProperty FullName
    if ($VideoFiles.Count -eq 0) { throw "No *.mkv files found in directory: $InputPath" }
    Write-Host "Found $($VideoFiles.Count) .mkv file(s). Processing all..."`n
}
elseif (Test-Path $InputPath -PathType Leaf) {
    if ([IO.Path]::GetExtension($InputPath) -ne ".mkv") {
        throw "Specified file is not an .mkv: $InputPath"
    }
    $VideoFiles = @($InputPath)
}
else {
    throw "Input path not found: $InputPath"
}

# Common setup (executed once)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FFmpeg    = Join-Path $ScriptDir "bin\ffmpeg.exe"
if (-not (Test-Path $FFmpeg)) { throw "ffmpeg.exe not found at: $FFmpeg" }

$VideoFilterChain = (Get-Content (Join-Path $ScriptDir "filters_video.cfg") |
                     Where-Object { $_ -notmatch '^\s*#' -and $_.Trim() -ne '' }) -join ','

$AudioFilterChain = (Get-Content (Join-Path $ScriptDir "filters_audio.cfg") |
                     Where-Object { $_ -notmatch '^\s*#' -and $_.Trim() -ne '' }) -join ','

Write-Host "Video filter chain: $VideoFilterChain"
Write-Host "Audio filter chain: $AudioFilterChain`n"

function Process-VideoFile {
    param([string]$VideoFile)

    $SourceFileName = [IO.Path]::GetFileName($VideoFile)          # e.g. MyTape.mkv
    $SourceFullPath = $VideoFile                                   # full path for metadata

    Write-Host "Processing: $SourceFullPath" -ForegroundColor Cyan

    # Export chapters
    $TempMeta = Join-Path $env:TEMP ("chapters_" + [guid]::NewGuid() + ".ffmetadata")
    & $FFmpeg -nostdin -v error -i $VideoFile -f ffmetadata -y $TempMeta
    if (-not (Test-Path $TempMeta)) {
        Write-Warning "Failed to export chapters from $SourceFileName"
        return
    }

    $lines = Get-Content $TempMeta -Encoding UTF8

    $Start = $null; $End = $null; $Title = $null; $CreationTime = $null

    function Extract-Chapter {
        param(
            [double]$StartNs,
            [double]$EndNs,
            [string]$Title,
            [string]$CreationTime
        )

        if (-not $StartNs -or -not $EndNs -or -not $Title) { return }
        if ($ChapterFilter -and $Title -ne $ChapterFilter) { return }

        $StartSec  = [math]::Round($StartNs / 1e9, 3)
        $EndSec    = [math]::Round($EndNs / 1e9, 3)
        $SafeTitle = $Title -replace '[\/:*?"<>|]', '_'

        # Output filename includes original base name to prevent collisions
        $OutFile = "{0}_{1}.mp4" -f ([IO.Path]::GetFileNameWithoutExtension($VideoFile)), $SafeTitle

        Write-Host "  → '$Title' ($StartSec`s – $EndSec`s) → $OutFile"

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
            -metadata "comment=Extracted chapter from source file: $SourceFullPath" `
            -metadata "encoder=FFmpeg + libx265 (VHS archival preset 2025)" `
            -y $OutFile

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "  FFmpeg failed (exit code $LASTEXITCODE) for chapter '$Title'"
        }
    }

    foreach ($line in $lines) {
        $line = $line.Trim()
        if ($line -eq "") { continue }
        if ($line -eq "[CHAPTER]") {
            Extract-Chapter -StartNs $Start -EndNs $End -Title $Title -CreationTime $CreationTime
            $Start = $null; $End = $null; $Title = $null; $CreationTime = $null
            continue
        }
        if ($line -match '^START=(\d+)')          { $Start = [double]$Matches[1]; continue }
        if ($line -match '^END=(\d+)')            { $End   = [double]$Matches[1]; continue }
        if ($line -match '^title=(.+)$')          { $Title = $Matches[1].Trim(); continue }
        if ($line -match '^creation_time=(.+)$')  { $CreationTime = $Matches[1].Trim(); continue }
    }

    # Last chapter
    Extract-Chapter -StartNs $Start -EndNs $End -Title $Title -CreationTime $CreationTime

    Remove-Item $TempMeta -ErrorAction SilentlyContinue
    Write-Host "Finished: $SourceFileName`n" -ForegroundColor Green
}

# Process every file
foreach ($file in $VideoFiles) {
    Process-VideoFile -VideoFile $file
}

Write-Host "All files processed successfully." -ForegroundColor Green
