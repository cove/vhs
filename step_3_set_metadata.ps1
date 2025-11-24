<#
.SYNOPSIS
    Applies custom metadata, chapters, and cover art to one or more *.mkv files.
    Expects a companion metadata folder structure created alongside the script.
    Usage: .\Apply-Metadata.ps1 *.mkv   (supports wildcards and multiple files)
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$InputFiles
)

# Fail fast on errors (equivalent to set -euo pipefail)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Resolve script directory (works when called via symlink, ./script.ps1, or full path)
$ScriptDir = if ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $PSScriptRoot
}

$FFmpeg = Join-Path $ScriptDir "bin\ffmpeg.exe"
if (-not (Test-Path $FFmpeg)) {
    Write-Error "ffmpeg.exe not found at: $FFmpeg"
    exit 1
}

# Validate at least one input
if ($InputFiles.Count -eq 0) {
    Write-Host "Usage: .\Apply-Metadata.ps1 *.mkv" -ForegroundColor Red
    Write-Host "Example: .\Apply-Metadata.ps1 `"D:\Rips\*.mkv`"" -ForegroundColor Yellow
    exit 1
}

# Resolve wildcards and full paths
$ResolvedFiles = @()
foreach ($pattern in $InputFiles) {
    if (Test-Path $pattern -PathType Leaf) {
        $ResolvedFiles += (Resolve-Path $pattern).Path
    } elseif (Test-Path (Split-Path $pattern) -PathType Container) {
        $ResolvedFiles += (Get-ChildItem -Path $pattern -File | ForEach-Object FullName)
    } else {
        Write-Warning "No files matched: $pattern"
    }
}

if ($ResolvedFiles.Count -eq 0) {
    Write-Error "No input .mkv files found."
    exit 1
}

foreach ($INPUT in $ResolvedFiles) {
    if (-not (Test-Path $INPUT)) {
        Write-Warning "File not found (may have been deleted): $INPUT"
        Write-Host ""
        continue
    }

    $FileName   = [IO.Path]::GetFileName($INPUT)
    $BaseName   = [IO.Path]::GetFileNameWithoutExtension($INPUT)
    $OutputFile = "${BaseName}_metadata.mkv"

    # Extract video name prefix up to the first sequence of digits (e.g. "HomeVideo1995" → "HomeVideo1995")
    if ($BaseName -match '^([^0-9]*[0-9]+)') {
        $VideoName = $Matches[1]
    } else {
        $VideoName = $BaseName   # fallback
    }

    $MetaDir     = Join-Path $ScriptDir "media_metadata\$VideoName"
    $Cover       = Join-Path $MetaDir "cover.jpg"
    $TitleFile   = Join-Path $MetaDir "title.txt"
    $CommentFile = Join-Path $MetaDir "comment.txt"
    $Chapters    = Join-Path $MetaDir "chapters.ffmetadata"

    # Check existence of all required metadata files
    $missing = @()
    foreach ($f in $Cover, $TitleFile, $CommentFile, $Chapters) {
        if (-not (Test-Path $f)) {
            $missing += $f
        }
    }

    if ($missing.Count -gt 0) {
        Write-Warning "Missing metadata files for '$FileName':"
        $missing | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        Write-Host "Skipping this file.`n"
        continue
    }

    # Read text metadata
    $Title   = (Get-Content $TitleFile -Raw).Trim()
    $Comment = (Get-Content $CommentFile -Raw).Trim()

    # Determine cover extension case-insensitively
    $CoverExt = [IO.Path]::GetExtension($Cover).Substring(1)  # removes dot
    $CoverExtLower = $CoverExt.ToLower()

    Write-Host "Processing: `"$FileName`" → `"$OutputFile`"" -ForegroundColor Cyan
    Write-Host "Using metadata folder: $MetaDir" -ForegroundColor DarkCyan

    & $FFmpeg -nostdin -v error -i "$INPUT" `
        -f ffmetadata -i "$Chapters" `
        -map 0:v:0 -map 0:a `
        -map_metadata 0 `
        -map_chapters -1 `
        -map_chapters 1 `
        -c copy `
        -metadata title="$Title" `
        -metadata comment="$Comment" `
        -attach "$Cover" `
        -metadata:s:t:0 mimetype="image/jpeg" `
        -metadata:s:t:0 filename="cover.$CoverExtLower" `
        -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -aspect 4:3 `
        -f matroska "$OutputFile" -y

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Success: $OutputFile" -ForegroundColor Green
    } else {
        Write-Warning "FFmpeg failed on $FileName (exit code: $LASTEXITCODE)"
    }
    Write-Host ""
}

Write-Host "All files processed." -ForegroundColor Green
