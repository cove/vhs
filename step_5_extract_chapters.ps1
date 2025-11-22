Param(
    [Parameter(Mandatory=$true)]
    [string]$Input
)

# Validate input file
if (-not (Test-Path $Input)) {
    Write-Error "Input file not found: $Input"
    exit 1
}

# Paths and filenames
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FFmpeg = Join-Path $ScriptDir "bin\ffmpeg.exe"

$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($Input)
$Output = Join-Path (Get-Location) "${BaseName}_metadata.mkv"

# Extract VIDEO_NAME prefix up to first number
if ($BaseName -match '^([^0-9]*[0-9]+)') {
    $VideoName = $matches[1]
} else {
    $VideoName = $BaseName
}

$MetaDir = Join-Path $ScriptDir "media_metadata\$VideoName"

$Cover = Join-Path $MetaDir "cover.jpg"
$TitleFile = Join-Path $MetaDir "title.txt"
$CommentFile = Join-Path $MetaDir "comment.txt"
$Chapters = Join-Path $MetaDir "chapters.ffmetadata"

# Validate metadata files
foreach ($f in @($Cover, $TitleFile, $CommentFile, $Chapters)) {
    if (-not (Test-Path $f)) {
        Write-Error "Missing expected metadata file: $f"
        exit 1
    }
}

# Read title and comment (first line)
$Title = Get-Content $TitleFile -Encoding UTF8 | Select-Object -First 1
$Comment = Get-Content $CommentFile -Encoding UTF8 | Select-Object -First 1

# Create temp chapters file with LF line endings (ffmpeg prefers Unix line endings)
$TempChapters = [System.IO.Path]::Combine($env:TEMP, "chapters_$([guid]::NewGuid().ToString()).ffmetadata")
(Get-Content $Chapters -Raw) -replace "`r`n", "`n" | Set-Content -Encoding UTF8 $TempChapters

# Cover extension lowercase
$CoverExt = ([System.IO.Path]::GetExtension($Cover)).TrimStart('.').ToLower()

Write-Host "Processing '$Input' -> '$Output' ..."
Write-Host "Using temporary chapters file: $TempChapters"

# Build ffmpeg arguments
$FFArgs = @(
    '-nostdin', '-v', 'error',
    '-i', $Input,
    '-f', 'ffmetadata', '-i', $TempChapters,
    '-map', '0:v:0', '-map', '0:a?',
    '-map_metadata', '0',
    '-map_chapters', '-1', '-map_chapters', '1',
    '-c', 'copy',
    '-metadata', "title=$Title",
    '-metadata', "comment=$Comment",
    '-attach', $Cover,
    '-metadata:s:t:0', 'mimetype=image/jpeg',
    '-metadata:s:t:0', "filename=cover.$CoverExt",
    '-color_primaries:v', '6',
    '-color_trc:v', '6',
    '-colorspace:v', '5',
    '-aspect', '4:3',
    '-f', 'matroska',
    $Output,
    '-y'
)

# Run ffmpeg
$proc = Start-Process -FilePath $FFmpeg -ArgumentList $FFArgs -NoNewWindow -Wait -PassThru

# Clean up temp chapters file
Remove-Item $TempChapters -ErrorAction SilentlyContinue

if ($proc.ExitCode -eq 0) {
    Write-Host "Done."
    Write-Host "Output: $Output"
} else {
    Write-Error "ffmpeg returned exit code $($proc.ExitCode)"
    exit $proc.ExitCode
}
