<#=====================================================================
  VHS-C → Chapter-by-chapter QTGMC + x265 (2025 – RIGID & FINAL)
  Always assumes chapters exist → always identical workflow
=====================================================================#>
$errorActionPreference = "Stop"

$ScriptDir = if ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $PSScriptRoot
}

$QTGMCDir = Join-Path $ScriptDir "software\FFmpeg-QTGMC Easy 2025.01.11"

# Prepend to PATH so this directory is searched first (only for this PS session)
$env:PATH = "$QTGMCDir;$env:PATH"

$ffmpeg = "ffmpeg.exe"
if (-not (Get-Command $ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg.exe not found in: $QTGMCDir"
    exit 1
}

$ffprobe = "ffprobe.exe"
if (-not (Get-Command $ffprobe -ErrorAction SilentlyContinue)) {
    Write-Error "ffprobe.exe not found in: $QTGMCDir"
    exit 1
}

$files = $args
if ($files.Count -eq 0) { Write-Host "Drag your .mkv files onto this script" -ForegroundColor Red; pause; exit }

foreach ($src in $files) {
    $src = (Resolve-Path $src).Path
    $name = [IO.Path]::GetFileNameWithoutExtension($src)
    $dir  = Split-Path $src -Parent
    $out  = Join-Path $dir ($name + "_chapters")
    New-Item -ItemType Directory -Force -Path $out | Out-Null

    Write-Host "`nProcessing: $name" -ForegroundColor Cyan

    # Get chapters and source creation_time (once)
    $chapters = (& $ffprobe -v error -print_format json -show_chapters $src | ConvertFrom-Json).chapters
    $sourceCreation = (& $ffprobe -v quiet -print_format json -show_format $src | ConvertFrom-Json).format.tags.creation_time

    for ($i = 0; $i -lt $chapters.Count; $i++) {
        $ch    = $chapters[$i]
        $num   = "{0:D2}" -f ($i+1)
        $title = $ch.tags.title.Trim()
        $start = $ch.start_time
        $end   = if ($i -lt $chapters.Count-1) { $chapters[$i+1].start_time } else { (& $ffprobe -v error -show_entries format=duration -of json $src | ConvertFrom-Json).format.duration }

        # creation_time: chapter tag → start time → source file
        $creation = $ch.tags.creation_time
        if (-not $creation) { $creation = (Get-Date "1970-01-01").AddSeconds([double]$start).ToString("yyyy-MM-ddTHH:mm:ss.000000Z") }
        if (-not $creation) { $creation = $sourceCreation }

        $safeTitle = $title -replace '[:<>"/\\|?*]', ' -'
        $final     = "$out\$num - $safeTitle.mp4"
        $tempRaw   = "$env:TEMP\vhs_chap_$num.mkv"

        Write-Host "   → $num - $title" -ForegroundColor Gray

        # 1. Extract raw chapter (fast, tiny file)
        & $ffmpeg -v error -ss $start -to $end -i $src -map 0:v -map 0:a? -map 0:s? -c copy -avoid_negative_ts make_zero -y $tempRaw

        # 2. QTGMC + x265 only this chapter
        $avs = "$ScriptDir\qtgmc_$num.avs"
@"
LoadPlugin("$QTGMCDir/ffms2.dll") 
LoadPlugin("$QTGMCDir/masktools2.dll") 
LoadPlugin("$QTGMCDir/Rgtools.dll") 
LoadPlugin("$QTGMCDir/mvtools2.dll")
LoadPlugin("$QTGMCDir/nnedi3.dll") 
LoadPlugin("$QTGMCDir/yadifmod2.dll") 
LoadPlugin("$QTGMCDir/fft3dfilter.dll") 
LoadPlugin("$QTGMCDir/LoadDLL64.dll")
LoadDLL("$QTGMCDir/libfftw3f-3.dll") 
Import("$QTGMCDir/Zs_RF_Shared.avsi") 
Import("$QTGMCDir/QTGMC.avsi")
FFmpegSource2("$tempRaw", atrack=-1) 
ConvertToYV12(matrix="Rec601")
QTGMC(preset="Faster") 
Crop(0,0,-2,-6) 
LanczosResize(640,480) 
<<<<<<< HEAD

=======
SetPixelAspectRatio(1.0) 
>>>>>>> 3c77b07cba05848a0c0e9c15d60b08ac7d70671b
Return Last
"@ | Set-Content -Path $avs -Encoding ASCII

        & $ffmpeg -i $avs -v error -i $tempRaw `
            -map 0:v -map 1:a? -map_metadata 1 `
            -metadata title="$title" `
            -metadata comment="Extracted chapter from $([IO.Path]::GetFileName($src))" `
            -metadata creation_time="$creation" `
            -c:v libx265 -preset slow -crf 18 -x265-params "profile=main10:aq-mode=3" `
            -c:a aac -b:a 48k `
            -af "highpass=f=80,lowpass=f=14000,acompressor=ratio=3:attack=8:release=60" `
            -movflags +faststart -y "$final"

        # 3. Delete temps immediately
        Remove-Item $tempRaw -Force
        Remove-Item $avs -Force
    }

    Write-Host "Finished → $out" -ForegroundColor Green
}

Write-Host "`nAll done! Every chapter processed identically. Zero temp space left." -ForegroundColor Magenta
pause
