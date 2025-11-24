<#=====================================================================
  VHS-C Restoration and Chapter Splitting Script
=====================================================================#>

# -------------------------- CONFIGURATION --------------------------
$AVS_Template = @"
LoadPlugin("ffms2.dll")
LoadPlugin("masktools2.dll")
LoadPlugin("Rgtools.dll")
LoadPlugin("mvtools2.dll")
LoadPlugin("nnedi3.dll")
LoadPlugin("yadifmod2.dll")
LoadPlugin("fft3dfilter.dll")
LoadPlugin("LoadDLL64.dll")
LoadDLL("libfftw3f-3.dll")
Import("Zs_RF_Shared.avsi")
Import("QTGMC.avsi")

FFmpegSource2("%SOURCE%", atrack=-1)
ConvertToYV12(matrix="Rec601")
QTGMC(preset="Faster")
Crop(0, 0, -2, -6)
LanczosResize(640, 480)
SetPixelAspectRatio(1.0)
Return Last
"@

$AudioFilter = "dehummer=f=60:mode=peak:q=3, highpass=f=80, arnndn=m=bdnr.pmd, lowpass=f=14000, acompressor=ratio=3:attack=8:release=60:makeup=2"
$VideoPreset = "slow"
$CRF         = "18"
# ------------------------------------------------------------------

# Resolve script directory (works when called via symlink, ./script.ps1, or full path)
$ScriptDir = if ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    $PSScriptRoot
}

$FFmpeg = Join-Path $ScriptDir "software\FFmpeg-QTGMC Easy 2025.01.11\ffmpeg.exe"
if (-not (Test-Path $FFmpeg)) {
    Write-Error "ffmpeg.exe not found at: $FFmpeg"
    exit 1
}

$Files = $args
if ($Files.Count -eq 0) {
    Write-Host "Drag & drop .mkv files onto this script or pass them as arguments." -ForegroundColor Red
    pause
    exit
}

foreach ($SourcePath in $Files) {
    $SourcePath = (Resolve-Path $SourcePath).Path
    $SourceName = [IO.Path]::GetFileNameWithoutExtension($SourcePath)
    $SourceFileName = [IO.Path]::GetFileName($SourcePath)
    $SourceDir  = Split-Path $SourcePath -Parent

    $OutFolder = Join-Path $SourceDir ("{0}_chapters" -f $SourceName)
    New-Item -ItemType Directory -Force -Path $OutFolder | Out-Null

    $AVS_Content = $AVS_Template -replace "%SOURCE%", $SourcePath.Replace("\","\\")
    $TempAVS = Join-Path $Env:TEMP ("{0}_qtgmc.avs" -f [guid]::NewGuid())
    Set-Content -Path $TempAVS -Value $AVS_Content -Encoding ASCII

    $CoverPath = Join-Path $SourceDir "cover.jpg"
    $AttachCover = if (Test-Path $CoverPath) { "-attach `"$CoverPath`" -metadata:s:t mimetype=image/jpeg" } else { "" }

    Write-Host "`nProcessing: $SourceFileName → $OutFolder" -ForegroundColor Cyan

    & $FFmpeg -i $TempAVS -i $SourcePath `
        -map 0:v -map 1:a? -map 1:s? `
        -map_metadata 1 -map_chapters -1 `
        $AttachCover `
        -c:v libx265 -preset $VideoPreset -crf $CRF -x265-params "profile=main10:level-idc=5.1:aq-mode=3" `
        -c:a aac -b:a 192k `
        -af $AudioFilter `
        -metadata title="%chapter_title" `
        -metadata comment="Extracted chapter from $SourceFileName" `
        -f segment `
        -segment_chapters all `
        -segment_format mp4 `
        -segment_format_options movflags=+faststart `
        -reset_timestamps 1 `
        "$OutFolder/%chapter_title%.mp4"

    Remove-Item $TempAVS -Force
    Write-Host "Finished $SourceFileName" -ForegroundColor Green
}

Write-Host "`nAll done! Every chapter now has:" -ForegroundColor Magenta
Write-Host "   • title = chapter name (e.g., 'Christmas Morning 1995')"
Write-Host "   • comment = 'Extracted chapter from OriginalFile.mkv'"
Write-Host "   • original creation_time, encoder, etc."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
