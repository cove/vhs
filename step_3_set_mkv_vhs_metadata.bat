@echo off
REM Usage: step_3_set_mkv_vhs_metadata.bat input_file.mkv

if "%~1"=="" (
    echo Usage: %~nx0 input_file.mkv
    exit /b 1
)

set "INPUT=%~1"
for %%I in ("%INPUT%") do set "FILENAME=%%~nxI"
for %%I in ("%INPUT%") do set "BASENAME=%%~nI"

echo Updating colorspace metadata for "%INPUT%"...

REM Update MKV metadata (no re-encoding)
REM For standard-definition BT.601 / SMPTE 170M, the correct numeric IDs are:
REM  - color_primaries: 6 (SMPTE 170M)
REM  - color_trc: 6 (SMPTE 170M)
REM  - colorspace: 5 (SMPTE 170M)
REM  - interlaced_frame: 1 (true)
REM  - aspect ratio: 4:3
REM  - Remove encoder metadata from video and audio streams so they can be compared fairly
REM  - -flags +ilme+ildct -top 0 Bottom Field First for interlaced video (BFF)
ffmpeg -i "%INPUT%" -c copy ^
  -color_primaries:v 6 ^
  -color_trc:v 6 ^
  -colorspace:v 5 ^
  -interlaced_frame 1 ^
  -aspect 4:3 ^
  -metadata:s:v:0 encoder="" ^
  -metadata:s:a:0 encoder="" ^
  -flags +ilme+ildct -vf "setfield=bff" ^
  "%BASENAME%_metadata.mkv"

echo Done! Output: "%BASENAME%_metadata.mkv"
pause
