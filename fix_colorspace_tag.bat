@echo off
REM Usage: update_colorspace_tag.bat input_file.mkv

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
REM Reference: https://en.wikipedia.org/wiki/Color_space#List_of_color_spaces
ffmpeg -i "%INPUT%" -c copy ^
  -color_primaries:v 6 ^
  -color_trc:v 6 ^
  -colorspace:v 5 ^
  "%BASENAME%_fixed.mkv"

echo Done! Output: "%BASENAME%_fixed.mkv"
pause
