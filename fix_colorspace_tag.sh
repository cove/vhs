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
ffmpeg -i "%INPUT%" -c copy ^
  -color_primaries bt601 ^
  -color_trc bt601 ^
  -colorspace smpte170m ^
  "%BASENAME%_fixed.mkv"

echo Done! Output: "%BASENAME%_fixed.mkv"
pause
