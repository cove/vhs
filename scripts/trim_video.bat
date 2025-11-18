@echo off
REM Usage: trim_video.bat input.mp4 output.mp4 01:04:28

if "%~3"=="" (
    echo Usage: %0 input_file output_file duration
    echo Example: %0 input.mp4 output.mp4 01:04:28
    exit /b 1
)

set INPUT=%~1
set OUTPUT=%~2
set DURATION=%~3

ffmpeg -i "%INPUT%" -t %DURATION% -c copy "%OUTPUT%"
echo Trim complete: %OUTPUT%
pause