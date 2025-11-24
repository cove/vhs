@echo off
REM Usage: truncate_video.bat input.mp4 output.mp4 01:04:28

if "%~3"=="" (
    echo Usage: %0 input_file output_file duration
    echo Example: %0 input.mp4 output.mp4 01:04:28
    exit /b 1
)

set INPUT=%~1
set OUTPUT=%~2
set DURATION=%~3

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "FFMPEG=%SCRIPT_DIR%\FFmpeg-QTGMC Easy 2025.01.11\ffmpeg.exe"
if not exist "%FFMPEG%" (
    echo [ERROR] ffmpeg.exe not found at "%FFMPEG%"
    pause
 
    exit /b 1
)

%FFMPEG% -nostdin -v error -i "%INPUT%" -t %DURATION% -c copy "%OUTPUT%"
echo Trim complete: %OUTPUT%
pause
exit /b 0
