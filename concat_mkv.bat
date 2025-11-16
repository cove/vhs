@echo off
REM Usage: concat_videos.bat video1.mp4 video2.mp4 output.mp4

if "%~3"=="" (
    echo Usage: %0 video1.mp4 video2.mp4 output.mp4
    exit /b 1
)

set "VIDEO1=%~1"
set "VIDEO2=%~2"
set "OUTPUT=%~3"

REM Create a temporary list file
set "LISTFILE=concat_list.txt"
echo file '%VIDEO1%' > %LISTFILE%
echo file '%VIDEO2%' >> %LISTFILE%

REM Run FFmpeg concat
ffmpeg -f concat -safe 0 -i %LISTFILE% -c copy "%OUTPUT%"

REM Clean up
del %LISTFILE%

echo Merge complete: %OUTPUT%
pause