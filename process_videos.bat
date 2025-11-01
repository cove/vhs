@echo off
REM Usage: process_videos.bat input_file.mkv

if "%~1"=="" (
    echo Usage: %~nx0 input_file.mkv
    exit /b 1
)

set "INPUT=%~1"
for %%I in ("%INPUT%") do set "FILENAME=%%~nxI"
for %%I in ("%INPUT%") do set "BASENAME=%%~nI"

call :vhs_info "%INPUT%"

echo Generating videos from "%INPUT%"...

REM Normal, good quality
ffmpeg -i "%INPUT%" -vf "yadif=1" -c:v libx264 -profile:v high -preset slow -crf 16 -c:a aac -movflags +faststart "%BASENAME% Normal.mp4"
call :vhs_info "%BASENAME% Normal.mp4"

REM Small, fast downloading
ffmpeg -i "%INPUT%" -c:v libx264 -profile:v baseline -preset slow -crf 20 -vf "scale=640:-2" -c:a aac -b:a 48k "%BASENAME% Small.mp4"
call :vhs_info "%BASENAME% Small.mp4"

exit /b 0

:vhs_info
setlocal
set "IN=%~1"
for %%I in ("%IN%") do set "BN=%%~nI"

REM ffprobe and mediainfo must be on PATH
echo Generating info for "%IN%"
REM ffprobe -v quiet -print_format json -show_format -show_streams "%IN%" > "%BN% info.json"
ffprobe -v quiet -show_format -show_streams "%IN%" > "%BN% info.txt"

REM Blake3: if missing, write notice
echo Generating blake3 hash for "%IN%"
b3sum_windows_x64_bin.exe "%IN%" > "%BN% blake3.txt"
if %errorlevel% neq 0 (
    exit /b 1
)

endlocal
goto :eof
