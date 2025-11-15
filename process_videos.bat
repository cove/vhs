@echo off
REM Usage: process_videos.bat input_file.mkv

if "%~1"=="" (
    echo Usage: %~nx0 input_file.avi
    exit /b 1
)

set "INPUT=%~1"
for %%I in ("%INPUT%") do set "FILENAME=%%~nxI"
for %%I in ("%INPUT%") do set "BASENAME=%%~nI"

echo Generating videos from "%INPUT%"...

ffmpeg -i "%INPUT%" -map 0:v:0 -c:v ffv1 -level 3 -g 1 -coder 1 -context 1 -slicecrc 1 -timecode 00:00:00:00 -map 0:a:0 -c:a flac -segment_time_metadata 1 "%BASENAME%.mkv"
call :vhs_info "%BASENAME%.mkv"

REM Normal, good quality
REM ffmpeg -i "%INPUT%" -vf "yadif=1" -c:v libx264 -profile:v high -preset slow -crf 16 -c:a aac -movflags +faststart "%BASENAME% Normal.mp4"
REM call :vhs_info "%BASENAME% Normal.mp4"

REM Small, fast downloading
ffmpeg -i "%BASENAME%.mkv" -vf "yadif=1" -pix_fmt yuv420p -c:v libx264 -profile:v baseline -preset slow -crf 20 -vf "scale=640:-2" -c:a aac -b:a 48k -movflags +faststart "%BASENAME% Small.mp4"
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
