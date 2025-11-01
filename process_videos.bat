@echo off
REM Usage: process_videos.bat input_file.mkv

if "%~1"=="" (
    echo Usage: %~nx0 input_file.mkv
    exit /b 1
)

set "INPUT=%~1"
for %%I in ("%INPUT%") do set "FILENAME=%%~nxI"
for %%I in ("%INPUT%") do set "BASENAME=%%~nI"

REM Remove "Master" (case-insensitive) from start and trim whitespace using PowerShell
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$p='%INPUT%'; $n=[System.IO.Path]::GetFileNameWithoutExtension($p); $r = $n -replace '^(?i)Master','' -replace '^\s+|\s+$',''; Write-Output $r"`) do set "BASENAME_NO_MASTER=%%A"

call :vhs_info "%INPUT%"

REM Normal, good quality
ffmpeg -i "%INPUT%" -vf "yadif=1" -c:v libx264 -profile:v high -preset slow -crf 16 -c:a aac -movflags +faststart "%BASENAME_NO_MASTER% Normal.mp4"
call :vhs_info "%BASENAME_NO_MASTER% Normal.mp4"

REM Small, fast downloading
ffmpeg -i "%INPUT%" -c:v libx264 -profile:v baseline -preset slow -crf 20 -vf "scale=640:-2" -c:a aac -b:a 48k "%BASENAME_NO_MASTER% Small.mp4"
call :vhs_info "%BASENAME_NO_MASTER% Small.mp4"

exit /b 0

:vhs_info
setlocal
set "IN=%~1"
for %%I in ("%IN%") do set "BN=%%~nI"

REM ffprobe and mediainfo must be on PATH
ffprobe -v quiet -print_format json -show_format -show_streams "%IN%" > "%BN% info.json"
mediainfo "%IN%" > "%BN% info.txt"

REM Blake3: if missing, write notice
b3sum_windows_x64_bin.exe "%IN%" > "%BN% blake3.txt"
if %errorlevel%!=0 (
    echo b3sum_windows_x64_bin.exe not found
    exit /b 1
)

endlocal
goto :eof
