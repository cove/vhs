@echo off
REM Usage: step_2_convert_avi_to_mkv.bat captured.avi

if "%~1"=="" (
    echo Usage: %~nx0 captured.avi
    exit /b 1
)

set "INPUT=%~1"
for %%I in ("%INPUT%") do set "FILENAME=%%~nxI"
for %%I in ("%INPUT%") do set "BASENAME=%%~nI"

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "FFMPEG=%SCRIPT_DIR%\bin\ffmpeg.exe"
if not exist "%FFMPEG%" (
    echo [ERROR] ffmpeg.exe not found at "%FFMPEG%"
    pause
    exit /b 1
)

echo Converting "%INPUT%" to mkv container and creating proxy...

REM Switch to mkv container, copying streams without re-encoding
%FFMPEG% -nostdin -v error -i "%INPUT%" ^
    -pix_fmt yuv420p ^
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
    -movflags +faststart ^
    -map 0:v:0 -c:v copy -map 0:a:0 -c:a copy "%BASENAME%.mkv"

REM Make a proxy version for viewing
%FFMPEG% -nostdin -v error -i "%INPUT%" ^
    -pix_fmt yuv420p ^
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
    -c:v libx264 -preset fast -crf 20 -profile:v baseline ^
    -c:a aac -b:a 41.1k -ac 1 -ar 44100 ^
    -movflags +faststart ^
    -metadata "title=%INPUT% Proxy" ^
    "%BASENAME%_proxy.mkv"

echo Conversion complete.
echo "%BASENAME%.mkv" and "%BASENAME%_proxy.mkv" created.

exit /b 0
