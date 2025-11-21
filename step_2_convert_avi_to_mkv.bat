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

echo Converting "%INPUT%" to "%BASENAME%.mkv"...
%FFMPEG% -nostdin -v error -i "%INPUT%" ^
    -pix_fmt yuv420p ^
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
    -movflags +faststart ^
    -map 0:v:0 -c:v copy -map 0:a:0 -c:a copy "%BASENAME%.mkv"

echo Creating "%INPUT%" proxy "%BASENAME%_proxy.mkv"...
%FFMPEG% -nostdin -v error -i "%BASENAME%.mkv" ^
    -pix_fmt yuv420p ^
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
    -movflags +faststart ^
    -c:v libx265 -preset veryfast -crf 20 -profile:v baseline ^
    -c:a aac -b:a 41.1k -ac 1 -ar 44100 ^
    -metadata "title=Proxy for %BASENAME%.mkv Proxy" ^
    "%BASENAME%_proxy.mkv"

echo Done.

exit /b 0
