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

echo Creating FFv1 archive "%BASENAME%_archive.mkv"...
%FFMPEG% -nostdin -v error -i "%BASENAME%.avi" ^
    -pix_fmt yuv422p ^
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
    -map 0:v:0 -c:v ffv1 ^
        -level 3 ^
        -g 1 ^
        -coder 1 ^
        -context 1 ^
        -slices 24 ^
        -slicecrc 1 ^
    -map 0:a:0 -c:a pcm_s16le ^
    -y "%BASENAME%_archive.mkv"

@REM echo Creating mp4 proxy "%BASENAME%_proxy.mp4"...
@REM %FFMPEG% -nostdin -v error -i "%BASENAME%_archive.mkv" ^
@REM     -pix_fmt yuv422p ^
@REM     -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
@REM     -movflags +faststart ^
@REM     -c:v libx265 -preset veryfast -crf 28 ^
@REM     -c:a aac -b:a 41.1k -ac 1 -ar 48000 ^
@REM     -metadata "title=Proxy for %BASENAME%_archive.mkv" ^
@REM     -y "%BASENAME%_proxy.mp4"

echo Done.

exit /b 0
