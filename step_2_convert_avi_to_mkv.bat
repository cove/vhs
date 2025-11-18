@echo off
REM Usage: step_2_convert_avi_to_mkv.bat captured.avi

if "%~1"=="" (
    echo Usage: %~nx0 captured.avi
    exit /b 1
)

set "INPUT=%~1"
for %%I in ("%INPUT%") do set "FILENAME=%%~nxI"
for %%I in ("%INPUT%") do set "BASENAME=%%~nI"

echo Generating videos from "%INPUT%"...

REM If the input is FFV1 then use copy, else use re-encode to FFV1
REM ffmpeg -i "%INPUT%" -pix_fmt yuv422p -map 0:v:0 -c:v ffv1 -level 3 -g 1 -coder 1 -context 1 -slicecrc 1 -timecode 00:00:00:00 -map 0:a:0 -c:a flac -segment_time_metadata 1 "%BASENAME%.mkv"
ffmpeg -i "%INPUT%" -map 0:v:0 -c:v copy -map 0:a:0 -c:a copy "%BASENAME%.mkv"

exit /b 0
