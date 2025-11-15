@echo off
setlocal

if "%~3"=="" (
    echo Usage: attach_extras "video.mkv" "cover.jpg" "label.jpg" ["notes.txt"]
    exit /b
)

set "MKV=%~1"
set "COVER=%~2"
set "LABEL=%~3"
set "NOTES=%~4"

if not exist "%COVER%" (
    echo ERROR: %COVER% not found.
    exit /b
)

if not exist "%LABEL%" (
    echo WARNING: %LABEL% not found. Skipping label.
)

if not exist "%NOTES%" (
    echo WARNING: %NOTES% not found. Skipping notes.
)

echo Updating "%MKV%" ...

set "TEMP=temp_attach.mkv"
if exist "%TEMP%" del "%TEMP%"

ffmpeg -i "%MKV%" -map 0 -c copy ^
  -attach "%COVER%" -metadata:s:t:0 mimetype=image/jpeg -metadata:s:t:0 filename="%~nx2" ^
  %LABEL:~0,1%==-^ ^( if not "%LABEL%"=="" ffmpeg -i "%MKV%" -map 0 -c copy -attach "%LABEL%" -metadata:s:t:1 mimetype=image/jpeg -metadata:s:t:1 filename="%~nx3" ^) ^
  %NOTES:~0,1%==-^ ^( if not "%NOTES%"=="" ffmpeg -i "%MKV%" -map 0 -c copy -attach "%NOTES%" -metadata:s:t:2 mimetype=text/plain -metadata:s:t:2 filename="%~nx4" ^) ^
  -f matroska "%TEMP%" -y >nul 2>&1

if exist "%TEMP%" (
    move /y "%TEMP%" "%MKV%" >nul
    echo Done.
) else (
    echo ERROR: ffmpeg failed.
)