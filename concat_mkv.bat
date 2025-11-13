@echo off
REM concat_mkv.bat - Concatenate two MKV files using ffmpeg
REM Usage: concat_mkv.bat input1.mkv input2.mkv [output.mkv]

n:: Check args
if "%~1"=="" goto usage
if "%~2"=="" goto usage

nset "IN1=%~1"
set "IN2=%~2"
if "%~3"=="" (
  set "OUT=%~n1_%~n2_merged.mkv"
) else (
  set "OUT=%~3"
)

rem Require ffmpeg on PATH
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo Error: ffmpeg not found in PATH. Install ffmpeg or add it to PATH.
  exit /b 1
)

rem Create temporary file list for concat demuxer
set "LIST=%TEMP%\ff_concat_%RANDOM%.txt"
necho file '%~f1' > "%LIST%"
echo file '%~f2' >> "%LIST%"

echo Concatenating:
echo   %IN1%
echo   %IN2%
echo -> %OUT%

rem Try stream copy (fast, no re-encode)
ffmpeg -hide_banner -loglevel info -f concat -safe 0 -i "%LIST%" -c copy "%OUT%"
if errorlevel 1 (
    exit /b 1
)

del "%LIST%" >nul 2>&1
echo Done: %OUT%
exit /b 0

n:usage
echo Usage: %~nx0 input1.mkv input2.mkv [output.mkv]
exit /b 1
