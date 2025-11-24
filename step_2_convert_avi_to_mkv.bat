@echo off
REM Usage: step_2_convert_avi_to_mkv.bat captured.mkv

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

REM Loop over every argument (each should be an .avi file)
for %%I in (%*) do (
    set "INPUT=%%~I"
    set "FILENAME=%%~nxI"
    set "BASENAME=%%~nI"

    echo Creating MVK encoded as FFv1 archive "%%~nI_archive.mkv"...

    "%FFMPEG%" -nostdin -v error -i "%%~I" ^
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
        -y "%%~nI_archive.mkv"

    echo Done with %%~I
    echo.
)

echo Done.

exit /b 0
