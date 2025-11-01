@echo off
setlocal enabledelayedexpansion
set "B3=%~dp0b3sum_windows_x64_bin.exe"

REM Check for required tools
where ffprobe >nul 2>&1
if errorlevel 1 (
    echo Error: ffprobe not found in PATH. Please install ffmpeg.
    exit /b 1
)

if not exist "%B3%" (
    echo Error: b3sum not found at "%B3%"
    echo Please place b3sum_windows_x64_bin.exe in the same folder as this script.
    exit /b 1
)

set "FOUND=0"
for %%F in (*.mp4 *.mkv) do (
    if exist "%%F" (
        set "FOUND=1"
        echo.
        echo Processing: %%F
        call :vhs_info "%%F"
    )
)

if "!FOUND!"=="0" (
    echo No .mp4 or .mkv files found in %CD%
    exit /b 0
)

echo All files processed successfully.
endlocal
exit /b 0

:vhs_info
setlocal
set "IN=%~1"
for %%I in ("%IN%") do set "BN=%%~nI"

REM Generate ffprobe info
echo Generating info for "%IN%"
ffprobe -v quiet -show_format -show_streams "%IN%" > "%BN% info.txt"
if errorlevel 1 (
    echo Error generating text info for "%IN%"
    exit /b 1
)

REM Generate blake3 hash
echo Generating blake3 hash for "%IN%"
"%B3%" "%IN%" > "%BN% blake3.txt"
if errorlevel 1 (
    echo Error generating blake3 hash for "%IN%"
    exit /b 1
)

endlocal
goto :eof
