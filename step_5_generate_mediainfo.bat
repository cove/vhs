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

set OUTBLAKE="00-manifest-blake3sums.txt"

REM remove any existing combined file
if exist "%OUTBLAKE%" del /f /q "%OUTBLAKE%"

set "FOUND=0"
for %%F in (*.mp4 *.mkv) do (
    if exist "%%F" (
        set "FOUND=1"
        echo.
        echo Processing: %%F

        REM append blake3 hash (full path preserved by b3sum output)
        "%B3%" "%%F" >> "%OUTBLAKE%"
        if errorlevel 1 (
            echo Error generating blake3 for "%%F"
            exit /b 1
        )

        call :vhs_mediainfo "%%F"
    )
)

if "!FOUND!"=="0" (
    echo No .mp4 or .mkv files found in %CD%
    exit /b 0
)

echo.
echo Combined blake3 file: "%OUTBLAKE%"
echo All files processed successfully.
endlocal
exit /b 0

:vhs_mediainfo
setlocal
set "IN=%~1"
for %%I in ("%IN%") do set "BN=%%~nI"

REM Generate ffprobe mediainfo (text)
echo Generating mediainfo for "%IN%"
ffprobe -v quiet -show_format -show_streams "%IN%" > "%BN% mediainfo.txt"
if errorlevel 1 (
    echo Error generating text mediainfo for "%IN%"
    endlocal
    exit /b 1
)

endlocal
goto :eof
