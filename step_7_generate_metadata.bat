@echo off
setlocal enabledelayedexpansion

REM Resolve base directory of script
set "BASE=%~dp0"

REM Paths to possible b3sum locations
set "B3A=%BASE%b3sum_windows_x64_bin.exe"
set "B3B=%BASE%bin\b3sum_windows_x64_bin.exe"

REM Pick whichever exists
if exist "%B3A%" (
    set "B3=%B3A%"
) else if exist "%B3B%" (
    set "B3=%B3B%"
) else (
    echo Error: b3sum not found.
    echo Expected:
    echo   %B3A%
    echo   %B3B%
    exit /b 1
)

REM Check mediainfo
where mediainfo >nul 2>&1
if errorlevel 1 (
    echo Error: mediainfo CLI not found in PATH.
    exit /b 1
)

set OUTBLAKE="00-manifest-blake3sums.txt"

if exist "%OUTBLAKE%" del /f /q "%OUTBLAKE%"

set "FOUND=0"
for %%F in (*.mp4 *.mkv) do (
    if exist "%%F" (
        set "FOUND=1"
        echo.
        echo Processing: %%F

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

echo Generating mediainfo for "%IN%"...

mediainfo --Output=Text "%IN%" > "%BN% mediainfo.txt"
if errorlevel 1 (
    echo Error generating mediainfo for "%IN%"
    endlocal
    exit /b 1
)

endlocal
goto :eof
