REM ...existing code...
@echo off
REM Run in directory with .mp4/.mkv and corresponding "basename blake3.txt"

setlocal enabledelayedexpansion

REM Iterate over mp4 and mkv files (if none exist, patterns are skipped)
for %%F in (*.mp4 *.mkv) do (
    call :process "%%~fF"
)

endlocal
exit /b 0

:process
REM %1 = full path to media file
set "FULL=%~1"
for %%I in ("%FULL%") do set "BASENAME=%%~nI"
set "BLAKE3_FILE=%BASENAME% blake3.txt"

echo Processing: %FULL%

REM --- BLAKE3 verification ---
if exist "%BLAKE3_FILE%" (
    REM read expected hash (first token)
    for /f "usebackq tokens=1" %%H in ("%BLAKE3_FILE%") do set "EXPECTED_B3=%%H"

    REM compute actual hash using available tool
    where b3sum >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=1" %%X in ('b3sum "%FULL%"') do set "ACTUAL_B3=%%X"
    ) else (
        where blake3 >nul 2>&1
        if !errorlevel! equ 0 (
            for /f "tokens=1" %%X in ('blake3 "%FULL%"') do set "ACTUAL_B3=%%X"
        ) else (
            set "ACTUAL_B3="
        )
    )

    if defined ACTUAL_B3 (
        set "EXPECTED_B3=!EXPECTED_B3: =!"
        set "ACTUAL_B3=!ACTUAL_B3: =!"
        if /i "!EXPECTED_B3!"=="!ACTUAL_B3!" (
            echo BLAKE3 OK: %FULL%
        ) else (
            echo BLAKE3 MISMATCH: %FULL%
            echo Expected: !EXPECTED_B3!
            echo Actual:   !ACTUAL_B3!
        )
    ) else (
        echo Blake3 tool not found; cannot verify %FULL%
    )
) else (
    echo No "%BLAKE3_FILE%" found for %BASENAME%
)

echo.
endlocal
goto :eof