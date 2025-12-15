@echo off
REM Verifies all files on the drive against the BLAKE3 checksum manifest

REM --- Set paths ---
set "DRIVE_DIR=..\..\"
set "DRIVE_CHECKSUM_FILE=Archive\00-drive-manifest-blake3sums.txt"
set "B3SUM_BIN=Archive\scripts\bin\b3sum_windows_x64_bin.exe"

echo Verifying: %DRIVE_CHECKSUM_FILE%
echo.

REM --- Run b3sum verification ---
cd /d "%DRIVE_DIR%" || (
    echo Failed to enter directory %DRIVE_DIR%
    exit /b 1
)

"%B3SUM_BIN%" -c "%DRIVE_CHECKSUM_FILE%"
set RETURN_CODE=%ERRORLEVEL%

echo.

IF %RETURN_CODE% EQU 0 (
    echo ALL FILES VERIFIED — CHECKSUMS MATCH!
) ELSE (
    echo SOME FILES FAILED VERIFICATION!
)

echo Verify manifest: %DRIVE_CHECKSUM_FILE%
echo All done.

exit /b %RETURN_CODE%
