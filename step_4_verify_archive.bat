@echo on
REM Verifies all files in the archive against the BLAKE3 checksum manifest

REM --- Set paths ---
set "ARCHIVE_DIR=..\"
set "ARCHIVE_CHECKSUM_FILE=00-archive-manifest-blake3sums.txt"
set "B3SUM_BIN=scripts\bin\b3sum_windows_x64_bin.exe"

echo Verifying: %ARCHIVE_CHECKSUM_FILE%
echo.

REM --- Run b3sum verification ---
cd %ARCHIVE_DIR%
"%B3SUM_BIN%" -c "%ARCHIVE_CHECKSUM_FILE%"
set RETURN_CODE=%ERRORLEVEL%

echo.

IF %RETURN_CODE% EQU 0 (
    echo ALL FILES VERIFIED — CHECKSUMS MATCH!
) ELSE (
    echo SOME FILES FAILED VERIFICATION!
)

exit /b %RETURN_CODE%
