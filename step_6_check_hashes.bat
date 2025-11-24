@echo off
REM Verify BLAKE3 hashes the directory

setlocal enabledelayedexpansion
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
echo Verifying BLAKE3 hashes against manifest "00-manifest-blake3sums.txt"...
"%B3%" -c "00-manifest-blake3sums.txt"

exit /b 0
