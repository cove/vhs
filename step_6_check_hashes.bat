@echo off
REM check_hashes_b3.bat - Verify BLAKE3 hashes the directory

setlocal enabledelayedexpansion
set "B3=%~dp0b3sum_windows_x64_bin.exe"
if not exist "%B3%" (
  echo b3sum binary not found at "%B3%"
  echo Please place b3sum_windows_x64.exe in the same folder as this script.
  endlocal
  exit /b 1
)

echo Using: "%B3%"
echo.

"%B3%" -c "00-manifest-blake3sums.txt"

exit /b 0
