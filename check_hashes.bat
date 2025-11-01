@echo off
REM check_hashes_b3.bat - Verify BLAKE3 hashes for .mp4 and .mkv files using b3sum_windows_x64.exe in the script directory

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
set "FOUND=0"
for %%F in (*.mp4 *.mkv) do (
  if exist "%%F" (
    set "FOUND=1"
    echo --------------------------------------------------
    echo Checking "%%~nF"
    if exist "%%~nF blake3.txt" (
      "%B3%" -c "%%~nF blake3.txt"
      if errorlevel 1 (
        echo Hash check FAILED for "%%~nF"
      ) else (
        echo Hash OK for "%%~nF"
      )
    ) else (
      echo Missing hash file: "%%~nF blake3.txt"
    )
  )
)

if "!FOUND!"=="0" (
  echo No .mp4 or .mkv files found in %CD%.
)


exit /b 0
