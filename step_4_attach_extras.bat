@echo off
setlocal enabledelayedexpansion

if "%~3"=="" (
  echo Usage: attach_extras.bat "video.mkv" "cover.jpg" "label.jpg" ["notes.txt"] ["title"] ["comment"] ["chapters.ffmetadata"]
  exit /b 1
)

set "MKV=%~1"
set "COVER=%~2"
set "LABEL=%~3"
set "NOTES=%~4"
set "TITLE=%~5"
set "COMMENT=%~6"
set "CHAPTERS=%~7"

if not exist "%MKV%" (
  echo ERROR: "%MKV%" not found.
  exit /b 1
)
if not exist "%COVER%" (
  echo ERROR: "%COVER%" not found.
  exit /b 1
)

if not exist "%LABEL%" (
  set "LABEL="
) else (
  for %%I in ("%LABEL%") do set "LABEL_EXT=%%~xI"
  set "LABEL_EXT=!LABEL_EXT:~1!"
)

if not exist "%NOTES%" (
  set "NOTES="
) else (
  for %%I in ("%NOTES%") do set "NOTES_EXT=%%~xI"
  set "NOTES_EXT=!NOTES_EXT:~1!"
)

if not exist "%CHAPTERS%" (
  set "CHAPTERS="
)

for %%I in ("%COVER%") do set "COVER_EXT=%%~xI"
set "COVER_EXT=!COVER_EXT:~1!"

rem temp output file
set "TMP=%TEMP%\attach_extras_%RANDOM%.mkv"

rem destination name: basename_updated.mkv
for %%I in ("%MKV%") do set "BASE=%%~nI"
set "DEST=%BASE%_updated.mkv"

echo Updating "%MKV%" ...
echo Tmp: "%TMP%"

rem Use PowerShell to construct argument array and run ffmpeg (handles quoting robustly)
powershell -NoProfile -Command ^
  "$args = @(); " ^
  "$args += '-i'; $args += '%MKV%'; " ^
  "if ('%CHAPTERS%' -ne '') { $args += '-i'; $args += '%CHAPTERS%'; } " ^
  "$args += '-map'; $args += '0'; $args += '-c'; $args += 'copy'; " ^
  "if ('%CHAPTERS%' -ne '') { $args += '-map_metadata'; $args += '1'; } " ^
  "if ('%TITLE%' -ne '') { $args += '-metadata'; $args += \"title=%TITLE%\"; } " ^
  "if ('%COMMENT%' -ne '') { $args += '-metadata'; $args += \"comment=%COMMENT%\"; } " ^
  "$args += '-attach'; $args += '%COVER%'; $args += '-metadata:s:t:0'; $args += 'mimetype=image/jpeg'; $args += '-metadata:s:t:0'; $args += \"filename=cover.%COVER_EXT%\"; " ^
  "if ('%LABEL%' -ne '') { $args += '-attach'; $args += '%LABEL%'; $args += '-metadata:s:t:1'; $args += 'mimetype=image/jpeg'; $args += '-metadata:s:t:1'; $args += \"filename=label.%LABEL_EXT%\"; } " ^
  "if ('%NOTES%' -ne '') { $args += '-attach'; $args += '%NOTES%'; $args += '-metadata:s:t:2'; $args += 'mimetype=text/plain'; $args += '-metadata:s:t:2'; $args += \"filename=notes.%NOTES_EXT%\"; } " ^
  "$args += '-f'; $args += 'matroska'; $args += '%TMP%'; $args += '-y'; " ^
  "Write-Output ('Running: ffmpeg ' + ($args -join ' ')); " ^
  "& ffmpeg @args; exit $LASTEXITCODE"

if errorlevel 1 (
  echo ERROR: ffmpeg failed.
  if exist "%TMP%" del /f /q "%TMP%" >nul 2>&1
  exit /b 1
)

move /y "%TMP%" "%DEST%" >nul 2>&1
if errorlevel 1 (
  echo ERROR: failed to move "%TMP%" to "%DEST%".
  exit /b 1
)

echo Done: "%DEST%"
endlocal
exit /b 0