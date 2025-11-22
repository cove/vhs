@echo off
setlocal enabledelayedexpansion

:: -------------------------
:: Usage: process_mkv_all_chapters_fix.bat video.mkv
:: -------------------------
if "%~1"=="" (
  echo Usage: %~nx0 video.mkv
  exit /b 1
)
set "INPUT=%~1"
if not exist "%INPUT%" (
  echo ERROR: %INPUT% not found.
  exit /b 1
)

for %%F in ("%INPUT%") do set "BASENAME=%%~nF"
set "OUTPUT=%BASENAME%_metadata.mkv"
set "SCRIPT_DIR=%~dp0"
set "FFMPEG=%SCRIPT_DIR%bin\ffmpeg.exe"

:: extract VIDEO_NAME prefix up to first number (PowerShell)
for /f %%V in ('powershell -noprofile -command ^
  "$b='%BASENAME%'; if($b -match '^([^0-9]*[0-9]+)'){ $matches[1] } else { $b }"') do set "VIDEO_NAME=%%V"

set "META_DIR=%SCRIPT_DIR%media_metadata\%VIDEO_NAME%"
set "COVER=%META_DIR%\cover.jpg"
set "TITLE_FILE=%META_DIR%\title.txt"
set "COMMENT_FILE=%META_DIR%\comment.txt"
set "CHAPTERS=%META_DIR%\chapters.ffmetadata"

:: validate metadata files
for %%F in ("%COVER%" "%TITLE_FILE%" "%COMMENT_FILE%" "%CHAPTERS%") do (
  if not exist "%%~F" (
    echo ERROR: Missing expected metadata file: %%~F
    exit /b 1
  )
)

:: read title/comment (first line)
set "TITLE="
set /p TITLE=<"%TITLE_FILE%"
set "COMMENT="
set /p COMMENT=<"%COMMENT_FILE%"

:: strip UTF-8 BOM if present (first char ï)
if defined TITLE if "!TITLE:~0,1!"=="ï" set "TITLE=!TITLE:~1!"
if defined COMMENT if "!COMMENT:~0,1!"=="ï" set "COMMENT=!COMMENT:~1!"

:: escape double-quotes (replace with single-quote)
set "TITLE_ESC=!TITLE:"='!"
set "COMMENT_ESC=!COMMENT:"='!"

:: ensure ffmpeg exists
if not exist "%FFMPEG%" (
  echo ERROR: ffmpeg not found at %FFMPEG%
  exit /b 1
)

:: Create a safe, ffmpeg-friendly chapters file: UTF8 no-BOM, LF line endings
set "TMP_CHAPTERS=%TEMP%\chapters_%RANDOM%.ffmetadata"
powershell -noprofile -command ^
  "Get-Content -LiteralPath '%CHAPTERS%' -Raw | `
   %[System.Text.RegularExpressions.Regex]::Replace($_, '\r\n', \"`n\") | `
   Out-File -LiteralPath '%TMP_CHAPTERS%' -Encoding utf8NoBOM"

if not exist "%TMP_CHAPTERS%" (
  echo ERROR: Failed to prepare chapters file.
  exit /b 1
)

:: get cover extension lowercased
for /f %%E in ('powershell -noprofile -command "(Get-Item -LiteralPath '%COVER%').Extension.ToLower().TrimStart('.')"') do set "EXT=%%E"

echo Processing "%INPUT%" -> "%OUTPUT%"...
echo Using chapters tmp: %TMP_CHAPTERS%

:: Run ffmpeg: note inputs order matters (0 = original, 1 = chapters metadata)
"%FFMPEG%" -nostdin -v error -i "%INPUT%" -f ffmetadata -i "%TMP_CHAPTERS%" ^
  -map 0:v:0 -map 0:a? ^
  -map_metadata 0 ^
  -map_chapters -1 ^
  -map_chapters 1 ^
  -c copy ^
  -metadata title="!TITLE_ESC!" ^
  -metadata comment="!COMMENT_ESC!" ^
  -attach "%COVER%" ^
  -metadata:s:t:0 mimetype=image/jpeg ^
  -metadata:s:t:0 filename="cover.%EXT%" ^
  -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -aspect 4:3 ^
  -f matroska "%OUTPUT%" -y

set "FFMPEG_EXIT=%ERRORLEVEL%"

:: cleanup temp chapters file
if exist "%TMP_CHAPTERS%" del /f /q "%TMP_CHAPTERS%"

if "%FFMPEG_EXIT%"=="0" (
  echo Done.
  echo Output: %OUTPUT%
  exit /b 0
) else (
  echo ffmpeg returned %FFMPEG_EXIT%.
  exit /b %FFMPEG_EXIT%
)
