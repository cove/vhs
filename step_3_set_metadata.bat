@echo off
setlocal enabledelayedexpansion

:: --- Argument check ---
if "%~1"=="" (
    echo Usage: process_mkv_all.bat video.mkv
    exit /b 1
)

set "INPUT=%~1"
if not exist "%INPUT%" (
    echo ERROR: %INPUT% not found.
    exit /b 1
)

:: --- Basename/output ---
for %%F in ("%INPUT%") do set "FILENAME=%%~nxF"
for %%B in ("%INPUT%") do set "BASENAME=%%~nB"
set "OUTPUT=%BASENAME%_metadata.mkv"

:: --- Script directory & ffmpeg path ---
set "SCRIPT_DIR=%~dp0"
set "FFMPEG=%SCRIPT_DIR%bin\ffmpeg.exe"

:: --- Extract VIDEO_NAME prefix up to first number ---
for /f %%V in ('powershell -noprofile -command ^
    "$b='%BASENAME%'; if($b -match '^([^0-9]*[0-9]+)'){ $matches[1] } else { $b }"') do set "VIDEO_NAME=%%V"

set "META_DIR=%SCRIPT_DIR%media_metadata\%VIDEO_NAME%"

set "COVER=%META_DIR%\cover.jpg"
set "TITLE_FILE=%META_DIR%\title.txt"
set "COMMENT_FILE=%META_DIR%\comment.txt"
set "CHAPTERS=%META_DIR%\chapters.ffmetadata"

:: --- Validate required metadata files ---
for %%F in ("%COVER%" "%TITLE_FILE%" "%COMMENT_FILE%" "%CHAPTERS%") do (
    if not exist %%F (
        echo ERROR: Missing expected metadata file: %%F
        exit /b 1
    )
)

:: --- Read title/comment ---
for /f "usebackq delims=" %%A in ("%TITLE_FILE%") do set "TITLE=%%A"
for /f "usebackq delims=" %%A in ("%COMMENT_FILE%") do set "COMMENT=%%A"

echo Processing "%INPUT%" -> "%OUTPUT%"...
echo Applying VHS tags and attachments...

:: --- Lowercase extension of cover file ---
for /f %%E in ('powershell -noprofile -command "(Get-Item '%COVER%').Extension.ToLower().TrimStart('.')"') do set "EXT=%%E"

:: --- Run ffmpeg ---
"%FFMPEG%" -nostdin -v error -i "%INPUT%" ^
    -f ffmetadata -i "%CHAPTERS%" ^
    -map 0:v:0 -map 0:a ^
    -map_metadata 0 ^
    -map_chapters -1 ^
    -map_chapters 1 ^
    -c copy ^
    -metadata title="%TITLE%" ^
    -metadata comment="%COMMENT%" ^
    -attach "%COVER%" ^
    -metadata:s:t:0 mimetype=image/jpeg ^
    -metadata:s:t:0 filename="cover.%EXT%" ^
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -aspect 4:3 ^
    -f matroska "%OUTPUT%" -y

echo Done.
echo Output: %OUTPUT%
exit /b 0
