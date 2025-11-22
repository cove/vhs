@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: process_mkv_all_fix.bat video.mkv
    exit /b 1
)

set "INPUT=%~1"
if not exist "%INPUT%" (
    echo ERROR: %INPUT% not found.
    exit /b 1
)

for %%F in ("%INPUT%") do set "FILENAME=%%~nxF"
for %%F in ("%INPUT%") do set "BASENAME=%%~nF"
set "OUTPUT=%BASENAME%_metadata.mkv"

set "SCRIPT_DIR=%~dp0"
set "FFMPEG=%SCRIPT_DIR%bin\ffmpeg.exe"

:: Extract VIDEO_NAME prefix up to first number (PowerShell)
for /f %%V in ('powershell -noprofile -command ^
    "$b='%BASENAME%'; if($b -match '^([^0-9]*[0-9]+)'){ $matches[1] } else { $b }"') do set "VIDEO_NAME=%%V"

set "META_DIR=%SCRIPT_DIR%media_metadata\%VIDEO_NAME%"

set "COVER=%META_DIR%\cover.jpg"
set "TITLE_FILE=%META_DIR%\title.txt"
set "COMMENT_FILE=%META_DIR%\comment.txt"
set "CHAPTERS=%META_DIR%\chapters.ffmetadata"

:: Validate required metadata files
for %%F in ("%COVER%" "%TITLE_FILE%" "%COMMENT_FILE%" "%CHAPTERS%") do (
    if not exist %%~F (
        echo ERROR: Missing expected metadata file: %%~F
        exit /b 1
    )
)

:: Read first line (keeps spaces). If your title is multi-line, change this.
set "TITLE="
set /p TITLE=<"%TITLE_FILE%"

set "COMMENT="
set /p COMMENT=<"%COMMENT_FILE%"

:: Remove a possible UTF-8 BOM (optional)
if defined TITLE (
    rem remove BOM bytes if present (UTF-8 BOM -> 0xEF 0xBB 0xBF)
    if "!TITLE:~0,1!"=="ï" (
        set "TITLE=!TITLE:~1!"
    )
)

:: Replace double quotes with single quotes to avoid breaking ffmpeg -metadata quoting
set "TITLE_ESC=!TITLE:"='!"
set "COMMENT_ESC=!COMMENT:"='!"

echo Processing "%INPUT%" -> "%OUTPUT%"...
echo Applying VHS tags and attachments...

:: Cover extension lowercase (powershell)
for /f %%E in ('powershell -noprofile -command "(Get-Item -LiteralPath ''%COVER%'').Extension.ToLower().TrimStart('.')"' ) do set "EXT=%%E"

:: Use delayed expansion variables in ffmpeg command
"%FFMPEG%" -nostdin -v error -i "%INPUT%" ^
    -f ffmetadata -i "%CHAPTERS%" ^
    -map 0:v:0 -map 0:a ^
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

echo Done.
echo Output: %OUTPUT%
exit /b 0
