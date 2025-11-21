@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ----------------------------------------------
REM Usage: process_mkv_all.bat video.mkv
REM ----------------------------------------------

if "%~1"=="" (
    echo Usage: process_mkv_all.bat video.mkv
    exit /b 1
)

set "MKV=%~1"
if not exist "%MKV%" (
    echo ERROR: File not found: %MKV%
    exit /b 1
)

REM Full script folder
set "SCRIPT_DIR=%~dp0"

REM Extract basename without extension
set "FILENAME=%~nx1"
set "BASENAME=%~n1"

REM Extract prefix up to first number (same logic as sed)
REM Example: bennett_8_metadata_extras -> bennett_8
set "VIDEO_NAME="
set "work=%BASENAME%"

for /l %%A in (0,1,200) do (
    if "!work:~%%A,1!"=="" goto extracted
    set "ch=!work:~%%A,1!"
    for %%D in (0 1 2 3 4 5 6 7 8 9) do (
        if "!ch!"=="%%D" (
            REM Found first number – copy until here (inclusive)
            set /a idx=%%A
            set /a end=idx+1
            set "VIDEO_NAME=!work:~0,%end%!"
            goto extracted
        )
    )
)
:extracted

if "%VIDEO_NAME%"=="" (
    echo ERROR: Could not extract video name prefix from %BASENAME%
    exit /b 1
)

set "META_DIR=%SCRIPT_DIR%media_metadata\%VIDEO_NAME%"

set "COVER=%META_DIR%\cover.jpg"
set "LABEL=%META_DIR%\label.jpg"
set "TITLE_FILE=%META_DIR%\title.txt"
set "COMMENT_FILE=%META_DIR%\comment.txt"
set "CHAPTERS=%META_DIR%\chapters.ffmetadata"

REM Validate expected metadata files
for %%F in ("%COVER%" "%LABEL%" "%TITLE_FILE%" "%COMMENT_FILE%" "%CHAPTERS%") do (
    if not exist %%F (
        echo ERROR: Missing required metadata file: %%F
        exit /b 1
    )
)

REM Read text files
set /p "TITLE="<"%TITLE_FILE%"
set /p "COMMENT="<"%COMMENT_FILE%"

echo Processing "%MKV%" ...
echo Cleaning metadata, removing attachments, applying VHS metadata...

REM ----------------------------------------------
REM TEMP FILES
REM ----------------------------------------------
set "TMP1=%TEMP%\mkv_clean_%RANDOM%.mkv"
set "TMP2=%TEMP%\mkv_final_%RANDOM%.mkv"

if exist "%TMP1%" del "%TMP1%"
if exist "%TMP2%" del "%TMP2%"

REM ----------------------------------------------
REM STEP 1 — CLEAN: remove attachments + wipe metadata + apply VHS metadata
REM ----------------------------------------------

ffmpeg -nostdin -v error -i "%MKV%" ^
  -map 0 ^
  -map -0:t ^
  -c copy ^
  -metadata title="" ^
  -metadata comment="" ^
  -metadata:s:v:0 encoder="" ^
  -metadata:s:a:0 encoder="" ^
  -metadata:s:v:0 field_order="BFF" ^
  -color_primaries:v 6 ^
  -color_trc:v 6 ^
  -colorspace:v 5 ^
  -aspect 4:3 ^
  "%TMP1%"

if errorlevel 1 (
    echo ERROR: ffmpeg failed in cleanup stage.
    exit /b 1
)

echo Adding chapters, cover, label, title, comment...

REM ----------------------------------------------
REM STEP 2 — ADD METADATA + ATTACHMENTS
REM ----------------------------------------------

REM Output file
set "OUT=%BASENAME%_metadata_extras.mkv"

ffmpeg -nostdin -v error ^
  -i "%TMP1%" ^
  -i "%CHAPTERS%" ^
  -map 0 -c copy -map_metadata 1 ^
  -metadata title="%TITLE%" ^
  -metadata comment="%COMMENT%" ^
  -attach "%COVER%" -metadata:s:t:0 mimetype=image/jpeg -metadata:s:t:0 filename="cover.jpg" ^
  -attach "%LABEL%" -metadata:s:t:1 mimetype=image/jpeg -metadata:s:t:1 filename="label.jpg" ^
  -f matroska "%TMP2%" -y

if errorlevel 1 (
    echo ERROR: ffmpeg failed during attachment stage.
    exit /b 1
)

move /y "%TMP2%" "%OUT%" >nul

echo Done.
echo Output: %OUT%

endlocal
exit /b 0