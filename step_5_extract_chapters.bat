@echo off
setlocal enabledelayedexpansion

:: --- Argument check ---
if "%~1"=="" (
    echo Usage: extract_chapters.bat input.mkv [chapter_name]
    exit /b 1
)

set "IN=%~1"
set "CHAPTER_FILTER=%~2"

:: --- Script directory & ffmpeg path ---
set "SCRIPT_DIR=%~dp0"
set "FFMPEG=%SCRIPT_DIR%bin\ffmpeg.exe"

if not exist "%IN%" (
    echo ERROR: Input file not found: %IN%
    exit /b 1
)

if not exist "%FFMPEG%" (
    echo ERROR: ffmpeg not found or not executable: %FFMPEG%
    exit /b 1
)

:: --- Load video filter chain ---
set "VIDEO_FILTER_CHAIN="
set "filters_video=%SCRIPT_DIR%filters_video.cfg"
if not exist "%filters_video%" (
    echo ERROR: video filter file not found: %filters_video%
    exit /b 1
)

for /f "usebackq tokens=* delims=" %%A in (`findstr /r /v "^[ ]*#" "%filters_video%"`) do (
    if not "%%A"=="" (
        if defined VIDEO_FILTER_CHAIN (
            set "VIDEO_FILTER_CHAIN=!VIDEO_FILTER_CHAIN!,%%A"
        ) else (
            set "VIDEO_FILTER_CHAIN=%%A"
        )
    )
)

echo Using video filter chain: %VIDEO_FILTER_CHAIN%

:: --- Load audio filter chain ---
set "AUDIO_FILTER_CHAIN="
set "filters_audio=%SCRIPT_DIR%filters_audio.cfg"
if not exist "%filters_audio%" (
    echo ERROR: audio filter file not found: %filters_audio%
    exit /b 1
)

for /f "usebackq tokens=* delims=" %%A in (`findstr /r /v "^[ ]*#" "%filters_audio%"`) do (
    if not "%%A"=="" (
        if defined AUDIO_FILTER_CHAIN (
            set "AUDIO_FILTER_CHAIN=!AUDIO_FILTER_CHAIN!,%%A"
        ) else (
            set "AUDIO_FILTER_CHAIN=%%A"
        )
    )
)

echo Using audio filter chain: %AUDIO_FILTER_CHAIN%
echo Extracting chapters from %IN%...

:: --- Export chapters metadata ---
set "META=%TEMP%\chapters_ffmeta.txt"
"%FFMPEG%" -nostdin -v error -i "%IN%" -f ffmetadata -y "%META%"

set "START="
set "END="
set "TITLE="

for /f "usebackq tokens=* delims=" %%L in ("%META%") do (
    set "LINE=%%L"

    if "!LINE!"=="[CHAPTER]" (
        call :process_chapter
        set "START="
        set "END="
        set "TITLE="
    )

    echo !LINE! | findstr /b "START=" >nul && (
        set "START=!LINE:START=!"
    )

    echo !LINE! | findstr /b "END=" >nul && (
        set "END=!LINE:END=!"
    )

    echo !LINE! | findstr /b "title=" >nul && (
        set "TITLE=!LINE:title=!"
    )
)

:: Process last chapter
call :process_chapter

echo Chapter extraction complete.
exit /b 0


:: --- Chapter handler ---
:process_chapter
if "%START%"=="" goto :eof
if "%END%"=="" goto :eof
if "%TITLE%"=="" goto :eof

echo %TITLE% | findstr /i "Capture Start Capture End" >nul && goto :eof

if defined CHAPTER_FILTER (
    if /i not "%TITLE%"=="%CHAPTER_FILTER%" goto :eof
)

:: Convert ns ➜ seconds (PowerShell handles decimals cleanly)
for /f %%S in ('powershell -noprofile -command "%START%/1000000000"') do set START_SEC=%%S
for /f %%S in ('powershell -noprofile -command "%END%/1000000000"') do set END_SEC=%%S

:: Safe filename
set "SAFE_TITLE=%TITLE%"
for %%C in (\,/,:,*,?,",<,>,|) do set "SAFE_TITLE=!SAFE_TITLE:%%C=_!"

set "OUT=%SAFE_TITLE%.mp4"

echo Extracting chapter to: %OUT%

"%FFMPEG%" -nostdin -v error -i "%IN%" ^
    -ss "%START_SEC%" -to "%END_SEC%" ^
    -pix_fmt yuv420p ^
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
    -vf "%VIDEO_FILTER_CHAIN%" ^
    -c:v libx265 -preset slow -crf 20 -profile:v main ^
    -af "%AUDIO_FILTER_CHAIN%" ^
    -c:a aac -b:a 41.1k -ac 1 -ar 44100 ^
    -movflags +faststart ^
    -metadata "title=%TITLE%" ^
    -metadata "comment=Extracted chapter from %IN% (video_filter_chain=%VIDEO_FILTER_CHAIN%, audio_filter_chain=%AUDIO_FILTER_CHAIN%)" -y ^
    "%OUT%"

goto :eof
