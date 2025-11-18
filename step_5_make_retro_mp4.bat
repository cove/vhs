@echo off
REM Retro, VHS-style MP4 conversion that tries to be faithful to original VHS look
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: step_5_make_retro_mp4.bat input.mkv
    exit /b 1
)

set "IN=%~1"
for %%I in ("%IN%") do set "BASE=%%~nI"

echo.
echo Creating retro MP4...
ffmpeg -i "%IN%" ^
  -vf "setfield=bff,yadif=0,colorspace=all=5:iall=5,eq=saturation=0.90:gamma=1.0,zscale=matrixin=170m:matrix=170m,crop=in_w-2:in_h-6:0:0,scale=640:480,setsar=1" ^
  -pix_fmt yuv420p ^
  -flags +ilme+ildct ^
  -map_metadata 0:g -metadata encoder="" ^
  -c:v libx264 -preset slow -crf 20 -profile:v baseline ^
  -c:a aac -b:a 41.1k -ac 1 -ar 44100 ^
  -movflags +faststart ^
  "%BASE% Retro.mp4"

echo.
echo Done!
echo Output:
echo   %BASE% Retro.mp4
