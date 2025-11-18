@echo off
REM Modern, high-quality version that uses modern color correction and deinterlacing and audio notch for VHS camera mechanical noise
REM Usage: step_6_make_modern_mp4.bat input_file.mkv

if "%~1"=="" (
    echo Usage: %~nx0 input_file.mkv
    exit /b 1
)

set "INPUT=%~1"
for %%I in ("%INPUT%") do set "FILENAME=%%~nxI"
for %%I in ("%INPUT%") do set "BASENAME=%%~nI"

echo Processing "%INPUT%"...
ffmpeg -i "%INPUT%" ^
  -vf "setfield=bff,yadif=1,zscale=matrixin=170m:matrix=170m,eq=saturation=0.9:gamma=1.0,crop=in_w-2:in_h-6:0:0,hqdn3d=0:0:3:3,scale=640:480,setsar=1" ^
  -c:v libx264 -preset veryslow -crf 16 -pix_fmt yuv420p ^
  -flags +ilme+ildct ^
  -crf 20 -profile:v main ^
  -c:a aac -b:a 41.1k -ac 1 -ar 44100 ^
  -af "firequalizer=gain='if(between(f,58,62),-6,0)'" ^
  -movflags +faststart ^
  "%BASENAME% Modern.mp4"

echo
echo "Done!"
echo "Output:"
echo "   %BASENAME% Modern.mp4"
