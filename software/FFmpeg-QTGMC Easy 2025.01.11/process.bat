@echo off
set "INPUT=%%~I"
set "FILENAME=%%~nxI"
set "BASENAME=%%~nI"

.\ffmpeg.exe -nostdin -v error ^
        -i qtgmc.avs ^
        -pix_fmt yuv422p ^
        -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 ^
        -map 0:v:0 -c:v ffv1 ^
            -level 3 ^
            -g 1 ^
            -coder 1 ^
            -context 1 ^
            -slices 24 ^
            -slicecrc 1 ^
        -map 0:a:0 -c:a pcm_s16le ^
        -y "%%~nI_qtgmc.mkv"
