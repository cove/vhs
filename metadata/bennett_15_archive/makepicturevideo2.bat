"..\..\software\Windows\FFmpeg-QTGMC Easy 2025.01.11\ffmpeg.exe" ^
	-loop 1 -framerate 25 -i bennett_4294_a.jpg -i audio.wav ^
	-filter_complex "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:a]showwaves=s=1920x1080:mode=line:colors=white,format=rgba[waves];[bg][waves]overlay=format=auto" ^
	-c:v libx264 -preset ultrafast -tune stillimage -c:a copy -pix_fmt yuv420p -shortest ^
	-fflags +genpts -start_at_zero -avoid_negative_ts make_zero ..\..\bennett_15_archive2.mkv


