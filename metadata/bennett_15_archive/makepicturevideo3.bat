"..\..\software\Windows\FFmpeg-QTGMC Easy 2025.01.11\ffmpeg.exe" ^
	-loop 1 -framerate 25 -i bennett_4294_a.jpg -i audio.wav ^
	-filter_complex "[1:a]channelsplit=channel_layout=stereo[L][R];[L]showwaves=s=1920x540:mode=line:colors=white:rate=25:scale=sqrtleftWave];[R]showwaves=s=1920x540:mode=line:colors=white:rate=25:scale=sqrt[rightWave][0:v]setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg][leftWave]overlay=0:0[tmp];[tmp][rightWave]overlay=0:540" ^
	-c:v libx264 -preset ultrafast -tune stillimage -c:a copy -pix_fmt yuv420p -shortest ^
	-fflags +genpts -start_at_zero -avoid_negative_ts make_zero ..\..\bennett_15_archive.mkv


