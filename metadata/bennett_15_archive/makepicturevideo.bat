"..\..\software\Windows\FFmpeg-QTGMC Easy 2025.01.11\ffmpeg.exe" ^
 	-loop 1 -framerate 25 -i bennett_4294_a.jpg -i ..\..\..\bennett_15_archive_cleaned.flac ^
	-vf "setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" ^
 	-c:v libx264 -preset ultrafast -tune stillimage -pix_fmt yuv420p -shortest ^
	-c:a copy ^
 	-fflags +genpts -start_at_zero -avoid_negative_ts make_zero ..\..\..\bennett_15_archive.mkv



