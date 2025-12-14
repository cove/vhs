FFmpeg - QTGMC Easy v2025.01.11
===============================

This package will help you to use QTGMC (in every modes) on any computer with windows 64 bits even if you don't have administrator rights.

You must have some knowledges about : Avisynth-Avisynth+ (http://avisynth.nl/) and FFmpeg (https://ffmpeg.org/).

To use QTGMC with FFmpeg , just unzip all the content of the zip file in a folder (for example : C:\QTGMC), then launch a command prompt or right click on windows flag, then choose "Run" and type "cmd", and go to the directory where
you extracted all the files (If you extracted all the files in C:\QTGMC, type in the command prompt : cd C:\QTGMC). 
You will have to type in the command prompt windows : "ffmpeg -i qtgmc.avs" and then choose the output file and format you wish.

For example : ffmpeg -i qtgmc.avs out.mp4

The qtgmc.avs will be opened by FFmpeg, the video will be post-processed and QTGMC (in Faster mode by default with the script) applied and then encoded to MP4. You can add also other commands related to FFmpeg (resizing, cropping, 
encoding to another format than MP4, etc...).

The file that will be post-processed is in.mp4 which is called in the qtgmc.avs script on line 24 with no sound or on line 26 with sound. If you want to post-processed an other video file, you will have to replace in.mp4 by the name of 
your video file.

If you want to use an other mode for QTGMC than Faster, you will have to edit qtgmc.avs on line 30. 
All the modes availabe are "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster", "Very Fast", "Super Fast", "Ultra Fast" & "Draft".

To keep you informed : https://forum.videohelp.com/threads/405720-FFmpeg-QTGMC-Easy%21

Hunk91

v2025.01.11
===========
In this version, all the plugins, scripts and softwares are updated to their last versions available on 2025.01.11.
LSMAHSource was added as well as all Avisynth's original plugins for more versatility and fleibility
List of all the plugings , scripts and softwares :
- FFmpegSource v5.0 plugin : http://avisynth.nl/index.php/FFMS2
- LSMASHSource v20240408 1194.0.0.0 : http://avisynth.nl/index.php/LSMASHSource
- MaskTools2 v2.2.30 plugin : http://avisynth.nl/index.php/Masktools2
- Rgtools v1.2 plugin : http://avisynth.nl/index.php/RgTools
- MVTools2 v2.7.46 with depans v20240503 plugin : http://avisynth.nl/index.php/MVTools (depans.Dll and DepansEstimate.Dll have been added to the package but may not be needed as only mvtools2.dll is called)
- Nnedi3 v0.9.4.63 Plugin : http://avisynth.nl/index.php/Nnedi3
- Yadifmod2 v0.2.8 Plugin : http://avisynth.nl/index.php/Yadifmod2
- FFT3DFilter v2.10 Plugin : http://avisynth.nl/index.php/FFT3DFilter
- LoadDLL v1.0 Plugin : http://avisynth.nl/index.php/LoadDLL
- Zs_RF_Shared v1.161 script : http://avisynth.nl/index.php/Zs_RF_Shared
- QTGMC v3.384s script : http://avisynth.nl/index.php/QTGMC
- DevIL v0.1.7.8 Dll
- libfftw3f-3.dll (fftw v3.3.5) : https://www.fftw.org/
- AviSynth+ 3.7.3(r4003) 64 bits : https://avs-plus.net/
- FFmpeg 7.1-full_build-www.gyan.dev 2024-09-30 : https://www.gyan.dev/ffmpeg/builds/

v2022.12.27
===========
In this version, all the plugins, scripts and software are the same as previous version (2022.11.13).
Only Nnedi3 was updated to its last version (v0.9.4.61).
LoadDll (v10.7) was added in order to use libfftw3f-3.dll (fftw v3.3.5), and also FFT3DFilter (v2.10) was added in order to use QTGMC in Very Slow and Placebo modes.

List of all the plugings , scripts and software :
- FFmpegSource v2.40 plugin : http://avisynth.nl/index.php/FFMS2
- MaskTools2 v2.2.30 plugin : http://avisynth.nl/index.php/Masktools2
- Rgtools v1.2 plugin : http://avisynth.nl/index.php/RgTools
- MVTools v2.7.45 with depans v20210608 plugin : http://avisynth.nl/index.php/MVTools (depans.Dll and DepansEstimate.Dll have been added to the package but may not be needed as only mvtools2.dll is called)
- Nnedi3 v0.9.4.61 Plugin : http://avisynth.nl/index.php/Nnedi3
- Yadifmod2 v0.2.7 Plugin : http://avisynth.nl/index.php/Yadifmod2
- Zs_RF_Shared v1.159 script : http://avisynth.nl/index.php/Zs_RF_Shared
- QTGMC v3.383s script : http://avisynth.nl/index.php/QTGMC
- DevIL v0.1.7.8 Dll
- libfftw3f-3.dll (fftw v3.3.5) : https://www.fftw.org/
- AviSynth+ 3.7.2(r3661) 64 bits : https://avs-plus.net/
- FFmpeg 5.1.2-full_build-www.gyan.dev 2022-09-26 : https://www.gyan.dev/ffmpeg/builds/

The qtgmc.avs script has been updated :
Loading of FFT3DFilter plugin was added on line 20, and loading of LoadDll plugin was added on line 22 (in order to load libfftw3f-3.dll on line 24). Those plugins and Dll are needed if you want to use "Very Slow" or "Placebo"modes.
If you don't use those modes, just add a # at beguinning of lines 20, 22 and 24.
Thanks to Aaron1 for pointing those issues ( https://forum.videohelp.com/threads/405720-FFmpeg-QTGMC-Easy!#post2676281 ) and thanks to jagabo ( https://forum.videohelp.com/threads/405720-FFmpeg-QTGMC-Easy!#post2676282 ) and Selur
( https://forum.videohelp.com/threads/405720-FFmpeg-QTGMC-Easy!#post2676296 ) for the informations and solutions.

v2022.11.13
===========
In this version, all the plugins, scripts and software are the same as the previous version (2022.11.05).
Only Yadimod2 is added to the pack as QTGMC needs it when set to "Ultra Fast". The calling script qtgmc.avs was updated accordingly.

List of all the plugings , scripts and software :
- FFmpegSource v2.40 plugin : http://avisynth.nl/index.php/FFMS2
- MaskTools2 v2.2.30 plugin : http://avisynth.nl/index.php/Masktools2
- Rgtools v1.2 plugin : http://avisynth.nl/index.php/RgTools
- MVTools v2.7.45 with depans v20210608 plugin : http://avisynth.nl/index.php/MVTools (depans.Dll and DepansEstimate.Dll have been added to the package but may not be needed as only mvtools2.dll is called)
- Nnedi3 v0.9.4.60 Plugin : http://avisynth.nl/index.php/Nnedi3
- Yadifmod2 v0.2.7 Plugin : http://avisynth.nl/index.php/Yadifmod2
- Zs_RF_Shared v1.159 script : http://avisynth.nl/index.php/Zs_RF_Shared
- QTGMC v3.383s script : http://avisynth.nl/index.php/QTGMC
- DevIL v0.1.7.8 Dll 
- AviSynth+ 3.7.2(r3661) 64 bits
- FFmpeg 5.1.2-full_build-www.gyan.dev 2022-09-26

The qtgmc.avs script has been updated :
Loading of Yadifmod2 plugin was added on line 18. This plugin is needed if you want to use "Ultra Fast" mode of QTGMC. If you don't use this mode, just add a # at beguinning of line 18.
Thanks to RogerTango for pointing this issue ( https://forum.videohelp.com/threads/405720-FFmpeg-QTGMC-Easy!#post2672390 )

v2022.11.05
===========
In this version, all the plugins, scripts and software have been updated to their latest versions. For most of the plugins, it is the same version as the ones inside v2022.05.12.
Only QTGMC script, Avisynth+ and FFmpeg have been updated to their last versions aswell as the calling script qtgmc.avs .
Script Zs_RF_Shared.avsi is now needed and have been added to the package as the lastest version of QTGMC needs it.
List of all the plugings , scripts and software :
- FFmpegSource v2.40 plugin : http://avisynth.nl/index.php/FFMS2
- MaskTools2 v2.2.30 plugin : avisynth.nl/index.php/Masktools2
- Rgtools v1.2 plugin : http://avisynth.nl/index.php/RgTools
- MVTools v2.7.45 with depans v20210608 plugin : http://avisynth.nl/index.php/MVTools (depans.Dll and DepansEstimate.Dll have been added to the package but may not be needed as only mvtools2.dll is called)
- Nnedi3 v0.9.4.60 Plugin : http://avisynth.nl/index.php/Nnedi3
- Zs_RF_Shared v1.159 script : http://avisynth.nl/index.php/Zs_RF_Shared
- QTGMC v3.383s script : http://avisynth.nl/index.php/QTGMC
- DevIL v0.1.7.8 Dll 
- AviSynth+ 3.7.2(r3661) 64 bits
- FFmpeg 5.1.2-full_build-www.gyan.dev 2022-09-26

The qtgmc.avs script has been updated :
On line 24, you need to open a video file with an audio track. If you only need to deinterlace a video file without an audio track, just add a # at the beguinning of line 24 and remove the # at the beguinning of line 22 to activate it. 
Thanks to c57d for letting me know! ( https://forum.videohelp.com/threads/405720-FFmpeg-QTGMC-Easy!#post2656704 )

The example video file named in.mp4 has now a silent audio track, which was not the case in the previous package.

v2022.05.12
===========
First version of the package.

- FFmpegSource v2.40 plugin : http://avisynth.nl/index.php/FFMS2
- MaskTools2 v2.2.30 plugin : avisynth.nl/index.php/Masktools2
- Rgtools v1.2 plugin : http://avisynth.nl/index.php/RgTools
- MVTools v2.7.45 : http://avisynth.nl/index.php/MVTools (depans.Dll and DepansEstimate.Dll have been removed from the package)
- Nnedi3 v0.9.4.60 Plugin : http://avisynth.nl/index.php/Nnedi3
- SMDegrain v3.1.2d script : http://avisynth.nl/index.php/SMDegrain
- QTGMC 3.358 script : http://avisynth.nl/index.php/QTGMC
- DevIL v0.1.7.8 Dll
- Avisynth+ 3.5(r3106) 64 bits
- FFmpeg 2022-05-08-git-f77ac5131c-full_build
