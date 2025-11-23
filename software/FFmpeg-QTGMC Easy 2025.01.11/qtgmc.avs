#QTGMC portable script - Faster mode
#Tested with Avisynth+ 3.7.3(r4003) 64 bits and FFMPEG 7.1-full_build-www.gyan.dev 2024-09-30 
#To know more about AviSynth go to : http://avisynth.nl/
#To know more about FFmpeg go to : https://ffmpeg.org/

#FFmpegSource v5.00 plugin : http://avisynth.nl/index.php/FFMS2
#To know more bout loading video files in Avisynth : http://avisynth.nl/index.php/FAQ_loading_clips
LoadPlugin("ffms2.dll")
#MaskTools2 v2.2.30 plugin : http://avisynth.nl/index.php/Masktools2
LoadPlugin("masktools2.dll")
#Rgtools v1.2 plugin : http://avisynth.nl/index.php/RgTools
LoadPlugin("Rgtools.dll")
#MVTools v2.7.46 with depans v20240503 plugin : http://avisynth.nl/index.php/MVTools
LoadPlugin("mvtools2.dll")
#Nnedi3 v0.9.4.63 Plugin : http://avisynth.nl/index.php/Nnedi3
LoadPlugin("nnedi3.dll")
#Yadifmod2 v0.2.8 Plugin (ONLY for Ultra Fast setting mode in QTGMC) : http://avisynth.nl/index.php/Yadifmod2
LoadPlugin("yadifmod2.dll")
#FFT3DFilter v2.10 Plugin (ONLY for Very Slow and Placebo modes) : http://avisynth.nl/index.php/FFT3DFilter
LoadPlugin("fft3dfilter.dll")
#LoadDLL v1.0 Plugin (ONLY for Very Slow and Placebo modes) : http://avisynth.nl/index.php/LoadDLL
LoadPlugin("LoadDLL64.dll")
#Loading libfftw3f-3.dll from FFTW v3.3.5(ONLY for Very Slow and Placebo modes) : https://www.fftw.org/
LoadDLL("libfftw3f-3.dll")
#Zs_RF_Shared v1.159 script : http://avisynth.nl/index.php/Zs_RF_Shared
Import("Zs_RF_Shared.avsi")
#QTGMC v3.384s script : http://avisynth.nl/index.php/QTGMC
Import("QTGMC.avsi")
#Opening of video file (the video file opened here is "in.mp4" and it has NO audio tack, you must replace "in.mp4" by the name of your video file)
#FFMpegSource2("in.mp4")
#Opening of video file (the video file opened here is "in.mp4" and it has an audio track, you must replace "in.mp4" by the name of your video file)
FFMpegSource2("in.mp4",atrack=1)
#Converion to YV12 color format: http://avisynth.nl/index.php/ConvertToYV12
ConvertToYV12()
#Applying QTGMC in Faster mode
QTGMC(preset="Faster")