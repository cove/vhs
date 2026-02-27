
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/ffms2.dll") 
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/masktools2.dll") 
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/Rgtools.dll") 
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/mvtools2.dll") 
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/DePanEstimate.dll")
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/DePan.dll")
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/nnedi3.dll") 
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/yadifmod2.dll") 
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/fft3dfilter.dll") 
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/LoadDLL64.dll")
LoadPlugin("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/SmoothAdjust.dll")
LoadPlugin("C:/Users/covec/Videos/vhs/software/Windows/FFmpeg-QTGMC Easy 2025.01.11/Descratch64.dll")
LoadDLL("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/libfftw3f-3.dll") 
Import("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/Zs_RF_Shared.avsi") 
Import("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/QTGMC.avsi") 
FFmpegSource2("C:\Users\covec\Videos\vhs\tmp\vhs_tuner_extracts\callahan_01_archive__1995_jim_linda_wedding_11_reception_wedding_cake_with_gloria_beverly_march_18_1995__34077_34504__clean\extracted.mkv", atrack=-1) 
chapter_start_frame = 34077
chapter_end_frame = 34504

Import("C:/Users/covec/Videos/vhs/metadata/callahan_01_archive/filter.avs")
c = last
expected_frames = 427
if (c.FrameCount >= (expected_frames * 2 - 2) && c.FrameCount <= (expected_frames * 2 + 2)) {
    c = c.SelectEven()
}

c = last
c

