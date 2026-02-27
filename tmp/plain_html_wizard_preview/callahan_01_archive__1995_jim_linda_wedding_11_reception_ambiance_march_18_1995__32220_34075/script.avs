
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
LoadDLL("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/libfftw3f-3.dll") 
Import("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/Zs_RF_Shared.avsi") 
Import("C:\Users\covec\Videos\vhs\software\Windows\FFmpeg-QTGMC Easy 2025.01.11/QTGMC.avsi") 
FFmpegSource2("C:\Users\covec\Videos\vhs\tmp\plain_html_wizard_preview\callahan_01_archive__1995_jim_linda_wedding_11_reception_ambiance_march_18_1995__32220_34075\repaired_extracted.mkv", atrack=-1) 
chapter_start_frame = 32220
chapter_end_frame = 34075

Import("C:/Users/covec/Videos/vhs/metadata/callahan_01_archive/filter.avs")
c = last
expected_frames = 1855
if (c.FrameCount >= (expected_frames * 2 - 2) && c.FrameCount <= (expected_frames * 2 + 2)) {
    c = c.SelectEven()
}

c = last
c

