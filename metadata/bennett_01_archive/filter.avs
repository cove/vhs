AssumeFPS(30000,1001)
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3)
Crop(4,2,-8,-10)
LanczosResize(640,480)
ConvertToYV12(interlaced=false)
# after deinterlace / capture / conversion
SmoothLevels(16, 1.0, 255, 16, 235, limiter=1, tvrange=true, dither=0)
Limiter(min_luma=16, max_luma=235, min_chroma=16, max_chroma=240)
Tweak(sat=0.85)

