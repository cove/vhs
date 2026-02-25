AssumeFPS(30000,1001)
AssumeBFF()
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Slow",SourceMatch=3,Lossless=2,TR1=1,TR2=1,Rep0=2,Rep2=1,MatchEnhance=0.7,MatchEdi="NNEDI3",MatchEdi2="NNEDI3",Sharpness=0.35,SLMode=1,FPSDivisor=2)
# stabilization disabled
Crop(4,2,-8,-6)
LanczosResize(640,480)
ConvertToYV12(interlaced=false)
# after deinterlace / capture / conversion
SmoothLevels(16, 1.0, 255, 16, 235, limiter=1, tvrange=true, dither=0)

# ultrasound-focused enhancement (toggleable)
function EnhanceUltrasound(clip c)
{
    base = c.ConvertToYV12(interlaced=false)
    y = base.ConvertToY8()                               # process luma only
    y = y.RemoveGrain(1)                                 # very light cleanup
    y = y.SmoothLevels(16, 1.0, 235, 16, 240, limiter=1, tvrange=true, dither=0)
    y = y.Sharpen(0.05)                                  # tiny edge boost
    y = y.ConvertToYV12(interlaced=false)
    return y.MergeChroma(base)                           # preserve original color tint
}

function EnhanceText(clip c)
{
    base = c.ConvertToYV12(interlaced=false)
    y = base.ConvertToY8()                               # sharpen text edges on luma only
    y = y.Sharpen(0.3)
    y = y.SmoothLevels(16, 1.03, 235, 14, 242, limiter=1, tvrange=true, dither=0)
    return base.MergeLuma(y.ConvertToYV12(interlaced=false))
}
ultrasound_enhance = false
text_enhance = true

# optional wobble fix for text edges (set to true on problem tapes)
function WobbleFix(clip c) { c.TurnLeft().Blur(0.3).TurnRight() }
wobble_fix = false
c = last
if (ultrasound_enhance) {
    c = EnhanceUltrasound(c)
}
if (text_enhance) {
    c = EnhanceText(c)
}
if (wobble_fix) {
    c = WobbleFix(c)
}
return c
