AssumeFPS(30000,1001)
AssumeBFF()
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3,MatchEdi="NNEDI3",MatchEdi2="NNEDI3")
# stabilization disabled
Crop(4,2,-8,-10)
LanczosResize(640,480)
ConvertToYV12(interlaced=false)
# after deinterlace / capture / conversion
SmoothLevels(16, 1.0, 255, 16, 235, limiter=1, tvrange=true, dither=0)

# ultrasound-focused enhancement (toggleable)
function EnhanceUltrasound(clip c)
{
    c = c.ConvertToY8()                                  # remove chroma noise
    c = c.RemoveGrain(2)                                 # mild spatial cleanup
    c = c.TemporalSoften(2,2,0,mode=2,scenechange=8)     # light temporal smoothing
    c = c.Sharpen(0.15)                                  # subtle edge boost
    c = c.SmoothLevels(16, 1.0, 235, 12, 245, limiter=1, tvrange=true, dither=0)
    return c.ConvertToYV12(interlaced=false)
}
ultrasound_enhance = true

# optional wobble fix for text edges (set to true on problem tapes)
function WobbleFix(clip c) { c.TurnLeft().Blur(0.3).TurnRight() }
wobble_fix = false
c = last
if (ultrasound_enhance) {
    c = EnhanceUltrasound(c)
}
c = Iif(wobble_fix, WobbleFix(c), c)
return c
