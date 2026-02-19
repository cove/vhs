AssumeFPS(30000,1001)
AssumeBFF()
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=0.9828,SourceMatch=3,Lossless=2,TR2=3,MatchEdi="NNEDI3",MatchEdi2="NNEDI3")
# stabilization disabled
Crop(10,2,-8,-10)
LanczosResize(640,480)
ConvertToYV12(interlaced=false)

# optional wobble fix for text edges (set to true on problem tapes)
function WobbleFix(clip c) { c.TurnLeft().Blur(0.3).TurnRight() }
wobble_fix = false
gamma_level = 1.0

# chapter-specific exposure by archive-global chapter START frame
# chapter 21: START=63773, END=64654
if (chapter_start_frame >= 63773 && chapter_start_frame < 64654) {
    gamma_level = 1.3
}
# chapters 22-23: START=64704/65100, END up to 65638
if (chapter_start_frame >= 64704 && chapter_start_frame < 65639) {
    gamma_level = 1.5
}

c = last
# after deinterlace / capture / conversion
c = c.SmoothLevels(16, gamma_level, 255, 16, 235, limiter=1, tvrange=true, dither=0)
if (wobble_fix) {
    c = WobbleFix(c)
}
return c
