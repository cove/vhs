import subprocess, sys
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
QTGMC_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
ARCHIVE = BASE.parent / "Archive"
VIDEOS = BASE.parent / "Videos"
VIDEOS.mkdir(exist_ok=True)

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg.exe not found at {FFMPEG}")
    sys.exit(1)

def run(cmd, cwd=None):
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def parse_chapters(path):
    chapters = []
    cur = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "[CHAPTER]":
            if cur:
                chapters.append(cur)
            cur = {}
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()
    if cur:
        chapters.append(cur)
    return chapters

def main():
    for src in ARCHIVE.glob("*.mkv"):
        name = src.stem
        prefix = "_".join(name.rsplit("_", 2)[:2])
        chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name} — no metadata")
            continue

        chapters = parse_chapters(chapters_file)
        if not chapters:
            print(f"No chapters for {src.name}")
            continue

        for ch in chapters:
            start = int(ch.get("start", 0))
            end = int(ch.get("end", 0))
            ch["duration"] = end - start

        chapters.sort(key=lambda x: x["duration"])

        print(f"Processing: {src.name} ({len(chapters)} chapters, shortest first)")

        for i, ch in enumerate(chapters):
            title = ch.get("title", f"Chapter {i+1}")
            start_sec = int(ch["start"])
            end_sec = int(ch["end"])
            ctime = ch.get("creation_time", "")
            location = ch.get("location", "")

            final = VIDEOS / f"{safe(title)}.mp4"
            if final.exists():
                print(f"  Skipping {final.name}")
                continue

            temp_raw = VIDEOS / f"temp_raw_{i+1:02d}.mkv"
            avs_file = VIDEOS / "qtgmc.avs"

            print(f"  → {title} ({start_sec:.3f}s → {end_sec:.3f}s)")

            # Extract chapter (stream copy)
            run([
                FFMPEG, "-v", "error",
                "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
                "-i", str(src),
                "-map", "0:v", "-map", "0:a",
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                "-y", str(temp_raw)
            ])

            # QTGMC script
            avs_file.write_text(f'''
SetFilterMTMode("DEFAULT_MT_MODE", 2)
LoadPlugin("{QTGMC_DIR}/ffms2.dll")
LoadPlugin("{QTGMC_DIR}/masktools2.dll")
LoadPlugin("{QTGMC_DIR}/Rgtools.dll")
LoadPlugin("{QTGMC_DIR}/mvtools2.dll")
LoadPlugin("{QTGMC_DIR}/nnedi3.dll")
LoadPlugin("{QTGMC_DIR}/yadifmod2.dll")
LoadPlugin("{QTGMC_DIR}/fft3dfilter.dll")
LoadPlugin("{QTGMC_DIR}/LoadDLL64.dll")
LoadDLL("{QTGMC_DIR}/libfftw3f-3.dll")
Import("{QTGMC_DIR}/Zs_RF_Shared.avsi")
Import("{QTGMC_DIR}/QTGMC.avsi")
FFmpegSource2("{temp_raw.name}", atrack=-1)
AssumeFPS(30000,1001)
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3)
Levels(16, 1.10, 235, 0, 255, coring=false)
ColorYUV(off_u=-6, off_v=+2)
MergeChroma(Blur(0.8))
Tweak(sat=1.25, bright=2)
Crop(0,0,-2,-6)
LanczosResize(640,480)
Prefetch()
''', encoding="ascii")

            cmd = [FFMPEG, "-i", str(avs_file), "-i", str(temp_raw),
                "-map", "0:v", "-map", "1:a", "-map_metadata", "-1",
                "-metadata", f"title={title}",
                "-metadata", f"creation_time={ctime}",
                "-metadata", f"com.apple.quicktime.creationdate={ctime}",
                "-metadata:s:s:0", "language=eng",
                "-metadata:s:a:0", "language=eng"]

            if location:
                cmd += [
                    "-metadata", f"com.apple.quicktime.location.ISO6709={location}",
                    "-metadata", f"location={location}"
                ]

            cmd += [
                # — 2025 x265 quality settings —
                "-c:v", "libx265",
                "-preset", "veryslow",  # ← biggest single quality jump
                "-crf", "16",           # ← 16 = visually lossless for VHS
                "-profile:v", "main10",
                "-pix_fmt", "yuv420p10le",

                # — Perceptual & detail-preserving tuning —
                "-x265-params", (
                    "profile=main10:"
                    "psy-rd=2.0:"       # ← keeps grain & detail
                    "psy-rq=1.0:"
                    "aq-mode=3:"        # ← best quality distribution
                    "aq-strength=1.0:"
                    "deblock=-1:-1:"    # ← preserve edges
                    "merange=57:"       # ← better motion search
                    "ref=6:"            # ← more reference frames
                    "bframes=8:"        # ← better compression
                    "keyint=600:"       # ← 10 sec GOP at 59.94 fps
                    "rc-lookahead=80:"  # ← huge quality boost
                    "no-sao=0:"         # ← keep SAO on (helps banding)
                    "no-strong-intra-smoothing=0"
                ),

                # — Apple / compatibility —
                "-tag:v", "hvc1",
                "-movflags", "+faststart+write_colr",
                "-brand", "mp42",

                # — Audio —
                "-c:a", "aac", "-b:a", "64k", "-ac", "1",
                "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=g=15",

                "-y", str(final)]
            run(cmd, cwd=VIDEOS)

            # Cleanup
            temp_raw.unlink(missing_ok=True)
            avs_file.unlink(missing_ok=True)

        print(f"Finished: {src.name}\n")

    print("All done")

if __name__ == "__main__":
    main()