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

def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def run(cmd, cwd=None):
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def parse_chapters(path):
    chapters = []
    globals = {}
    cur = {}
    in_chapter = False
    seen_chapter = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not seen_chapter and "=" in line and not line.startswith(("[", ";")):
            k, v = line.split("=", 1)
            globals[k.strip().lower()] = v.strip()
            continue

        if line == "[CHAPTER]":
            seen_chapter = True
            if cur and in_chapter:          # save previous chapter
                chapters.append(cur)
            cur = {}
            in_chapter = True
            continue

        if in_chapter and "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()

        if in_chapter and not line and cur:
            chapters.append(cur)
            cur = {}
            in_chapter = False

    if cur and in_chapter:
        chapters.append(cur)

    return globals, chapters

def main():
    for src in ARCHIVE.glob("*.mkv"):
        name = src.stem
        prefix = "_".join(name.rsplit("_", 2)[:2])
        chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name} — no metadata")
            continue

        globals, chapters = parse_chapters(chapters_file)
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
            date = ctime[:4]
            location = ch.get("location", "")
            filter_avs = ch.get("filter_avs", "")

            final = VIDEOS / f"{safe(title)}.mp4"
            if final.exists():
                print(f"  Skipping {final.name}")
                continue

            temp_raw = VIDEOS / f"temp_raw_{i+1:02d}.mkv"
            avs_file = VIDEOS / f"qtgmc_{i+1:02d}.avs"

            print(f"  → {title} ({start_sec:.3f}s → {end_sec:.3f}s)")

            # Extract chapter (stream copy)
            run([
                FFMPEG, "-v", "warning",
                "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
                "-i", str(src),
                "-map", "0:v", "-map", "0:a",
                "-c", "copy", "-avoid_negative_ts", "make_zero",
                "-y", str(temp_raw)
            ])

            start_hms = format_hms(start_sec)
            end_hms = format_hms(end_sec)

            avs_script = None

            if True:
                avs_script = f'''
SetFilterMTMode("DEFAULT_MT_MODE", 2)
SetMemoryMax(8192)                     # helps very slow preset on large captures
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
LWLibavVideoSource("{temp_raw.name}", cache=false, fpsnum=30000, fpsden=1001)
AssumeBFF()                            # flip to BFF only if your capture card records bottom-first
ConvertToYV12(matrix="Rec601", interlaced=true)
Crop(12, 8, -12, -34)                  # removes ALL head-switching noise + side junk
DepanEstimate(range=6, trust=1.2, pixaspect=0.9091)   # 0.9091 = NTSC non-square
DepanStabilize(data=last, cutoff=1.2, mirror=12, prev=1, next=1)
QTGMC(
    Preset="Very Slow",
    SourceMatch=3, Lossless=2, TR2=3,           # maximum original-frame recovery
    Sharpness=0.7,                              # 0.8 → 0.7 to avoid ringing on residual edges
    SLMode=2, SLRad=3,                          # stronger spatial luma smoothing only where needed
    EZDenoise=0.4, EZKeepGrain=1.2,             # tiny bit more denoising, slightly more grain retention
    GrainRestore=0.4,                           # brings back natural VHS grain after SourceMatch
    Border=4,                                   # prevents QTGMC from hallucinating at frame edges
    EdiThreads=2)                               # stability improvement on multi-core
SeparateFields()
nnedi3(field=-2, dh=true, nsize=4, nns=4, qual=2)   # high-quality preset
Weave()
MergeChroma(last.QTGMC_chroma, last)        # keep QTGMC’s luma, replace chroma only
# Usually 2–6 pixels on any side – check one frame in VirtualDub/AvsPmod
Crop(4, 4, -4, -4)                     # safe default; adjust or comment out if not needed
LanczosResize(640,480)                 # or 720×540 / 720×534 for square-pixel output
Prefetch()
'''
            else:
                avs_script = f'''
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
{filter_avs}
Crop(0,0,-2,-6)
LanczosResize(640,480)
Prefetch()
'''
            avs_file.write_text(avs_script, encoding="ascii")

            cmd = [
                    FFMPEG, "-v", "warning",
                    "-i", str(avs_file), "-i", str(temp_raw),
                    "-map", "0:v", "-map", "1:a", "-map_metadata", "-1",
                    "-metadata", f"title={title}",
                    "-metadata", f"comment=Chapter from {src.name} @ {start_hms}-{end_hms}",
                    "-metadata", f"creation_time={ctime}",
                    "-metadata", f"CreateDate={ctime}",
                    "-metadata", f"MediaCreateDate={ctime}",
                    "-metadata", f"com.apple.quicktime.creationdate={ctime}",
                    "-metadata", f"com.apple.quicktime.uuid={globals.get('uuid', '')}",
                    "-metadata", f"date={date}",
                    "-metadata", f"genre={globals.get('genre', '')}",
                    "-metadata", f"composer={globals.get('composer', '')}",
                    "-metadata", f"artist={globals.get('artist', '')}",
                    "-metadata", f"album={globals.get('album', '')}",
            ]

            if location:
                iso6709 = location.rstrip("/") + "/"
                gpscoords = location.rstrip("/")
                longitude, latitude = gpscoords.split(",")

                cmd += [
                    "-metadata", f"com.apple.quicktime.location.ISO6709={iso6709}",
                    "-metadata", f"location={iso6709}",
                    "-metadata", f"location-eng={iso6709}",
                    "-metadata", f"GPSCoordinates={gpscoords}",
                    "-metadata", f"GPSLatitude={latitude}",
                    "-metadata", f"GPSLongitude={longitude}",
                ]

            cmd += [
                # — 2025 x265 quality settings —
                "-c:v", "libx265",
                "-preset", "veryslow",  # ← biggest single quality jump
                "-crf", "16",           # ← 16 = visually lossless for VHS
                "-profile:v", "main10",
                "-pix_fmt", "yuv420p10le",

                # — Perceptual & detail-preserving tuning —
                "-x265-params", "merange=57:psy-rd=2.0:aq-mode=3:aq-strength=1.0:bframes=8:keyint=600:rc-lookahead=80:no-sao=0:no-strong-intra-smoothing=0",
                "-x265-params", "deblock=-1:-1",
                "-x265-params", "ref=6",

                # — Apple / compatibility —
                "-tag:v", "hvc1",
                "-movflags", "+faststart+write_colr+use_metadata_tags",
                "-brand", "mp42",

                # — Audio —
                "-c:a", "aac", "-b:a", "64k", "-ac", "1",
                "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=g=15",

                "-y", str(final)
            ]
            run(cmd, cwd=VIDEOS)

            # Cleanup
            temp_raw.unlink(missing_ok=True)
            avs_file.unlink(missing_ok=True)

        print(f"Finished: {src.name}\n")

    print("All done")

if __name__ == "__main__":
    main()
