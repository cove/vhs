import os, sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
QTGMC = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
FFMPEG = QTGMC / "ffmpeg.exe"
FFPROBE = BASE / "bin" / "ffprobe.exe"
ARCHIVE = BASE.parent / "Archive"

def run(cmd, cwd=None):
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)

def parse_chapters(path):
    chapters, cur = [], {}
    for line in path.read_text().splitlines():
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

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "---------"))


def valid_media(path, min_bytes=100_000):
    # Size check
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        return False

    # ffprobe container check
    try:
        subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

mkv_files = list(ARCHIVE.glob("bennett*.mkv"))
if not mkv_files:
    print(f"No .mkv files found in {ARCHIVE}")
    sys.exit()

print(f"Found {len(mkv_files)} files in {ARCHIVE}\n")

for src in mkv_files:
    name = src.stem
    prefix = "_".join(name.rsplit("_", 2)[:2])
    out_dir = src.parent / ".." / f"Videos"
    out_dir.mkdir(exist_ok=True)

    chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
    if not chapters_file.exists():
        print(f"Skipping {src.name} — missing metadata")
        continue

    chapters = parse_chapters(chapters_file)
    print(f"Processing: {src.name}")

    for i, ch in enumerate(chapters, 1):
        title = ch.get("title", f"chapter_{i}")
        start, end = ch.get("start"), ch.get("end")
        ctime = ch.get("creation_time", "")

        final = out_dir / f"{safe(title)}.mp4"

        if valid_media(final):
            print(f"  Skipping {final.name} (exists)")
            continue

        temp_raw = out_dir / f"temp_raw_{i:02d}.mkv"
        avs_file = out_dir / f"qtgmc_{i:02d}.avs"

        # Extract raw
        run([FFMPEG, "-v", "error", "-stats", "-i", src,
             "-map_metadata", "-1",
             "-ss", start, "-to", end,
             "-map", "0:v", "-map", "0:a",
             "-c", "copy", "-avoid_negative_ts", "make_zero",
             "-y", temp_raw], cwd=out_dir)

        if not valid_media(temp_raw):
            print(f"  Error: invalid video, something went wrong with {temp_raw} ({temp_raw.stat().st_size} bytes).")
            sys.exit(1)

        # Write QTGMC script
        avs_file.write_text(f"""
LoadPlugin("{QTGMC}/ffms2.dll")
LoadPlugin("{QTGMC}/masktools2.dll")
LoadPlugin("{QTGMC}/Rgtools.dll")
LoadPlugin("{QTGMC}/mvtools2.dll")
LoadPlugin("{QTGMC}/nnedi3.dll")
LoadPlugin("{QTGMC}/yadifmod2.dll")
LoadPlugin("{QTGMC}/fft3dfilter.dll")
LoadPlugin("{QTGMC}/LoadDLL64.dll")
LoadDLL("{QTGMC}/libfftw3f-3.dll")
Import("{QTGMC}/Zs_RF_Shared.avsi")
Import("{QTGMC}/QTGMC.avsi")
FFmpegSource2("{temp_raw.name}", atrack=-1)
AssumeFPS(30000,1001)
ConvertToYV12(matrix="Rec601")
AssumeFPS(30000,1001)
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3,SLMode=2,SMode=2)

Levels(16, 1.10, 235, 0, 255, coring=false)
ColorYUV(off_u=-12, off_v=+6)
MergeChroma(Blur(0.8))
Tweak(sat=1.25, bright=2, cont=1.05)
# ColorYUV(gain_u=-30, gain_v=-30)

Crop(0,0,-2,-6)
LanczosResize(640,480)
""", encoding="ascii")

        # Encode
        run([
            FFMPEG,
            "-i", avs_file, "-i", temp_raw, "-stats",
            "-map", "0:v", "-map", "1:a",
            "-map_metadata", "-1",
            "-metadata", f"title={title}",
            "-metadata", f"creation_time={ctime}",
            "-metadata", f"description=Source VHS tape archive: {src.name}",
            "-metadata", f"com.apple.quicktime.creationdate={ctime}",
            "-tag:v", "hvc1", "-brand", "mp42",
            "-c:v", "libx265", "-preset", "fast", "-crf", "18",
            "-profile:v", "main10",
            "-pix_fmt", "yuv420p10le",
            "-c:a", "aac", "-b:a", "48k", "-ac", "1",
            "-af", "highpass=f=80,lowpass=f=14000,acompressor",
            "-movflags", "+faststart+write_colr",
            "-y", final
        ], cwd=out_dir)

        if not valid_media(final):
            print(f"  Error: invalid video, something went wrong with {final} ({final.stat().st_size} bytes).")
            sys.exit(1)

        # Cleanup
        for p in [temp_raw, temp_raw.with_suffix(".mkv.ffindex"), avs_file]:
            p.unlink(missing_ok=True)

    print(f"Finished: {out_dir.name}\n")

print("Done.")
