import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
FFMPEG = BASE_DIR / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
QTGMC_DIR = BASE_DIR / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
ARCHIVE_DIR = BASE_DIR / ".." / "Archive"

mkv_files = list(ARCHIVE_DIR.glob("bennett*.mkv"))
if not mkv_files:
    print("No .mkv files found in {ARCHIVE_DIR}")
    sys.exit(0)

print(f"Found {len(mkv_files)} files in {ARCHIVE_DIR}\n")

for src in mkv_files:
    name = src.stem
    prefix = "_".join(name.rsplit("_", 2)[:2])  # bennett_1_metadata_archive → bennett_1
    out_dir = src.parent / f"{name}_chapters"
    out_dir.mkdir(exist_ok=True)

    chapters_file = BASE_DIR / "media_metadata" / prefix / "chapters.ffmetadata"
    if not chapters_file.exists():
        print(f"Skipping {src.name} — no metadata: {chapters_file}")
        continue

    print(f"Processing: {src.name}")

    # Parse chapters
    chapters = []
    title = ctime = start = end = None
    with open(chapters_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[CHAPTER]":
                if title is not None:
                    chapters.append((title, ctime, start, end))
                title = ctime = start = end = None
                continue
            if line.startswith("title="):
                title = line[6:].strip()
            elif line.startswith("creation_time="):
                ctime = line[14:].strip()
            elif line.startswith("START="):
                start = int(line[6:])
            elif line.startswith("END="):
                end = int(line[4:])
        if title is not None:
            chapters.append((title, ctime, start, end))

    for i, (title, ctime, start, end) in enumerate(chapters):
        num = f"{i+1:02d}"
        safe_title = title.translate(str.maketrans(r'<>:"/\|?*', "---------"))
        final_temp = out_dir / f"{safe_title}_temp.mp4"
        final = out_dir / f"{safe_title}.mp4"
        temp_raw = out_dir / f"temp_raw_{num}.mkv"
        temp_raw_ffindex = out_dir / f"temp_raw_{num}.mkv.ffindex"

        if os.path.exists(final) and os.path.getsize(final) > 100_000:
            print(f"   Skipping existing chapter: {final.name} (delete if you want to reprocess it)")
            continue

        # Step 1: Extract raw chapter
        print(f"Processing: {src} - Chapter: {title}")
        subprocess.run([FFMPEG, "-v", "error", "-stats",
            "-i", str(src),
            "-map_metadata", "-1",
            "-ss", str(start), "-to", str(end),
            "-map", "0:v", "-map", "0:a",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-y", str(temp_raw)],
        check=True, cwd=out_dir)
        
        if os.path.exists(temp_raw) and os.path.getsize(temp_raw) < 100_000:
            print(f"Failed to extract chapter: {src} - Chapter: {title}")
            sys.exit(1)
    
        # Step 2: QTGMC + x265 in one pass
        avs = f'''
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
ConvertToYV12(matrix="Rec601")
QTGMC(preset="Faster")
Crop(0,0,-2,-6)
LanczosResize(640,480)
Return Last
'''
        avs_file = out_dir / f"qtgmc_{num}.avs"
        avs_file.write_text(avs, encoding="ascii")

        subprocess.run([
            FFMPEG,
            "-i", str(f"qtgmc_{num}.avs"), "-i", str(temp_raw), "-stats",
            "-map", "0:v", "-map", "1:a",
            "-map_metadata", "-1",
            "-metadata", f"title={title}",
            "-metadata", f"creation_time={ctime or ''}",
            "-metadata", f"description=Source VHS tape archive: {src.name}",
            "-metadata", f"com.apple.quicktime.creationdate={ctime or ''}",
            "-tag:v", "hvc1",
            "-brand", "mp42",
            "-c:v", "libx265", "-preset", "fast", "-crf", "18",
            "-profile:v", "main10",
            "-pix_fmt", "yuv420p10le",
            "-c:a", "aac", "-b:a", "48k", "-ac", "1",
            "-af", "highpass=f=80,lowpass=f=14000,acompressor",
            "-movflags", "+faststart+write_colr",
            "-y", str(final)
        ], check=True, cwd=out_dir)

        # Clean up everything
        temp_raw.unlink(missing_ok=True)
        temp_raw_ffindex.unlink(missing_ok=True)
        avs_file.unlink(missing_ok=True)
        final_temp.unlink(missing_ok=True)


    print(f"Finished: {out_dir.name}\n")

print("Done.")
