#!/usr/bin/env python3

import sys
import subprocess
import tempfile
from pathlib import Path

FFMPEG    = "software/FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"
QTGMC_DIR = "software/FFmpeg-QTGMC Easy 2025.01.11"

for src_path in sys.argv[1:]:
    src = Path(src_path).resolve()
    name = src.stem

    # Extract prefix: bennett_1_metadata_archive → bennett_1
    prefix = name.rsplit("_", 2)[0] if "_" in name else name

    out_dir = src.parent / f"{name}_chapters"
    out_dir.mkdir(exist_ok=True)

    chapters_file = Path(__file__).parent / "media_metadata" / prefix / "chapters.ffmetadata"

    # Parse ffmetadata — simple, perfect, bulletproof
    chapters = []
    title = None
    ctime = None
    start = end = None

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
        final = out_dir / f"{safe_title}.mp4"
        temp_raw = Path(f"temp_{num}_{safe_title}.mkv")

        print(f"Processing chapter: {title}")
        subprocess.run([
            FFMPEG, "-v", "error", "-stats",
            "-ss", str(start), "-to", str(end),
            "-i", str(src),
            "-map", "0:v", "-map", "0:a?",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-y", str(temp_raw)
        ], check=True)

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
FFmpegSource2("{temp_raw}", atrack=-1)
ConvertToYV12(matrix="Rec601")
QTGMC(preset="Faster")
Crop(0,0,-2,-6)
LanczosResize(640,480)
Return Last
'''
        avs_file = Path(f"qtgmc_{num}.avs")
        avs_file.write_text(avs, encoding="ascii")

        subprocess.run([
            FFMPEG,"-v", "error", "-stats",
            "-i", str(avs_file), "-i", str(temp_raw), 
            "-map", "0:v", "-map", "1:a?",
            "-metadata", f"title={title}",
            "-metadata", f"creation_time={ctime or ''}",
            "-c:v", "libx265", "-preset", "slow", "-crf", "18",
            "-x265-params", "profile=main10",
            "-tag:v", "hvc1",
            "-c:a", "aac", "-b:a", "48k",
            "-af", "highpass=f=80,lowpass=f=14000,acompressor",
            "-movflags", "+faststart",
            "-y", str(final)
        ], check=True)

        temp_raw.unlink(missing_ok=True)
        avs_file.unlink(missing_ok=True)

    print(f"Done: {out_dir.name}")

print("All finished.")
