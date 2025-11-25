#!/usr/bin/env python3
# vhs_c_qtgmc_chapter_split.py
# Drag & drop .mkv files → perfect 640x480 QTGMC + x265 chapters
# 2025 rigid workflow — assumes chapters exist

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

# CHANGE ONLY THESE LINES
FFMPEG   = r"software\FFmpeg-QTGMC Easy 2025.01.11\ffmpeg.exe"
FFPROBE  = r"software\FFmpeg-QTGMC Easy 2025.01.11\ffprobe.exe"
QTGMC_DIR = r"software\FFmpeg-QTGMC Easy 2025.01.11"
CRF = 18

# Check tools exist
for tool in (FFMPEG, FFPROBE):
    if not Path(tool).exists():
        print(f"ERROR: {tool} not found!")
        input("Press Enter to exit...")
        sys.exit(1)

if len(sys.argv) < 2:
    print("Drag your .mkv files onto this script")
    input("Press Enter to exit...")
    sys.exit(1)

for file in sys.argv[1:]:
    src = Path(file).resolve()
    if not src.exists():
        print(f"File not found: {src}")
        continue

    name = src.stem
    out_dir = src.parent / f"{name}_chapters"
    out_dir.mkdir(exist_ok=True)

    print(f"\nProcessing: {src.name}")

    # Get chapters and source creation_time
    chapters = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-print_format", "json", "-show_chapters", src], text=True))["chapters"]
    source_creation = json.loads(subprocess.check_output([FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", src], text=True))["format"]["tags"].get("creation_time")

    for i, ch in enumerate(chapters):
        num = f"{i+1:02d}"
        title = ch["tags"]["title"].strip()
        start = float(ch["start_time"])
        end = float(chapters[i+1]["start_time"]) if i < len(chapters)-1 else float(json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", src], text=True))["format"]["duration"])

        # creation_time priority
        creation = ch["tags"].get("creation_time")
        if not creation:
            from datetime import datetime, timezone
            creation = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() + start
            creation = datetime.fromtimestamp(creation, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
        if not creation:
            creation = source_creation or ""

        safe_title = "".join(c if c not in r'<>:"/\|?*' else " - " for c in title)
        final = out_dir / f"{num} - {safe_title}.mp4"
        temp_raw = Path(tempfile.gettempdir()) / f"vhs_temp_{num}.mkv"

        print(f"   → {num} - {title}")

        # 1. Extract raw chapter
        subprocess.run([
            FFMPEG, "-v", "error", "-ss", str(start), "-to", str(end),
            "-i", str(src), "-map", "0:v", "-map", "0:a?", "-c", "copy",
            "-avoid_negative_ts", "make_zero", "-y", str(temp_raw)
        ], check=True)

        # 2. Create .avs with QTGMC
        avs_content = f'''
LoadPlugin("{QTGMC_DIR}\\ffms2.dll")
LoadPlugin("{QTGMC_DIR}\\masktools2.dll")
LoadPlugin("{QTGMC_DIR}\\Rgtools.dll")
LoadPlugin("{QTGMC_DIR}\\mvtools2.dll")
LoadPlugin("{QTGMC_DIR}\\nnedi3.dll")
LoadPlugin("{QTGMC_DIR}\\yadifmod2.dll")
LoadPlugin("{QTGMC_DIR}\\fft3dfilter.dll")
LoadPlugin("{QTGMC_DIR}\\LoadDLL64.dll")
LoadDLL("{QTGMC_DIR}\\libfftw3f-3.dll")
Import("{QTGMC_DIR}\\Zs_RF_Shared.avsi")
Import("{QTGMC_DIR}\\QTGMC.avsi")

FFmpegSource2("{temp_raw}", atrack=-1)
ConvertToYV12(matrix="Rec601")
QTGMC(preset="Faster")
Crop(0,0,-2,-6)
LanczosResize(640,480)
Return Last
'''

        avs_file = Path(tempfile.gettempdir()) / f"qtgmc_{num}.avs"
        avs_file.write_text(avs_content, encoding="ascii")

        # 3. Encode with QTGMC + x265
        subprocess.run([
            FFMPEG,
            "-i", str(avs_file), "-i", str(temp_raw),
            "-map", "0:v", "-map", "1:a?",
            "-metadata", f"title={title}",
            "-metadata", f"comment=Extracted chapter from {src.name}",
            "-metadata", f"creation_time={creation}",
            "-c:v", "libx265", "-preset", "slow", "-crf", str(CRF),
            "-x265-params", "profile=main10:aq-mode=3",
            "-c:a", "aac", "-b:a", "48k",
            "-af", "highpass=f=80,lowpass=f=14000,acompressor=ratio=3:attack=8:release=60",
            "-movflags", "+faststart",
            "-y", str(final)
        ], check=True)

        # 4. Clean up
        temp_raw.unlink(missing_ok=True)
        avs_file.unlink(missing_ok=True)

    print(f"Finished → {out_dir.name}")

print("\nAll done! Every chapter processed perfectly.")
input("Press Enter to exit...")
