#!/usr/bin/env python3
# vhs_c_qtgmc_chapter_split.py
# Drag & drop .mkv files → perfect QTGMC + x265 chapters
# creation_time comes from media_metadata/.../chapters.ffmetadata

import os
import sys
import json
import subprocess
import tempfile
import re
from pathlib import Path
from datetime import datetime, timezone

FFMPEG    = r"software/FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"
FFPROBE   = r"software/FFmpeg-QTGMC Easy 2025.01.11/ffprobe.exe"
QTGMC_DIR = r"software/FFmpeg-QTGMC Easy 2025.01.11"

for tool in (FFMPEG, FFPROBE):
    if not Path(tool).exists():
        print(f"ERROR: {tool} not found!")
        sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python this_script.py video1.mkv")
    sys.exit(1)

script_dir = Path(__file__).parent.resolve()

for file in sys.argv[1:]:
    src = Path(file).resolve()
    if not src.exists():
        print(f"File not found: {src}")
        continue

    name = src.stem
    out_dir = src.parent / f"{name}_chapters"
    out_dir.mkdir(exist_ok=True)

    # Extract video prefix (e.g. Tape1995_01.mkv → Tape1995)
    match = re.match(r"^(.*?[0-9]+)", name)
    video_prefix = match.group(1) if match else name

    # Load external ffmetadata file
    chapters_file = script_dir / "media_metadata" / video_prefix / "chapters.ffmetadata"
    if not chapters_file.exists():
        print(f"ERROR: No chapters.ffmetadata found for {src.name}")
        print(f"   Expected: {chapters_file}")
        continue

    print(f"\nProcessing: {src.name}")
    print(f"   Using metadata: {chapters_file}")

    # Parse ffmetadata to get titles + creation_time
    chapter_data = []
    with open(chapters_file, "r", encoding="utf-8") as f:
        content = f.read()
        blocks = re.split(r"\[CHAPTER\]", content)
        for block in blocks[1:]:  # Skip header
            title_match = re.search(r"title=(.+)", block)
            time_match = re.search(r"creation_time=(.+)", block)
            title = title_match.group(1).strip() if title_match else "Untitled"
            ctime = time_match.group(1).strip() if time_match else None
            chapter_data.append((title, ctime))

    # Get chapter timestamps from the source video
    chapters_json = subprocess.check_output([
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_chapters", "-show_format", str(src)
    ], text=True)
    data = json.loads(chapters_json)
    chapters = data["chapters"]
    duration = float(data["format"]["duration"])

    if len(chapter_data) != len(chapters):
        print(f"WARNING: Metadata has {len(chapter_data)} chapters, video has {len(chapters)}")
        print("   Using minimum count...")

    num_chapters = min(len(chapter_data), len(chapters))

    for i in range(num_chapters):
        ch = chapters[i]
        num = f"{i+1:02d}"
        title, creation_from_meta = chapter_data[i]

        start = float(ch["start_time"])
        end = float(chapters[i+1]["start_time"]) if i < num_chapters-1 else duration

        # Use creation_time from metadata file first
        creation = creation_from_meta
        if not creation:
            # Fallback: calculate from start time
            creation = datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() + start
            creation = datetime.fromtimestamp(creation, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")

        safe_title = "".join(c if c not in r'<>:"/\|?*' else " - " for c in title)
        final = out_dir / f"{num} - {safe_title}.mp4"
        temp_raw = Path(tempfile.gettempdir()) / f"vhs_temp_{num}.mkv"

        print(f"   → {num} - {title}  ({creation})")

        # Extract raw chapter
        subprocess.run([
            FFMPEG, "-v", "error", "-ss", str(start), "-to", str(end),
            "-i", str(src), "-map", "0:v", "-map", "0:a?", "-c", "copy",
            "-avoid_negative_ts", "make_zero", "-y", str(temp_raw)
        ], check=True)

        # QTGMC .avs
        avs_content = f'''
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
        avs_file.write_text(avs_content, encoding="ascii")

        # Encode
        subprocess.run([
            FFMPEG,
            "-i", str(avs_file), "-i", str(temp_raw),
            "-map", "0:v", "-map", "1:a?",
            "-metadata", f"title={title}",
            "-metadata", f"comment=Chapter from {src.name}",
            "-metadata", f"creation_time={creation}",
            "-c:v", "libx265", "-preset", "slow", "-crf", "18",
            "-x265-params", "aq-mode=3:profile=main10",
            "-c:a", "aac", "-b:a", "48k",
            "-af", "highpass=f=80,lowpass=f=14000,acompressor=ratio=3:attack=8:release=60",
            "-movflags", "+faststart",
            "-y", str(final)
        ], check=True)

        temp_raw.unlink(missing_ok=True)
        avs_file.unlink(missing_ok=True)

    print(f"Finished → {out_dir.name}")

print("\nAll done! creation_time from media_metadata is now correctly applied.")
