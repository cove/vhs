#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
FFMPEG = SCRIPT_DIR / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"

if not os.path.exists(FFMPEG):
    print("ERROR: ffmpeg.exe not found!")
    print(f"   Looking for: {FFMPEG.resolve()}")
    sys.exit(1)

if len(sys.argv) < 2:
    print("python this_script.py video1.mkv video2.mkv ...")
    sys.exit(1)

for file in sys.argv[1:]:
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    name = os.path.splitext(file)[0]
    output = name + "_archive.mkv"

    print(f"Converting: {os.path.basename(file)}  →  {os.path.basename(output)}")

    cmd = [
        str(FFMPEG),
        "-nostdin", "-v", "error", "-stats",
        "-i", file,
        "-pix_fmt", "yuv422p",
        "-color_primaries:v", "6",
        "-color_trc:v", "6",
        "-colorspace:v", "5",
        "-color_range:v", "1",
        "-map", "0:v:0", "-c:v", "ffv1",
        "-level", "3", "-g", "1", "-coder", "1", "-context", "1",
        "-slices", "24", "-slicecrc", "1",
        "-map", "0:a?", "-c:a", "pcm_s16le",
        "-y", output
    ]

    subprocess.run(cmd)

    print("Done!\n")

print("All finished!")
