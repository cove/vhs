"""
Archive Video Conversion Script

This script converts one or more MKV video files into an archival format using FFmpeg.
- Video is encoded as FFV1 with YUV422 color space and full color metadata.
- Audio is encoded as PCM 16-bit little-endian.
- Output files are saved with "_archive.mkv" appended to the original filename.

Usage:
    python this_script.py video1.mkv video2.mkv ...
"""

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
FFMPEG = SCRIPT_DIR / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"

if not FFMPEG.exists():
    print("ERROR: ffmpeg.exe not found!")
    print(f"   Looking for: {FFMPEG.resolve()}")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python this_script.py video1.mkv video2.mkv ...")
    sys.exit(1)

for file in sys.argv[1:]:
    file_path = Path(file)
    if not file_path.exists():
        print(f"File not found: {file}")
        continue

    output = file_path.with_name(file_path.stem + "_archive.mkv")

    print(f"Converting: {file_path.name}  →  {output.name}")

    cmd = [
        str(FFMPEG),
        "-nostdin", "-v", "error", "-stats",
        "-i", str(file_path),
        "-pix_fmt", "yuv422p",
        "-color_primaries:v", "6",
        "-color_trc:v", "6",
        "-colorspace:v", "5",
        "-color_range:v", "1",
        "-map", "0:v:0",
        "-c:v", "ffv1",
        "-level", "3",
        "-g", "1",
        "-coder", "1",
        "-context", "1",
        "-slices", "24", "-slicecrc", "1",
        "-map", "0:a", "-c:a", "pcm_s16le",
        "-y", str(output)
    ]

    subprocess.run(cmd, check=True)
    print(f"Done converting {file_path.name}\n")

print("All finished!")
