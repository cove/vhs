"""
Archive Video Conversion Script

This script converts one or more captured AVI files into an archival MKV using FFmpeg.
- Video is encoded as FFV1 with YUV422 color space and SD color metadata tags.
- Audio is encoded as PCM 16-bit little-endian.
- Output files are saved with "_archive.mkv" appended to the original filename.

Usage:
    python this_script.py video1.avi video2.avi ...
"""

import sys
import subprocess
from pathlib import Path

from common import FFMPEG_BIN, ensure_ffmpeg_exists

try:
    ensure_ffmpeg_exists()
except FileNotFoundError as e:
    print(f"ERROR: {e}")
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
        str(FFMPEG_BIN),
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
