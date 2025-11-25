#!/usr/bin/env python3
# truncate_video.py
# Drag & drop video + type duration → trims with no re-encode

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
FFMPEG = SCRIPT_DIR / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"

if not Path(FFMPEG).exists():
    print("ERROR: ffmpeg.exe not found!")
    print(f"   Expected: {Path(FFMPEG).resolve()}")
    sys.exit(1)

if len(sys.argv) != 4:
    print("   python truncate_video.py input.mp4 output.mp4 01:04:28")
    sys.exit(1)

input_file = Path(sys.argv[1]).resolve()
output_file = Path(sys.argv[2]).resolve()
duration = sys.argv[3]

if not input_file.exists():
    print(f"File not found: {input_file}")
    sys.exit(1)

print(f"Trimming: {input_file.name}")
print(f"Duration: {duration} → {output_file.name}")

import subprocess
subprocess.run([
    FFMPEG,
    "-nostdin", "-v", "error",
    "-i", str(input_file),
    "-t", duration,
    "-c", "copy",
    "-y", str(output_file)
], check=True)

print("Trim complete!")
