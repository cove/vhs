import sys
from pathlib import Path

FFMPEG = Path(__file__).parent / ".." / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"

if len(sys.argv) != 3:
    print("Usage: python truncate_video.py input.mp4 01:04:28")
    sys.exit(1)

input_file = Path(sys.argv[1]).resolve()
duration = sys.argv[2]

if not input_file.exists():
    print(f"File not found: {input_file}")
    sys.exit(1)

output_file = input_file.with_name(f"{input_file.stem}_trunc{input_file.suffix}")

print(f"Trimming {input_file.name} → {output_file.name} (duration: {duration})")

import subprocess
subprocess.run([
    str(FFMPEG),
    "-nostdin", "-v", "error",
    "-i", str(input_file),
    "-t", duration,
    "-c", "copy",
    "-y", str(output_file)
], check=True)

print("Done.")
