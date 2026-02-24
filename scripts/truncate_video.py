#!/usr/bin/env python3
"""
Truncate a video to a specific duration without re-encoding.
Usage:
    python truncate_video.py input.mp4 01:04:28
"""
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent.resolve()
FFMPEG_DIR = None

if sys.platform == "win32":
    FFMPEG_DIR = BASE / "software" / "Windows" / "FFmpeg-QTGMC Easy 2025.01.11"
    FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
    FFPROBE = FFMPEG_DIR / "ffprobe.exe"
elif sys.platform == "darwin":
    FFMPEG_DIR = BASE / "bin"
    FFMPEG = FFMPEG_DIR / "ffmpeg-8.0.1.darwin.arm64"
    FFPROBE = FFMPEG_DIR / "ffprobe-8.0.1.darwin.arm64"
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python truncate_video.py input.mp4 01:04:28")
        sys.exit(1)

    input_file = Path(sys.argv[1]).resolve()
    duration = sys.argv[2]

    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    output_file = input_file.with_name(f"{input_file.stem}_trunc{input_file.suffix}")

    print(f"Trimming {input_file.name} -> {output_file.name} (duration: {duration})")

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

if __name__ == "__main__":
    main()
