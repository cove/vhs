"""
Generate Half-Size Proxy Versions from Archive MKV Files

This script creates low-resolution, fast-decodable proxy MP4 files at half the original size.
- Video is encoded with H.265 (libx265) at ½ size, CRF 28, ultrafast preset, tuned for fast decode.
- Audio is mono AAC at 16 kbps.
- Metadata from the source MKV is preserved.
- Output files are written to the Proxy directory, with "_small_proxy" replacing "_archive" in the filename.

Requirements:
- ffmpeg executable must be available at software/FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe
"""

import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
ARCHIVE = BASE.parent / "Archive"
PROXY = BASE.parent / "Proxy"
PROXY.mkdir(exist_ok=True)

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg not found at {FFMPEG}")
    sys.exit(1)

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

print(f"Generating ½-size PROXY → {PROXY}\n")

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    proxy = PROXY / f"{name.replace('_archive', '_small_proxy')}.mp4"

    if proxy.exists():
        print(f"Skipping {src.name} (proxy exists)")
        continue

    print(f"Processing: {src.name} → {proxy.name}")

    run([
        FFMPEG,
        "-nostdin",
        "-v", "error",
        "-i", str(src),
        "-map_metadata", "0",
        "-map_chapters", "0",
        "-vf", "scale=iw/2:ih/2",
        "-c:v", "libx265",
        "-preset", "ultrafast",
        "-crf", "28",
        "-tune", "fastdecode",
        "-x265-params", "no-sao=1:rect=0:strong-intra-smoothing=0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "16k",
        "-ac", "1",
        "-tag:v", "hvc1",
        "-brand", "mp42",
        "-movflags", "+faststart+write_colr+use_metadata_tags",
        "-y", str(proxy)
    ])

print("\nAll PROXY done")
