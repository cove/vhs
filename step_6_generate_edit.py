"""
Generate Edit Versions from Archive MKV Files

This script converts all MKV files in the Archive directory into ProRes edit-friendly MOV files.
- Video is encoded as ProRes 422 (profile 1) with vendor 'ap10'.
- Audio is converted to mono PCM 16-bit.
- Metadata from the source MKV is preserved.
- Output files are written to the Edit directory, with "_edit" replacing "_archive" in the filename.

Requirements:
- ffmpeg executable must be available at software/FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe
"""

import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
ARCHIVE = BASE.parent / "Archive"
EDIT_VERSIONS = BASE.parent / "Edit"
EDIT_VERSIONS.mkdir(exist_ok=True)

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg not found at {FFMPEG}")
    sys.exit(1)

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

print(f"Generating edit versions → {EDIT_VERSIONS}\n")

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    final = EDIT_VERSIONS / f"{name.replace('_archive', '_edit')}.mov"

    if final.exists():
        print(f"Skipping {src.name} (edit version exists)")
        continue

    print(f"Processing: {src.name} → {final.name}")

    run([
        FFMPEG,
        "-nostdin",
        "-i", str(src),
        "-map_metadata", "0",
        "-c:v", "prores_ks",
        "-profile:v", "1",
        "-vendor", "ap10",
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-y",
        str(final)
    ])

print("\nAll edit versions done")
