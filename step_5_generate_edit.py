import sys, subprocess
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
    final = EDIT_VERSIONS / f"{name.replace("_archive", "_edit")}.mov"

    if final.exists():
        print(f"Skipping {src.name} (edit version exists)")
        continue

    print(f"Processing: {src.name} → {final.name}")

    subprocess.run([
        "ffmpeg",
        "-i", str(src),
        "-map_metadata", "0",

        # VIDEO: DNxHR LB (small, fully editable)
        "-c:v", "dnxhr",
        "-profile:v", "lb",
        "-pix_fmt", "yuv422p",

        # AUDIO
        "-c:a", "pcm_s16le",

        "-y",
        str(final)
    ], check=True)

print("\nAll edit versions done")
