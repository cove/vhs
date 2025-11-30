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

print(f"Generating MP4 edit versions → {EDIT_VERSIONS}\n")

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    edit_mp4 = EDIT_VERSIONS / f"{name}_edit.mp4"

    if edit_mp4.exists():
        print(f"Skipping {src.name} (edit version exists)")
        continue

    print(f"Processing: {src.name} → {edit_mp4.name}")

    run([
        FFMPEG, "-v", "error",
        "-i", str(src),
        "-c:v", "libx265",
        "-preset", "slow",          # slower preset for max quality
        "-crf", "14",               # lower CRF = higher quality
        "-tune", "grain",           # preserves details/film grain
        "-x265-params", "no-sao=0:rect=1:strong-intra-smoothing=1",
        "-pix_fmt", "yuv420p10le",  # 10-bit for higher fidelity
        "-c:a", "aac", "-b:a", "48k",  # high-quality stereo audio
        "-ac", "1",
        "-tag:v", "hvc1",
        "-brand", "mp42",
        "-movflags", "+faststart+write_colr+use_metadata_tags",
        "-map_metadata", "0",  # preserve all metadata from source
        "-y", str(edit_mp4)
    ])

print("\nAll edit versions done")
