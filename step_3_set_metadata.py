#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
FFMPEG = SCRIPT_DIR / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"

if len(sys.argv) < 2:
    print("python this_script.py video1.mkv")
    sys.exit(1)

for file in sys.argv[1:]:
    if not os.path.isfile(file):
        print(f"File not found: {file}")
        continue

    name = os.path.splitext(os.path.basename(file))[0]
    folder = os.path.dirname(file) or "."

    # Extract prefix: bennett_1_metadata_archive → bennett_1
    prefix = name.rsplit("_", 2)[0] if "_" in name else name

    meta_dir = os.path.join(os.path.dirname(__file__), "media_metadata", prefix)
    cover = os.path.join(meta_dir, "cover.jpg")
    title_file = os.path.join(meta_dir, "title.txt")
    comment_file = os.path.join(meta_dir, "comment.txt")
    chapters_file = os.path.join(meta_dir, "chapters.ffmetadata")

    # Read title and comment
    with open(title_file, "r", encoding="utf-8") as f:
        title = f.read().strip()
    with open(comment_file, "r", encoding="utf-8") as f:
        comment = f.read().strip()

    output = os.path.join(folder, f"{name}_metadata.mkv")

    print(f"Processing: {os.path.basename(file)} → {os.path.basename(output)}")

    cmd = [
        str(FFMPEG),
        "-nostdin", "-v", "error", "-stats",
        "-i", file,
        "-f", "ffmetadata", "-i", chapters_file,
        "-map", "0:v:0", "-map", "0:a",
        "-map_metadata", "0",
        "-map_chapters", "-1",
        "-map_chapters", "1",
        "-c", "copy",
        "-metadata", f"title={title}",
        "-metadata", f"comment={comment}",
        "-attach", cover,
        "-metadata:s:t:0", "mimetype=image/jpeg",
        "-metadata:s:t:0", "filename=cover.jpg",
        "-color_primaries:v", "6",
        "-color_trc:v", "6",
        "-colorspace:v", "5",
        "-aspect", "4:3",
        "-f", "matroska",
        "-y", output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Success!\n")
    else:
        print("FFmpeg failed:")
        print(result.stderr)
        print()

print("All done!")
