#!/usr/bin/env python3

import glob
import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
FFMPEG = SCRIPT_DIR / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"

BASE_DIR = Path(__file__).parent.resolve()
ARCHIVE_DIR = BASE_DIR / ".." / "Archive"
B3SUM = BASE_DIR / "bin" / "b3sum_windows_x64_bin.exe"
MEDIAINFO = BASE_DIR / "bin" / "mediainfo.exe"

output_file = ARCHIVE_DIR / "00-manifest-blake3sums.txt"
if output_file.exists():
    output_file.unlink()

mkv_files = list(glob.glob(str(ARCHIVE_DIR / "*.mkv")))
if not mkv_files:
    print("No .mkv files found.")
    sys.exit(0)

for mkv in mkv_files:
    if not os.path.isfile(mkv):
        print(f"File not found: {mkv}")
        continue

    name = os.path.splitext(os.path.basename(mkv))[0]
    folder = os.path.dirname(mkv)

    # Extract prefix: bennett_1_metadata_archive → bennett_1
    prefix = "_".join(name.rsplit("_", 2)[:2])

    meta_dir = Path("..") / "media_metadata" / prefix
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

    print(f"Processing: {os.path.basename(mkv)} → {os.path.basename(output)}")

    cmd = [
        str(FFMPEG),
        "-nostdin", "-v", "error", "-stats",
        "-i", str(mkv),
        "-f", "ffmetadata", "-i", str(chapters_file),
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
        "-y", str(output)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Success!\n")
    else:
        print("FFmpeg failed:")
        print(result.stderr)
        print()

print("All done!")
