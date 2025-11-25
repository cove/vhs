#!/usr/bin/env python3

import os
import sys
import subprocess

ffmpeg = r"FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"

if not os.path.exists(ffmpeg):
    print("ERROR: ffmpeg.exe not found!")
    print(f"   Expected here: {os.path.abspath(ffmpeg)}")
    input("Press Enter to exit...")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Drag .mkv files onto this script")
    input("Press Enter to exit...")
    sys.exit(1)

for file in sys.argv[1:]:
    if not os.path.isfile(file):
        print(f"File not found: {file}")
        continue

    name = os.path.splitext(os.path.basename(file))[0]
    folder = os.path.dirname(file) or "."

    # Try to extract video name (e.g. HomeVideo1995_01.mkv → HomeVideo1995)
    import re
    match = re.match(r"^(.*?[0-9]+)", name)
    video_name = match.group(1) if match else name

    meta_dir = os.path.join(os.path.dirname(__file__), "media_metadata", video_name)
    cover = os.path.join(meta_dir, "cover.jpg")
    title_file = os.path.join(meta_dir, "title.txt")
    comment_file = os.path.join(meta_dir, "comment.txt")
    chapters_file = os.path.join(meta_dir, "chapters.ffmetadata")

    # Check if all metadata files exist
    missing = []
    for f in (cover, title_file, comment_file, chapters_file):
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        print(f"Skipping: {os.path.basename(file)}")
        print("   Missing metadata:")
        for m in missing:
            print(f"     {m}")
        print()
        continue

    # Read title and comment
    with open(title_file, "r", encoding="utf-8") as f:
        title = f.read().strip()
    with open(comment_file, "r", encoding="utf-8") as f:
        comment = f.read().strip()

    output = os.path.join(folder, f"{name}_metadata.mkv")

    print(f"Processing: {os.path.basename(file)} → {os.path.basename(output)}")

    cmd = [
        ffmpeg,
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
input("Press Enter to close...")
