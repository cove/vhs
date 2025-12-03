import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"

ARCHIVE = BASE.parent / "Archive"
CLIPS = BASE.parent / "Clips"
VIDEOS = BASE.parent / "Videos"
SUBTITLES = BASE.parent / "Subtitles"

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

all_videos = (f for folder in [VIDEOS, CLIPS] for f in folder.glob("*.mp4"))
for src in all_videos:
    prefix = src.stem
    subtitle_file = SUBTITLES / f"{prefix}.vtt"

    if src.name.endswith(".subtitle_temp.mp4"):
        continue

    if not subtitle_file.exists():
        print(f"No subtitles found for {src.name}, skipping")
        continue

    final_file = src
    temp_file = src.with_suffix(".subtitle_temp.mp4")

    cmd = [
        FFMPEG,
        "-v", "warning",
        "-i", str(src),
        "-i", str(subtitle_file),
        "-map", "0:v",
        "-map", "0:a",
        "-map", "1",
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:v:0", "language=eng",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "default",
        "-y", str(temp_file)
    ]

    print(f"Applying subtitles to {src.name}")
    run(cmd)

print("All done")
