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

for src in CLIPS.glob("*.mp4"), VIDEOS.glob("*.mp4"):
    each = Path(str(src))
    prefix = each.stem
    subtitle_file = SUBTITLES / f"{prefix}.vtt"
    if not subtitle_file.exists():
        print(f"No subtitles found for {each.name}, skipping")
        continue

    final_file = each
    temp_file = Path(str(each)).with_suffix(".subtitle_temp.mp4")

    cmd = [
        FFMPEG,
        "-i", str(each),
        "-i", str(subtitle_file),
        "-map", "0:v",
        "-map", "0:a",
        "-map", "1",
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "default",
        "-y",
        str(temp_file)
    ]

    print(f"Applying subtitles to {each.name}")
    run(cmd)

print("All done")
