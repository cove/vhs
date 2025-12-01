import subprocess, sys, os
from pathlib import Path
import whisper
from whisper.utils import get_writer

# Load model once
model = whisper.load_model("turbo")

CLIPS = Path("../Clips")
FFMPEG_DIR = Path("software/FFmpeg-QTGMC Easy 2025.01.11/")
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg not found at {FFMPEG}")
    input("Press Enter...")
    sys.exit(1)

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

srt_writer = get_writer("srt", str(CLIPS))

for mp4 in CLIPS.glob("*.mp4"):
    if "_temp" in mp4.name:
        print(f"Skipping {mp4.name}")
        continue

    print(f"Transcribing: {mp4.name}")

    # Whisper transcription
    result = model.transcribe(
        str(mp4),
        language="en",
        fp16=False
    )

    # Output SRT path
    temp_srt = CLIPS / mp4.with_name(f"{mp4.stem}_subtitles_temp.srt")
    temp_output = mp4.with_name(f"{mp4.stem}_subtitles_temp.mp4")
    srt_writer(result, str(temp_srt))

    # Mux subtitles into MP4
    run([
        FFMPEG, "-v", "error",
        "-i", str(mp4),
        "-i", str(temp_srt),
        "-map", "0", "-map", "1",
        "-c", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=English",
        "-disposition:s:0", "default",
        "-map_chapters", "-1"
        "-y", str(temp_output)
    ])

    mp4.replace(temp_output)
    temp_srt.unlink(missing_ok=True)

    print(f"  Done → {mp4.name} now has subtitles\n")

print("All done")
