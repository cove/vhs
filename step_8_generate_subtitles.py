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

def has_subtitles(file_path):
    """Check if MP4 already has a subtitle stream"""
    result = subprocess.run([
        str(FFMPEG), "-v", "error", "-i", str(file_path),
        "-f", "null", "-"
    ], capture_output=True, text=True)
    return "Subtitle:" in result.stderr or "Stream #0:[1-9]+.*Subtitle" in result.stderr

srt_writer = get_writer("srt", str(CLIPS))

for mp4 in CLIPS.glob("*.mp4"):
    if "_subtitles_temp" in mp4.name:
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
    temp_srt = CLIPS / "temp_subtitles.srt"
    temp_output = mp4.with_name(f"{mp4.stem}_subtitles_temp.mp4")
    final_output = mp4.with_name(f"{mp4.stem}.mp4")
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
        "-y", str(temp_output)
    ])

    # # Verify subtitles are present
    # if not has_subtitles(temp_output):
    #     print(f"  ERROR: subtitles failed for {mp4.name}")
    #     temp_srt.unlink(missing_ok=True)
    #     temp_output.unlink(missing_ok=True)
    #     continue

    # Success — replace original
    mp4.replace(final_output)
    temp_output.replace(mp4)

    # Cleanup
    temp_srt.unlink(missing_ok=True)

    print(f"  Done → {mp4.name} now has subtitles\n")

print("All done")
