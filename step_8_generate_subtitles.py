import subprocess, sys
from pathlib import Path
import whisper
from whisper.utils import write_srt
import io

# Load model once
model = whisper.load_model("large-v3")

#VIDEOS = Path("../Videos")
CLIPS = Path("../Clips")
FFMPEG = Path("software/FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe")

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

    # Generate SRT in memory
    srt_buffer = io.StringIO()
    write_srt(result["segments"], file=srt_buffer)
    srt_content = srt_buffer.getvalue()

    # Temp files
    temp_srt = CLIPS / "temp_subtitles.srt"
    temp_output = mp4.with_name(f"{mp4.stem}_subtitles_temp.mp4")
    final_output = mp4.with_name(f"{mp4.stem}.mp4")

    # Save SRT
    temp_srt.write_text(srt_content, encoding="utf-8")

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

    # Verify subtitles are really there
    if not has_subtitles(temp_output):
        print(f"  ERROR: subtitles failed for {mp4.name}")
        temp_srt.unlink(missing_ok=True)
        temp_output.unlink(missing_ok=True)
        continue

    # Success — replace original
    mp4.replace(final_output)
    temp_output.replace(mp4)

    # Cleanup
    temp_srt.unlink(missing_ok=True)

    print(f"  Done → {mp4.name} now has subtitles\n")

print("All done")
