"""
Transcribe MP4 Clips Using Whisper

- Extracts and normalizes audio from MP4 clips
- Transcribes audio to VTT subtitle files in Subtitles/
- Uses the "turbo" Whisper model
"""

import subprocess
import sys
from pathlib import Path
import whisper
from whisper.utils import get_writer
import os

# --- Paths / Environment ---
BASE = Path(__file__).parent.resolve()
FFMPEG_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
CLIPS = BASE.parent / "Clips"
SUBTITLES = BASE.parent / "Subtitles"
SUBTITLES.mkdir(exist_ok=True)

# Ensure FFMPEG is in PATH for whisper
os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg not found at {FFMPEG}")
    sys.exit(1)

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

def extract_audio(src_mp4: Path, audio_wav: Path):
    """Extract and normalize audio from a video file."""
    run([
        FFMPEG, "-v", "warning",
        "-i", str(src_mp4),
        "-vn",
        "-af", "highpass=f=120,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=13,aresample=16000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le",
        "-y",
        str(audio_wav)
    ])

def transcribe_audio(model, audio_wav: Path, output_vtt: Path):
    """Transcribe audio to a VTT file using Whisper."""
    vtt_writer = get_writer("vtt", str(SUBTITLES))
    result = model.transcribe(str(audio_wav), language="en", fp16=False)
    vtt_writer(result, str(output_vtt))
    return result

model = whisper.load_model("turbo")

for mp4 in CLIPS.glob("*.mp4"):
    if "_temp" in mp4.name:
        print(f"Skipping {mp4.name}")
        continue

    temp_audio = CLIPS / f"{mp4.stem}_temp.wav"
    output_vtt = SUBTITLES / f"{mp4.stem}.vtt"

    if output_vtt.exists():
        print(f"Skipping {mp4.name} (VTT already exists)")
        continue

    print(f"Transcribing: {mp4.name}")

    extract_audio(mp4, temp_audio)
    transcribe_audio(model, temp_audio, output_vtt)

    temp_audio.unlink(missing_ok=True)
    print(f"  Done → {output_vtt.name}\n")

print("All done")
