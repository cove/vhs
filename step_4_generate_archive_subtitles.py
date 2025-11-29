import subprocess, sys
from pathlib import Path

ARCHIVE = Path(__file__).parent.parent / "Archive"
MODEL = "large-v3"          # ← best quality (tiny/base/small/medium also work)
LANGUAGE = "en"             # ← change if needed

if not ARCHIVE.exists():
    print(f"ERROR: Archive folder not found at {ARCHIVE}")
    sys.exit(1)

mkv_files = list(ARCHIVE.glob("*.mkv"))
if not mkv_files:
    print("No .mkv files found in Archive/")
    sys.exit(0)

print(f"Found {len(mkv_files)} files — transcribing with Whisper {MODEL}\n")

for src in mkv_files:
    print(f"Transcribing: {src.name}")

    # Whisper command — outputs all formats next to the file
    subprocess.run([
        "whisper", str(src),
        "--model", MODEL,
        "--language", LANGUAGE,
        "--output_dir", str(src.parent),
        "--output_format", "all",        # ← .srt .vtt .txt .json .tsv
        "--word_timestamps", "True",     # ← optional: word-level timing
        "--highlight_words", "True"
    ], check=True)

    # Add "_subtitles" to every output file
    stem = src.stem
    for ext in ["srt", "vtt", "txt", "json", "tsv"]:
        old = src.parent / f"{stem}.{ext}"
        new = src.parent / f"{stem}_subtitles.{ext}"
        if old.exists():
            old.rename(new)

    print(f"Done → {stem}_subtitles.srt (and .vtt .txt .json .tsv)\n")

print("All transcriptions complete!")
