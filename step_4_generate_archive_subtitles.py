import whisper
from pathlib import Path

# Load model once (large-v3 = best quality)
model = whisper.load_model("large-v3")

ARCHIVE = Path("../Archive")
VIDEOS = Path("../Videos")
VIDEOS.mkdir(exist_ok=True)

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    print(f"Transcribing: {src.name}")

    # Transcribe directly in Python
    result = model.transcribe(
        str(src),
        language="en",
        word_timestamps=True,
        fp16=False,           # True = faster on GPU, False = stable on CPU
    )

    # Save all formats with "_subtitles" suffix
    base = src.parent / name

    # SRT
    with open(base.with_suffix("_subtitles.srt"), "w", encoding="utf-8") as f:
        whisper.utils.write_srt(result["segments"], file=f)

    # VTT
    with open(base.with_suffix("_subtitles.vtt"), "w", encoding="utf-8") as f:
        whisper.utils.write_vtt(result["segments"], file=f)

    # Plain text
    with open(base.with_suffix("_subtitles.txt"), "w", encoding="utf-8") as f:
        f.write(result["text"])

    # JSON (full data)
    import json
    with open(base.with_suffix("_subtitles.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done → {name}_subtitles.srt (and .vtt .txt .json)\n")

print("All done — pure Python Whisper!")