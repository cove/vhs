import whisper
from pathlib import Path

# Load model once (large-v2 = best quality)
model = whisper.load_model("large-v2")

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
        fp16=False,           # True = faster on GPU, False = stable on CPU
    )

    base = src.parent / f"{name}_en_subtitles"

    # SRT
    with open(base.with_suffix(".srt"), "w", encoding="utf-8") as f:
        whisper.utils.write_srt(result["segments"], file=f)

    # VTT
    with open(base.with_suffix(".vtt"), "w", encoding="utf-8") as f:
        whisper.utils.write_vtt(result["segments"], file=f)

    # Plain text
    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(result["text"])

    # JSON (full data)
    import json
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done → {name}.srt (and .vtt .txt .json)\n")

print("All done — pure Python Whisper!")