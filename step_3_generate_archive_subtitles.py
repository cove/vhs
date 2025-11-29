import sys
from pathlib import Path
import json
import torch
from faster_whisper import WhisperModel
from faster_whisper.utils import write_srt, write_vtt, write_txt

ARCHIVE = Path("../Archive")
VIDEOS = Path("../Videos")
VIDEOS.mkdir(exist_ok=True)

# Detect ROCm GPU (AMD) – faster-whisper cannot use it
if torch.version.hip:
    print("AMD ROCm GPU detected — faster-whisper does not support GPU on ROCm. Using CPU.\n")
else:
    print("Running on CPU.\n")

# Best CPU setting for AMD systems in 2025
model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8_float16",       # fastest accurate CPU mode
    cpu_threads=12,
    num_workers=4,
    download_root=str(VIDEOS / ".cache")
)

print("Whisper model loaded (CPU / int8_float16)\n")

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    print(f"Transcribing: {src.name}")

    segments, info = model.transcribe(
        str(src),
        language="en",
        beam_size=5,
        best_of=5,
        patience=1.0,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=True,
        prepend_punctuations="\"'“¿([{-",
        append_punctuations="\"'.。,，!！?？:;)]}-",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    base = src.parent / f"{name}_eng_subtitles"

    with open(base.with_suffix(".srt"), "w", encoding="utf-8") as f:
        write_srt(segments, file=f)
    with open(base.with_suffix(".vtt"), "w", encoding="utf-8") as f:
        write_vtt(segments, file=f)
    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        write_txt(segments, file=f)
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {"segments": [s._asdict() for s in segments]},
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Done → {name}_eng_subtitles.*\n")

print("All done.")
