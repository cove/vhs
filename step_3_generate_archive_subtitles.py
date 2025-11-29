#!/usr/bin/env python3
# whisper_flash_attn_insane_speed.py
# 2025 maximum-speed Whisper — Flash Attention 2 + batched + GPU

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from pathlib import Path

ARCHIVE = Path("../Archive")
VIDEOS = Path("../Videos")
VIDEOS.mkdir(exist_ok=True)

# Check GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Flash Attention 2 = 2–4× faster than normal attention
attn = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
    model_kwargs={"attn_implementation": attn},
)

print(f"Whisper ready — Flash Attention: {attn == 'flash_attention_2'}")

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    print(f"Transcribing: {src.name}")

    outputs = pipe(
        str(src),
        chunk_length_s=30,
        batch_size=24,              # ← maximum batch size (uses all VRAM)
        return_timestamps=True,
        generate_kwargs={"language": "en"}
    )

    base = VIDEOS / f"{name}_subtitles"

    # Save all formats
    import json
    with open(base.with_suffix(".srt"), "w", encoding="utf-8") as f:
        f.write(outputs["text"])  # simple SRT from chunks
    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(outputs["text"])
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Done → {name}_subtitles.srt (.txt .json)\n")

print("All done")
