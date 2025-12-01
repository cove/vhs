import sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
ARCHIVE = BASE.parent / "Archive"
PROXY = BASE.parent / "Proxy"
PROXY.mkdir(exist_ok=True)

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg not found at {FFMPEG}")
    sys.exit(1)

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

print(f"Generating ½-size PROXY → {PROXY}\n")

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    proxy = PROXY / f"{name.replace("_archive", "_small_proxy")}.mp4"

    if proxy.exists():
        print(f"Skipping {src.name} (proxy exists)")
        continue

    print(f"Processing: {src.name} → {proxy.name}")

    run([
        FFMPEG, "-v", "error",
        "-i", str(src),
        "-vf", "scale=iw/2:ih/2",  # ½ size
        "-c:v", "libx265",
        "-preset", "ultrafast",  # ← fastest preset
        "-crf", "28",  # ← 28 = tiny but still clear enough
        "-tune", "fastdecode",  # ← smaller file, faster decode
        "-x265-params", "no-sao=1:rect=0:strong-intra-smoothing=0",
        "-pix_fmt", "yuv420p",  # 8-bit (10-bit = waste for proxy)
        "-c:a", "aac", "-b:a", "16k",  # ← 16 kbps mono = almost nothing
        "-ac", "1",  # force mono
        "-tag:v", "hvc1",
        "-brand", "mp42",
        "-movflags", "+faststart+write_colr+use_metadata_tags",
        "-y", str(proxy)
    ])

print("\nAll PROXY done")
