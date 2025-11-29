import sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
ARCHIVE = BASE.parent / "Archive"
PROXIES = BASE.parent / "Proxies"
PROXIES.mkdir(exist_ok=True)

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg not found at {FFMPEG}")
    sys.exit(1)

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

print(f"Generating ½-size proxies → {PROXIES}\n")

for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    proxy = PROXIES / f"{name}_proxy.mp4"

    if proxy.exists():
        print(f"Skipping {src.name} (proxy exists)")
        continue

    print(f"Processing: {src.name} → {proxy.name}")

    run([
        FFMPEG, "-v", "error",
        "-i", str(src),
        "-vf", "scale=iw/2:ih/2",              # exactly ½ size
        "-preset", "fast",
        "-crf", "28",                          # tiny file, still clear
        "-c:a", "aac", "-b:a", "32k",
        "-y", str(proxy)
    ])

print("\nAll proxies done")