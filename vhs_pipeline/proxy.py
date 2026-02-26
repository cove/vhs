from __future__ import annotations

import sys
from pathlib import Path

from common import ARCHIVE_DIR, FFMPEG_BIN, METADATA_DIR, run

PROXY_FPS = "30000/1001"


def build_proxy_command(src: Path, ffmetadata_path: Path, proxy: Path) -> list[str]:
    # Keep frame-index lockstep with archive while reducing proxy size.
    font_expr = ""
    win_font = Path("C:/Windows/Fonts/consola.ttf")
    if win_font.exists():
        font_expr = "fontfile='C\\:/Windows/Fonts/consola.ttf'"
    drawtext = (
        "drawtext="
        + "text='frame=%{eif\\:n\\:d}'"
        + (f":{font_expr}" if font_expr else "")
        + ":x=16:y=16:fontsize=24:"
        + "fontcolor=white:box=1:boxcolor=black@0.55:borderw=2"
    )
    vf = f"scale=iw/2:ih/2:flags=lanczos,setpts=N/(30000/1001*TB),{drawtext}"
    return [
        str(FFMPEG_BIN),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(src),
        "-f",
        "ffmetadata",
        "-i",
        str(ffmetadata_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-map_metadata",
        "1",
        "-vf",
        vf,
        "-r",
        PROXY_FPS,
        "-fps_mode:v:0",
        "cfr",
        "-vsync",
        "cfr",
        "-video_track_timescale",
        "30000",
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-tune",
        "fastdecode",
        "-crf",
        "28",
        "-x264-params",
        "keyint=30:min-keyint=1:scenecut=40",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "48k",
        "-ar",
        "48000",
        "-ac",
        "1",
        "-movflags",
        "+faststart+use_metadata_tags",
        "-y",
        str(proxy),
    ]


def make_proxies():
    print(f"Generating PROXY {ARCHIVE_DIR}\n")
    count = 0
    for src in ARCHIVE_DIR.glob("*.mkv"):
        archive = src.stem
        proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
        ffmetadata_path = METADATA_DIR / archive / "chapters.ffmetadata"

        if proxy.exists() and proxy.stat().st_size > 100_000:
            print(f"Skipping {proxy} (already processed)")
            continue

        if not ffmetadata_path.exists():
            print(f"Skipping {src.name}: metadata not found: {ffmetadata_path}")
            continue

        print(f"Processing: {src.name} {proxy.name}")
        run(build_proxy_command(src, ffmetadata_path, proxy))
        count += 1

    print(f"Created {count} proxies.")
    print("All done")
    return 0


def main(argv=None):
    _ = argv
    return make_proxies()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

