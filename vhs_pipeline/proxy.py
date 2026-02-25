from __future__ import annotations

import sys

from common import ARCHIVE_DIR, FFMPEG_BIN, METADATA_DIR, run


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
        run(
            [
                FFMPEG_BIN,
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
                "setpts=N/(30000/1001*TB)",
                "-r",
                "30000/1001",
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
        )
        count += 1

    print(f"Created {count} proxies.")
    print("All done")
    return 0


def main(argv=None):
    _ = argv
    return make_proxies()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

