#
# Injects ffmetadata (global tags + chapters) into existing archive MKV files
# without re-encoding. Safe to run multiple times.
#
import sys
import subprocess
from pathlib import Path

from common import FFMPEG_BIN, METADATA_DIR, ensure_ffmpeg_exists


def main(paths):
    ensure_ffmpeg_exists()

    for file in paths:
        src = Path(file)
        if not src.exists():
            print(f"File not found: {src}")
            continue

        if src.suffix.lower() != ".mkv":
            print(f"Skipping non-MKV: {src}")
            continue

        archive_stem = src.stem
        ffmetadata_path = METADATA_DIR / archive_stem / "chapters.ffmetadata"
        if not ffmetadata_path.exists():
            print(f"Metadata not found, skipping: {ffmetadata_path}")
            continue

        tmp = src.with_suffix(".metadata.tmp.mkv")

        cmd = [
            str(FFMPEG_BIN),
            "-nostdin", "-v", "error",
            "-i", str(src),
            "-f", "ffmetadata", "-i", str(ffmetadata_path),
            "-map", "0",
            "-c", "copy",
            "-map_metadata", "1",
            "-map_chapters", "1",
            "-y", str(tmp),
        ]

        print(f"Embedding metadata: {src.name}")
        subprocess.run(cmd, check=True)

        src.replace(src.with_suffix(".pre-metadata.mkv"))
        tmp.replace(src)
        print(f"Updated: {src.name} (backup: {src.with_suffix('.pre-metadata.mkv').name})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step_2b_embed_metadata.py archive1.mkv archive2.mkv ...")
        sys.exit(1)
    main(sys.argv[1:])
