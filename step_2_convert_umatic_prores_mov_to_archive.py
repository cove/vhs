#
# Converts U-Matic source files (MOV) to archival MKV using FFV1 video and 24-bit PCM audio.
# For Digital Roots, Albany, CA services. Produces lossless preservation masters.
#
from common import *
import sys
import subprocess
from pathlib import Path


def main(input_file):
    input_path = Path(input_file)
    output_file = input_path.with_name(input_path.stem + "_archive.mkv")
    ffmetadata_path = METADATA_DIR / output_file.stem / "chapters.ffmetadata"

    print(f"Converting: {input_path.name} -> {output_file.name}")

    cmd = [
        str(FFMPEG_BIN),
        "-nostdin", "-v", "error", "-stats",
        "-i", str(input_path),
    ]

    if ffmetadata_path.exists():
        cmd += ["-f", "ffmetadata", "-i", str(ffmetadata_path)]
        cmd += ["-map_metadata", "1", "-map_chapters", "1"]
        print(f"  Embedding metadata from: {ffmetadata_path}")
    else:
        print(f"  Metadata not found, skipping embed: {ffmetadata_path}")

    cmd += [
        "-pix_fmt", "yuv422p10",
        "-map", "0:v:0",
        "-c:v", "ffv1",
        "-level", "3",
        "-g", "1",
        "-coder", "1",
        "-context", "1",
        "-slices", "24", "-slicecrc", "1",
        "-map", "0:a",
        "-c:a", "pcm_s24le",
        "-y", str(output_file),
    ]

    subprocess.run(cmd, check=True)
    print(f"Done converting {output_file.name}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_umatic.py video1.mov video2.mov ...")
        sys.exit(1)
    main(sys.argv[1])
