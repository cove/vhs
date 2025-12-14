#
# Converts one or more MKV files to archival MKV using FFV1 video and 16-bit PCM audio.
# Produces lossless preservation masters suitable for long-term storage.
#
from common import *

def main():
    if len(sys.argv) < 2:
        print("Usage: python this_script.py video1.mkv video2.mkv ...")
        sys.exit(1)

    for file in sys.argv[1:]:
        file_path = Path(file)
        if not file_path.exists():
            print(f"File not found: {file}")
            continue

        output = file_path.with_name(file_path.stem + "_archive.mkv")

        print(f"Converting: {file_path.name}  →  {output.name}")

        cmd = [
            str(FFMPEG_BIN),
            "-nostdin", "-v", "error", "-stats",
            "-i", str(file_path),
            "-pix_fmt", "yuv422p",
            "-map", "0:v:0",
            "-c:v", "ffv1",
            "-level", "3",
            "-g", "1",
            "-coder", "1",
            "-context", "1",
            "-slices", "24", "-slicecrc", "1",
            "-map", "0:a", "-c:a", "pcm_s16le",
            "-y", str(output)
        ]

        subprocess.run(cmd, check=True)
        print(f"Done converting {file_path.name}\n")

    print("All finished!")

if __name__ == "__main__":
    main()
