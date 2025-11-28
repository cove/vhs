import os, glob, sys, subprocess
from pathlib import Path

ARCHIVE = Path("../Archive")
FFMPEG = Path("software/FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe")
FFPROBE = Path("bin/ffprobe.exe")

out_manifest = ARCHIVE / "00-manifest-blake3sums.txt"
out_manifest.unlink(missing_ok=True)

files = glob.glob(str(ARCHIVE / "*.mkv"))
if not files:
    print("No .mkv files found.")
    sys.exit(0)

def duration(path):
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True,
        ).strip()
        return float(out) if out else 0
    except:
        return 0

for mkv in files:
    if mkv.endswith("_metadata.mkv"):
        print("Skipping:", mkv)
        continue

    p = Path(mkv)
    name = p.stem
    prefix = "_".join(name.rsplit("_", 2)[:2])
    meta_dir = Path("media_metadata") / prefix

    title = (meta_dir / "title.txt").read_text().strip()
    comment = (meta_dir / "comment.txt").read_text().strip()
    chapters = meta_dir / "chapters.ffmetadata"
    cover = meta_dir / "cover.jpg"

    temp = p.with_name(f"{name}_metadata.mkv")
    final = p

    print(f"Processing: {p.name}")

    r = subprocess.run([
        str(FFMPEG), "-nostdin", "-v", "warning", "-stats",
        "-i", str(p), "-f", "ffmetadata", "-i", str(chapters),
        "-map", "0:v:0", "-map", "0:a",
        "-map_metadata", "0", "-map_chapters", "-1", "-map_chapters", "1",
        "-c", "copy",
        "-metadata:s:v:0", "avg_frame_rate=30000/1001",
        "-metadata:s:a:0", "channel_layout=mono",
        "-metadata", f"title={title}",
        "-metadata", f"comment={comment}",
        "-attach", str(cover),
        "-metadata:s:t:0", "mimetype=image/jpeg",
        "-metadata:s:t:0", "filename=cover.jpg",
        "-color_primaries:v", "6", "-color_trc:v", "6", "-colorspace:v", "5",
        "-aspect", "4:3",
        "-f", "matroska", "-y", str(temp),
        ], capture_output=True, text=True)
    if r.returncode:
        print("FFmpeg failed:\n", r.stderr)
        temp.unlink(missing_ok=True)
        continue

    # Basic FFmpeg validity check
    print(f"Verifying: {p.name}")
    v = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(temp), "-f", "null", "-"],
        stderr=subprocess.PIPE, text=True
    )
    if v.stderr.strip():
        print("Validation error:\n", v.stderr)
        continue

    # Duration check
    if abs(duration(mkv) - duration(temp)) > 1:
        print("Duration mismatch.")
        continue

    temp.replace(final)
    print("Success →", final.name, "\n")

print("All done.")
