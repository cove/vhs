#
# Builds grid preview videos for filter test runs (darkness and wobble sets).
#
import math
from pathlib import Path

from common import *

TARGET_ARCHIVE = "callahan_01_archive"
RUN_DIR = ""  # optional: "run_001"
TILE_W = 320
TILE_H = 240


def pick_latest_run(base_dir: Path) -> Path | None:
    runs = []
    for p in base_dir.glob("run_*"):
        if not p.is_dir():
            continue
        try:
            runs.append((int(p.name.split("_", 1)[1]), p))
        except (ValueError, IndexError):
            continue
    if not runs:
        return None
    runs.sort(key=lambda t: t[0])
    return runs[-1][1]


def sort_key(path: Path):
    name = path.name
    prefix = name.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return name


def segment_name(path: Path):
    parts = path.name.split("_", 2)
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]
    return ""


def make_grid(files, out_path):
    n = len(files)
    if n == 0:
        return False

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    inputs = []
    filters = []
    layout = []

    for i, f in enumerate(files):
        inputs += ["-i", str(f)]
        filters.append(f"[{i}:v]scale={TILE_W}:{TILE_H}[v{i}]")
        x = (i % cols) * TILE_W
        y = (i // cols) * TILE_H
        layout.append(f"{x}_{y}")

    filter_complex = ";".join(filters) + ";" + "".join([f"[v{i}]" for i in range(n)])
    filter_complex += f"xstack=inputs={n}:layout=" + "|".join(layout) + ":fill=black[v]"

    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-an",
        "-shortest",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-y", str(out_path),
    ]
    run(cmd)
    return True


def main():
    ensure_ffmpeg_exists()

    base_dir = CLIPS_DIR / "filter_tests" / TARGET_ARCHIVE
    if RUN_DIR:
        run_dir = base_dir / RUN_DIR
    else:
        run_dir = pick_latest_run(base_dir)

    if not run_dir or not run_dir.exists():
        print(f"Run folder not found under: {base_dir}")
        sys.exit(1)

    all_mp4 = sorted(run_dir.glob("*.mp4"), key=sort_key)
    darkness = [p for p in all_mp4 if segment_name(p) == "darkness"]
    wobble = [p for p in all_mp4 if segment_name(p) == "wobble"]

    if not darkness and not wobble:
        print(f"No test videos found in {run_dir}")
        sys.exit(1)

    if darkness:
        out = run_dir / "grid_darkness.mp4"
        make_grid(darkness, out)
        print("Created:", out)

    if wobble:
        out = run_dir / "grid_wobble.mp4"
        make_grid(wobble, out)
        print("Created:", out)


if __name__ == "__main__":
    main()
