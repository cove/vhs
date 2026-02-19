#!/usr/bin/env python3.11
#
# Builds full-size 2x2 quadrant previews for filter test runs.
#
import re
import subprocess
from pathlib import Path

from common import *

TARGET_ARCHIVE = "qtgmc_preset_quadrants"
RUN_DIR = ""  # optional: "run_001"
MAX_TILES = 4


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
    return path.name


def segment_name(path: Path):
    stem = path.stem

    # New format: <segment>_<index>_<preset>
    m = re.match(r"(.+)_\d{2}_.+$", stem)
    if m:
        return m.group(1)

    # Legacy format: <index>_<segment>_...
    parts = stem.split("_", 2)
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]

    return stem


def probe_size(path: Path):
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w_str, h_str = out.split("x")
    return int(w_str), int(h_str)


def make_quadrant(files, out_path):
    files = list(files[:MAX_TILES])
    n = len(files)
    if n == 0:
        return False

    tile_w, tile_h = probe_size(files[0])
    inputs = []
    filters = []
    layout = []

    for i, f in enumerate(files):
        inputs += ["-i", str(f)]
        filters.append(f"[{i}:v]scale={tile_w}:{tile_h}[v{i}]")
        x = (i % 2) * tile_w
        y = (i // 2) * tile_h
        layout.append(f"{x}_{y}")

    filter_complex = ";".join(filters)
    filter_complex += ";" + "".join([f"[v{i}]" for i in range(n)])
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

    all_mp4 = [
        p for p in sorted(run_dir.glob("*.mp4"), key=sort_key)
        if not p.name.startswith(("quadrant_", "grid_"))
    ]
    if not all_mp4:
        print(f"No test videos found in {run_dir}")
        sys.exit(1)

    groups = {}
    for p in all_mp4:
        seg = segment_name(p)
        groups.setdefault(seg, []).append(p)

    created = 0
    for seg, files in sorted(groups.items()):
        out = run_dir / f"quadrant_{safe(seg)}.mp4"
        if make_quadrant(files, out):
            print("Created:", out)
            created += 1

    if created == 0:
        print("No quadrants created.")
        sys.exit(1)


if __name__ == "__main__":
    main()
