#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from step_16_classify_badframes_tracking_loss import (
    expand_ranges_to_set,
    parse_badframe_ranges,
    run_tracking_loss_classification,
)
from step_6_make_videos import resolve_badframes_tsv, run_make_videos


ARCHIVE = "callahan_01_archive"
VIDEO_PATH = REPO_ROOT.parent / "Archive" / f"{ARCHIVE}.mkv"
OUTPUT_DIR = REPO_ROOT / "metadata" / ARCHIVE / "tracking_badframe_full_video"
ARCHIVE_BADFRAMES_TSV = REPO_ROOT / "metadata" / ARCHIVE / "badframes.tsv"
CHECK_WINDOW_START = 6000
CHECK_WINDOW_END = 7000
MIN_BAD_FRAMES_IN_WINDOW = 50
TITLE_FILTERS = ["Jim & Linda Wedding"]
NO_BOB = False


def count_bad_frames_in_window(badframes_tsv: Path, start_frame: int, end_frame: int) -> int:
    ranges = parse_badframe_ranges(badframes_tsv)
    bad_frames = expand_ranges_to_set(ranges, int(start_frame), int(end_frame))
    return int(len(bad_frames))


def validate_step6_badframes_path(expected_badframes_tsv: Path) -> Path:
    resolved = resolve_badframes_tsv(
        ARCHIVE,
        override_path=expected_badframes_tsv,
        override_archive=ARCHIVE,
    )
    if not resolved:
        raise RuntimeError("step_6 badframes resolution returned no path.")

    resolved_path = Path(resolved).resolve()
    expected_path = Path(expected_badframes_tsv).resolve()
    if resolved_path != expected_path:
        raise RuntimeError(
            f"step_6 badframes path mismatch: resolved={resolved_path} expected={expected_path}"
        )
    return resolved_path


def main() -> None:
    result = run_tracking_loss_classification(
        archive=ARCHIVE,
        video=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        existing_badframes=ARCHIVE_BADFRAMES_TSV,
        scores_tsv=None,
        start_frame=0,
        max_frame=-1,
        frame_step=1,
        crop_top=50,
        crop_bottom=50,
        crop_left=50,
        crop_right=50,
        sobel_ksize=3,
        weight_edge=0.45,
        weight_row=0.25,
        weight_field=0.30,
        otsu_bins=256,
        threshold_mode="ostu",
        bad_rate=-1.0,
        threshold_value=1.2,
        calibrate_bad_rate_from_existing=False,
        export_bad_png_count=0,
    )
#
#
#        crop_top=0,
#        crop_bottom=0,
#        crop_left=0,
#        crop_right=0,
#      threshold_mode="value",
#      bad_rate=-1.0,
#      threshold_value=1.2,
#
    generated_badframes_tsv = Path(result["badframes_path"])
    if not generated_badframes_tsv.exists():
        raise FileNotFoundError(
            f"Expected badframes output missing after step_16: {generated_badframes_tsv}"
        )

    bad_count_window = count_bad_frames_in_window(
        generated_badframes_tsv,
        start_frame=CHECK_WINDOW_START,
        end_frame=CHECK_WINDOW_END,
    )
    if bad_count_window < MIN_BAD_FRAMES_IN_WINDOW:
        raise RuntimeError(
            "Tracking-loss sanity check failed: "
            f"predicted bad frames in {CHECK_WINDOW_START}-{CHECK_WINDOW_END} = {bad_count_window}, "
            f"expected at least {MIN_BAD_FRAMES_IN_WINDOW}. "
            "Investigate detector tuning/output before replacing metadata badframes.tsv."
        )

    ARCHIVE_BADFRAMES_TSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated_badframes_tsv, ARCHIVE_BADFRAMES_TSV)
    if generated_badframes_tsv.read_bytes() != ARCHIVE_BADFRAMES_TSV.read_bytes():
        raise RuntimeError(
            "badframes.tsv content mismatch after copy; expected archive badframes.tsv to match classify output."
        )
    resolved_step6_badframes = validate_step6_badframes_path(ARCHIVE_BADFRAMES_TSV)

    # Explicit override to guarantee step_6 uses the archive badframes.tsv we just updated.
    run_make_videos(
        title_filters=TITLE_FILTERS,
        no_bob=NO_BOB,
        badframes_tsv=ARCHIVE_BADFRAMES_TSV,
        badframes_archive=ARCHIVE,
    )

    print("Done.")
    print(f"Tracking summary: {result['summary_path']}")
    print(f"Generated badframes: {generated_badframes_tsv}")
    print(f"Updated archive badframes: {ARCHIVE_BADFRAMES_TSV}")
    print(f"step_6 resolved badframes path: {resolved_step6_badframes}")
    print(
        f"Sanity check window {CHECK_WINDOW_START}-{CHECK_WINDOW_END}: "
        f"{bad_count_window} predicted bad frames"
    )


if __name__ == "__main__":
    main()
