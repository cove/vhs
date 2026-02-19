#!/usr/bin/env python3.11
#
# Builds side-by-side chapter comparison videos:
# left = original chapter window (sourced from archive proxy for speed)
# right = processed chapter output from step_6_make_videos.py
#
import argparse
from pathlib import Path

from common import *


DEFAULT_HEIGHT = 480
OUTPUT_SUFFIX = "_original_vs_processed.mp4"
TARGET_FPS = "60000/1001"


def title_selected(title, filters):
    if not filters:
        return True
    text = str(title or "").strip().lower()
    for f in filters:
        needle = str(f or "").strip().lower()
        if needle and needle in text:
            return True
    return False


def archive_selected(name, filters):
    if not filters:
        return True
    text = str(name or "").strip().lower()
    for f in filters:
        needle = str(f or "").strip().lower()
        if needle and needle in text:
            return True
    return False


def find_processed_chapter_mp4(title):
    chapter_name = f"{safe(title)}.mp4"
    candidates = [
        VIDEOS_DIR / chapter_name,
        CLIPS_DIR / chapter_name,
    ]
    for p in candidates:
        if is_chapter_done(p):
            return p
    return None


def make_side_by_side(original_source_path, processed_path, start_sec, duration_sec, out_path, height):
    filter_complex = (
        f"[0:v]setpts=PTS-STARTPTS,fps={TARGET_FPS},scale=-2:{height}:flags=lanczos,setsar=1[left];"
        f"[1:v]setpts=PTS-STARTPTS,fps={TARGET_FPS},scale=-2:{height}:flags=lanczos,setsar=1[right];"
        "[left][right]hstack=inputs=2[v]"
    )

    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        str(original_source_path),
        "-i",
        str(processed_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-an",
        "-shortest",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-r",
        TARGET_FPS,
        "-movflags",
        "+faststart",
        "-y",
        str(out_path),
    ]
    run(cmd)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Create side-by-side chapter comparisons (original vs processed chapter)."
    )
    p.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Only process archive names containing this substring (case-insensitive). Repeatable.",
    )
    p.add_argument(
        "--title",
        action="append",
        default=[],
        help="Only process chapter titles containing this substring (case-insensitive). Repeatable.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Target height for each side before stacking (default: {DEFAULT_HEIGHT}).",
    )
    p.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max number of comparisons to create (0 = no limit).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even if they already exist.",
    )
    p.add_argument(
        "--output-root",
        default=str(CLIPS_DIR / "chapter_comparisons"),
        help="Output root directory for comparison videos.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ensure_ffmpeg_exists()

    if args.height < 120:
        raise ValueError("--height must be at least 120")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped_existing = 0
    skipped_missing_processed = 0
    skipped_missing_inputs = 0

    archive_mkvs = sorted(ARCHIVE_DIR.glob("*.mkv"), key=lambda p: p.name.lower())
    if not archive_mkvs:
        print(f"No archive MKVs found in {ARCHIVE_DIR}")
        return

    for src in archive_mkvs:
        archive_name = src.stem
        if not archive_selected(archive_name, args.archive):
            continue

        original_source_path = ARCHIVE_DIR / f"{archive_name}_proxy.mp4"
        chapters_file = METADATA_DIR / archive_name / "chapters.ffmetadata"
        if not original_source_path.exists():
            print(f"Skipping {archive_name}: missing original source clip {original_source_path}")
            skipped_missing_inputs += 1
            continue
        if not chapters_file.exists():
            print(f"Skipping {archive_name}: missing chapters metadata {chapters_file}")
            skipped_missing_inputs += 1
            continue

        _, chapters = parse_chapters(chapters_file)
        if not chapters:
            print(f"Skipping {archive_name}: no chapters found")
            continue

        out_dir = output_root / archive_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, ch in enumerate(chapters, start=1):
            title = str(ch.get("title", "")).strip()
            if not title:
                continue
            if not title_selected(title, args.title):
                continue

            start_sec = float(ch.get("start", 0.0))
            end_sec = float(ch.get("end", 0.0))
            duration_sec = max(0.0, end_sec - start_sec)
            if duration_sec <= 0.0:
                print(f"Skipping chapter with invalid duration: {title}")
                continue

            processed_path = find_processed_chapter_mp4(title)
            if not processed_path:
                print(f"Skipping {archive_name} / {title}: missing processed chapter MP4")
                skipped_missing_processed += 1
                continue

            out_name = f"{idx:02d}_{safe(title)}{OUTPUT_SUFFIX}"
            out_path = out_dir / out_name

            if is_chapter_done(out_path) and not args.overwrite:
                print(f"Skipping existing comparison: {out_path.name}")
                skipped_existing += 1
                continue

            print(f"Building comparison: {archive_name} / {title}")
            make_side_by_side(
                original_source_path=original_source_path,
                processed_path=processed_path,
                start_sec=start_sec,
                duration_sec=duration_sec,
                out_path=out_path,
                height=args.height,
            )
            created += 1

            if args.max > 0 and created >= args.max:
                print(f"Reached --max={args.max}.")
                print(f"Created {created} comparison video(s).")
                return

    print(f"Created {created} comparison video(s).")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped missing processed chapter MP4: {skipped_missing_processed}")
    print(f"Skipped missing original-source/metadata inputs: {skipped_missing_inputs}")
    print("Done.")


if __name__ == "__main__":
    main()
