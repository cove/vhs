from __future__ import annotations

import subprocess
from pathlib import Path


def _normalize_inputs(paths):
    values = [Path(p) for p in (paths or [])]
    if not values:
        raise ValueError("No input files provided.")
    return values


def _build_archive_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_archive.mkv")


def _build_ffmetadata_path(metadata_dir: Path, archive_stem: str) -> Path:
    return Path(metadata_dir) / archive_stem / "chapters.ffmetadata"


def _run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def _common_config():
    from common import FFMPEG_BIN, METADATA_DIR, ensure_ffmpeg_exists

    return FFMPEG_BIN, METADATA_DIR, ensure_ffmpeg_exists


def _convert_file(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg_bin: Path,
    metadata_dir: Path,
    pixel_format: str,
    audio_codec: str,
    include_sd_color_tags: bool,
):
    ffmetadata_path = _build_ffmetadata_path(metadata_dir, output_path.stem)

    print(f"Converting: {input_path.name} -> {output_path.name}")

    cmd = [
        str(ffmpeg_bin),
        "-nostdin",
        "-v",
        "error",
        "-stats",
        "-i",
        str(input_path),
    ]

    if ffmetadata_path.exists():
        cmd += ["-f", "ffmetadata", "-i", str(ffmetadata_path)]
        cmd += ["-map_metadata", "1", "-map_chapters", "1"]
        print(f"  Embedding metadata from: {ffmetadata_path}")
    else:
        print(f"  Metadata not found, skipping embed: {ffmetadata_path}")

    cmd += [
        "-pix_fmt",
        str(pixel_format),
    ]

    if include_sd_color_tags:
        cmd += [
            "-color_primaries:v",
            "6",
            "-color_trc:v",
            "6",
            "-colorspace:v",
            "5",
            "-color_range:v",
            "1",
        ]

    cmd += [
        "-map",
        "0:v:0",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-coder",
        "1",
        "-context",
        "1",
        "-slices",
        "24",
        "-slicecrc",
        "1",
        "-map",
        "0:a",
        "-c:a",
        str(audio_codec),
        "-y",
        str(output_path),
    ]
    _run(cmd)
    print(f"Done converting {output_path.name}\n")


def convert_avi_to_archive(paths):
    ffmpeg_bin, metadata_dir, ensure_ffmpeg_exists = _common_config()
    ensure_ffmpeg_exists()
    count = 0
    for file in _normalize_inputs(paths):
        if not file.exists():
            print(f"File not found: {file}")
            continue
        output = _build_archive_output_path(file)
        _convert_file(
            input_path=file,
            output_path=output,
            ffmpeg_bin=ffmpeg_bin,
            metadata_dir=metadata_dir,
            pixel_format="yuv422p",
            audio_codec="pcm_s16le",
            include_sd_color_tags=True,
        )
        count += 1
    print("All finished!")
    return count


def convert_umatic_to_archive(paths):
    ffmpeg_bin, metadata_dir, ensure_ffmpeg_exists = _common_config()
    ensure_ffmpeg_exists()
    count = 0
    for file in _normalize_inputs(paths):
        if not file.exists():
            print(f"File not found: {file}")
            continue
        output = _build_archive_output_path(file)
        _convert_file(
            input_path=file,
            output_path=output,
            ffmpeg_bin=ffmpeg_bin,
            metadata_dir=metadata_dir,
            pixel_format="yuv422p10",
            audio_codec="pcm_s24le",
            include_sd_color_tags=False,
        )
        count += 1
    print("All finished!")
    return count


def embed_metadata_into_archives(paths):
    ffmpeg_bin, metadata_dir, ensure_ffmpeg_exists = _common_config()
    ensure_ffmpeg_exists()
    count = 0
    for file in _normalize_inputs(paths):
        src = Path(file)
        if not src.exists():
            print(f"File not found: {src}")
            continue
        if src.suffix.lower() != ".mkv":
            print(f"Skipping non-MKV: {src}")
            continue

        ffmetadata_path = _build_ffmetadata_path(metadata_dir, src.stem)
        if not ffmetadata_path.exists():
            print(f"Metadata not found, skipping: {ffmetadata_path}")
            continue

        tmp = src.with_suffix(".metadata.tmp.mkv")
        backup = src.with_suffix(".pre-metadata.mkv")

        cmd = [
            str(ffmpeg_bin),
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
            "0",
            "-c",
            "copy",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-y",
            str(tmp),
        ]

        print(f"Embedding metadata: {src.name}")
        _run(cmd)
        backup.unlink(missing_ok=True)
        src.replace(backup)
        tmp.replace(src)
        print(f"Updated: {src.name} (backup: {backup.name})")
        count += 1
    return count
