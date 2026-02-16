#!/usr/bin/env python3.11
#
# Generates short QTGMC comparison clips and full-size 2x2 quadrant previews.
# Each case samples 30 frames from the midpoint of a selected chapter.
#
import re
from pathlib import Path

from common import *

SAMPLE_FRAMES = 30
FPS_NUM = 30000
FPS_DEN = 1001

TEST_CASES = [
    {
        "name": "wedding_vows_mid",
        "archive": "callahan_01_archive",
        "title_contains": "Wedding Vows",
    },
    {
        "name": "ultrasound_mid",
        "archive": "callahan_04_archive",
        "title_contains": "Ultra Sound - 01",
    },
]

BASE_QTGMC_OPTS = {
    "EZKeepGrain": 1.0,
    "Sharpness": 1.2,
    "SourceMatch": 3,
    "Lossless": 2,
    "TR2": 3,
    "MatchEdi": "NNEDI3",
    "MatchEdi2": "NNEDI3",
}

QTGMC_PRESETS = [
    "Medium",
    "Slow",
    "Very Slow",
    "Placebo",
]


def fmt_avs_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return f"\"{v}\""


def qtgmc_call(opts):
    parts = []
    for k, v in opts.items():
        parts.append(f"{k}={fmt_avs_value(v)}")
    return "QTGMC(" + ",".join(parts) + ")"


def sec_to_frame(seconds):
    return int(round(float(seconds) * FPS_NUM / FPS_DEN))


def chapter_mid_window(ch, sample_frames):
    start_frame = sec_to_frame(ch["start"])
    end_frame = sec_to_frame(ch["end"])
    if end_frame < start_frame:
        return start_frame, start_frame

    chapter_len = end_frame - start_frame + 1
    n = min(sample_frames, chapter_len)
    mid = (start_frame + end_frame) // 2
    clip_start = mid - (n // 2)
    clip_start = max(start_frame, clip_start)
    clip_end = clip_start + n - 1
    if clip_end > end_frame:
        clip_end = end_frame
        clip_start = max(start_frame, clip_end - n + 1)
    return clip_start, clip_end


def pick_chapter(chapters, title_contains):
    needle = (title_contains or "").strip().lower()
    if not needle:
        return chapters[0] if chapters else None
    for ch in chapters:
        title = str(ch.get("title", "")).lower()
        if needle in title:
            return ch
    return None


def rewrite_filter_qtgmc(filter_text, qtgmc_opts):
    out = []
    replaced = False
    pattern = re.compile(r"^\s*QTGMC\s*\(")
    for line in filter_text.splitlines():
        if (not replaced) and pattern.search(line):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + qtgmc_call(qtgmc_opts))
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise RuntimeError("Could not find QTGMC(...) in filter script.")
    return "\n".join(out) + "\n"


def build_avs(src_path, start_frame, end_frame, filter_text):
    src = str(src_path).replace("\\", "/")
    lines = [
        f'LoadPlugin("{QTGMC_DIR}/ffms2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/masktools2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/Rgtools.dll")',
        f'LoadPlugin("{QTGMC_DIR}/mvtools2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/DePanEstimate.dll")',
        f'LoadPlugin("{QTGMC_DIR}/DePan.dll")',
        f'LoadPlugin("{QTGMC_DIR}/nnedi3.dll")',
        f'LoadPlugin("{QTGMC_DIR}/yadifmod2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/fft3dfilter.dll")',
        f'LoadPlugin("{QTGMC_DIR}/LoadDLL64.dll")',
        f'LoadPlugin("{QTGMC_DIR}/SmoothAdjust.dll")',
        f'LoadDLL("{QTGMC_DIR}/libfftw3f-3.dll")',
        f'Import("{QTGMC_DIR}/Zs_RF_Shared.avsi")',
        f'Import("{QTGMC_DIR}/QTGMC.avsi")',
        f'FFmpegSource2("{src}", atrack=-1)',
        f"Trim({start_frame},{end_frame})",
    ]
    return "\n".join(lines) + "\n" + filter_text


def encode_variant(avs_path, out_path):
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(avs_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        "-y", str(out_path),
    ]
    run(cmd)


def make_full_size_quadrant(inputs, out_path):
    if len(inputs) != 4:
        raise ValueError(f"Quadrant needs 4 inputs, got {len(inputs)}")
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
    ]
    for f in inputs:
        cmd += ["-i", str(f)]
    cmd += [
        "-filter_complex",
        "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[v]",
        "-map", "[v]",
        "-an",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-y", str(out_path),
    ]
    run(cmd)


def main():
    ensure_ffmpeg_exists()

    base_dir = CLIPS_DIR / "filter_tests" / "qtgmc_preset_quadrants"
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base_dir.glob("run_*"):
        if p.is_dir():
            try:
                existing.append(int(p.name.split("_", 1)[1]))
            except (ValueError, IndexError):
                continue
    next_run = (max(existing) + 1) if existing else 1
    out_dir = base_dir / f"run_{next_run:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    avs_dir = out_dir / "avs"
    avs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir}")
    summary_lines = []
    generated_cases = 0

    for case in TEST_CASES:
        archive_stem = case["archive"]
        archive_path = ARCHIVE_DIR / f"{archive_stem}.mkv"
        chapters_file = METADATA_DIR / archive_stem / "chapters.ffmetadata"
        filter_path = METADATA_DIR / archive_stem / "filter.avs"

        if not archive_path.exists():
            print(f"Skipping {case['name']}: missing archive {archive_path}")
            continue
        if not chapters_file.exists():
            print(f"Skipping {case['name']}: missing chapters {chapters_file}")
            continue
        if not filter_path.exists():
            print(f"Skipping {case['name']}: missing filter {filter_path}")
            continue

        _, chapters = parse_chapters(chapters_file)
        ch = pick_chapter(chapters, case.get("title_contains", ""))
        if not ch:
            print(f"Skipping {case['name']}: no chapter matched '{case.get('title_contains', '')}'")
            continue

        start_frame, end_frame = chapter_mid_window(ch, SAMPLE_FRAMES)
        print(
            f"Case {case['name']}: {archive_stem} | {ch.get('title')} | "
            f"frames {start_frame}-{end_frame}"
        )

        base_filter = filter_path.read_text(encoding="utf-8")
        case_outputs = []
        for idx, preset in enumerate(QTGMC_PRESETS, start=1):
            qopts = dict(BASE_QTGMC_OPTS)
            qopts["Preset"] = preset
            patched_filter = rewrite_filter_qtgmc(base_filter, qopts)

            avs_name = safe(f"{case['name']}_{idx:02d}_{preset.lower().replace(' ', '_')}")
            avs_path = avs_dir / f"{avs_name}.avs"
            out_path = out_dir / f"{avs_name}.mp4"

            avs_text = build_avs(archive_path, start_frame, end_frame, patched_filter)
            avs_path.write_text(avs_text, encoding="utf-8")
            encode_variant(avs_path, out_path)
            case_outputs.append(out_path)

            summary_lines.append(
                f"{out_path.name}\tcase={case['name']}\tarchive={archive_stem}\t"
                f"title={ch.get('title')}\tpreset={preset}\tframes={start_frame}-{end_frame}"
            )

        quadrant_path = out_dir / f"quadrant_{safe(case['name'])}.mp4"
        make_full_size_quadrant(case_outputs, quadrant_path)
        summary_lines.append(
            f"{quadrant_path.name}\tcase={case['name']}\tarchive={archive_stem}\t"
            f"title={ch.get('title')}\tquadrant_presets={','.join(QTGMC_PRESETS)}\t"
            f"frames={start_frame}-{end_frame}"
        )
        generated_cases += 1

    if generated_cases == 0:
        print("No cases generated.")
        sys.exit(1)

    (out_dir / "variants.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Generated cases: {generated_cases}")
    print(f"Presets per case: {len(QTGMC_PRESETS)}")


if __name__ == "__main__":
    main()
