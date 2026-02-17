#!/usr/bin/env python3.11
#
# Focused deinterlace readability test for ultrasound text.
# Produces short clips for multiple variants and a 2x2 comparison.
#
import re
import sys
from pathlib import Path

from common import *

FPS_NUM = 30000
FPS_DEN = 1001
SAMPLE_FRAMES = 90

CASE = {
    "name": "ultrasound_text",
    "archive": "callahan_04_archive",
    "title_contains": "Ultrasound - 01",
}


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


def replace_qtgmc_line(filter_text, replacement_call):
    out = []
    replaced = False
    pat = re.compile(r"^\s*QTGMC\s*\(")
    for line in filter_text.splitlines():
        if (not replaced) and pat.search(line):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + replacement_call)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise RuntimeError("Could not find QTGMC(...) in filter script.")
    return "\n".join(out) + "\n"


def set_bool_toggle(filter_text, var_name, value):
    pat = re.compile(rf"(?mi)^(\s*{re.escape(var_name)}\s*=\s*)(true|false)\s*$")
    repl = rf"\1{'true' if value else 'false'}"
    out, n = pat.subn(repl, filter_text, count=1)
    if n == 0:
        return filter_text
    return out


def inject_horizontal_stroke_comp(filter_text, blend=0.30):
    marker = "return c"
    idx = filter_text.rfind(marker)
    if idx < 0:
        raise RuntimeError("Could not find final 'return c' to inject stroke compensation.")
    block = (
        "# one-off text readability compensation for horizontal strokes\n"
        "htext = c.ConvertToY8()\n"
        "hfix = mt_expand(htext, mode=\"vertical\")\n"
        f"htext = Merge(htext, hfix, {float(blend):.2f})\n"
        "c = c.MergeLuma(htext.ConvertToYV12(interlaced=false))\n"
        "return c"
    )
    return filter_text[:idx] + block + filter_text[idx + len(marker):]


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


def build_raw_avs(src_path, start_frame, end_frame):
    src = str(src_path).replace("\\", "/")
    lines = [
        f'LoadPlugin("{QTGMC_DIR}/ffms2.dll")',
        f'FFmpegSource2("{src}", atrack=-1)',
        f"Trim({start_frame},{end_frame})",
    ]
    return "\n".join(lines) + "\n"


def encode_variant(avs_path, out_path):
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(avs_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-y",
        str(out_path),
    ]
    run(cmd)


def make_quadrant(inputs, out_path):
    if len(inputs) != 4:
        raise ValueError(f"Quadrant needs 4 inputs, got {len(inputs)}")
    cmd = [FFMPEG_BIN, "-nostdin", "-v", "error"]
    for f in inputs:
        cmd += ["-i", str(f)]
    cmd += [
        "-filter_complex",
        "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[v]",
        "-map",
        "[v]",
        "-an",
        "-shortest",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(out_path),
    ]
    run(cmd)


def qtgmc_call(opts):
    parts = []
    for k, v in opts.items():
        if isinstance(v, bool):
            vv = "true" if v else "false"
        elif isinstance(v, (int, float)):
            vv = str(v)
        else:
            vv = f'"{v}"'
        parts.append(f"{k}={vv}")
    return "QTGMC(" + ",".join(parts) + ")"


def main():
    ensure_ffmpeg_exists()

    archive_stem = CASE["archive"]
    archive_path = ARCHIVE_DIR / f"{archive_stem}.mkv"
    chapters_file = METADATA_DIR / archive_stem / "chapters.ffmetadata"
    filter_path = METADATA_DIR / archive_stem / "filter.avs"

    if not archive_path.exists():
        raise FileNotFoundError(f"Missing archive: {archive_path}")
    if not chapters_file.exists():
        raise FileNotFoundError(f"Missing chapters: {chapters_file}")
    if not filter_path.exists():
        raise FileNotFoundError(f"Missing filter: {filter_path}")

    _, chapters = parse_chapters(chapters_file)
    ch = pick_chapter(chapters, CASE["title_contains"])
    if not ch:
        raise RuntimeError(f"No chapter matched '{CASE['title_contains']}'")

    start_frame, end_frame = chapter_mid_window(ch, SAMPLE_FRAMES)

    base_dir = CLIPS_DIR / "filter_tests" / "ultrasound_text_readability"
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
    print(
        f"Case: {CASE['name']} | {archive_stem} | {ch.get('title')} | "
        f"frames {start_frame}-{end_frame}"
    )

    base_filter = filter_path.read_text(encoding="utf-8")
    baseline_filter = base_filter
    hstroke_equalized_filter = set_bool_toggle(base_filter, "text_enhance", False)
    hstroke_equalized_filter = inject_horizontal_stroke_comp(hstroke_equalized_filter, blend=0.35)

    variants = [
        (
            "00_raw_control",
            build_raw_avs(archive_path, start_frame, end_frame),
        ),
        (
            "01_qtgmc_baseline",
            build_avs(archive_path, start_frame, end_frame, baseline_filter),
        ),
        (
            "02_qtgmc_text_tuned",
            build_avs(
                archive_path,
                start_frame,
                end_frame,
                replace_qtgmc_line(
                    baseline_filter,
                    qtgmc_call(
                        {
                            "Preset": "Slow",
                            "SourceMatch": 3,
                            "Lossless": 2,
                            "TR1": 1,
                            "TR2": 1,
                            "Rep0": 2,
                            "Rep2": 1,
                            "MatchEnhance": 0.7,
                            "MatchEdi": "NNEDI3",
                            "MatchEdi2": "NNEDI3",
                            "Sharpness": 0.35,
                            "SLMode": 1,
                        }
                    ),
                ),
            ),
        ),
        (
            "03_qtgmc_hstroke_equalized",
            build_avs(
                archive_path,
                start_frame,
                end_frame,
                replace_qtgmc_line(
                    hstroke_equalized_filter,
                    qtgmc_call(
                        {
                            "Preset": "Slow",
                            "SourceMatch": 3,
                            "Lossless": 2,
                            "TR1": 1,
                            "TR2": 1,
                            "Rep0": 2,
                            "Rep2": 0,
                            "MatchEnhance": 0.65,
                            "MatchEdi": "NNEDI3",
                            "MatchEdi2": "NNEDI3",
                            "SMode": 0,
                            "SLMode": 0,
                            "Sharpness": 0.0,
                        }
                    ),
                ),
            ),
        ),
    ]

    outputs = []
    summary = []
    for name, avs_text in variants:
        avs_path = avs_dir / f"{safe(name)}.avs"
        mp4_path = out_dir / f"{safe(name)}.mp4"
        avs_path.write_text(avs_text, encoding="utf-8")
        encode_variant(avs_path, mp4_path)
        outputs.append(mp4_path)
        summary.append(f"{mp4_path.name}\tvariant={name}\tframes={start_frame}-{end_frame}")

    quadrant = out_dir / "quadrant_ultrasound_text.mp4"
    make_quadrant(outputs, quadrant)
    summary.append(
        "quadrant_ultrasound_text.mp4\tlayout=top-left:raw, top-right:qtgmc_baseline, "
        "bottom-left:qtgmc_text_tuned, bottom-right:qtgmc_hstroke_equalized"
    )
    (out_dir / "variants.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("Wrote:")
    for p in outputs:
        print(f"  {p}")
    print(f"  {quadrant}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
