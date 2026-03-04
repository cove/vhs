#!/usr/bin/env python3.11
#
# Processes archival MKV files by extracting chapters, deinterlacing/applying filters,
# transcribing audio to SRT/VTT, converting SRT to ASS subtitles, and encoding final MP4s
# with embedded metadata and subtitles for access/delivery copies.
#
import argparse, shutil, time, re

try:
    import whisper
    from whisper.utils import get_writer
except Exception:
    whisper = None
    get_writer = None
from common import *

ASS_NEWLINE = "\\N"
BADFRAME_MAX_SPAN_DEFAULT = 1200
BADFRAME_POST_QTGMC_MULTIPLIER = 1
BADFRAME_SOURCE_CLEARANCE = 0
RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV = "RENDER_DEBUG_EXTRACT_FRAME_NUMBERS"
# Do not bridge clean gaps between bad bursts; keep each contiguous bad run
# independent so source selection stays local to nearby clean frames.
BADFRAME_BRIDGE_ALWAYS_GAP = 0
BADFRAME_BRIDGE_SINGLETON_GAP = 0
# Auto source selection prefers forward clean frames and falls back to prior
# clean frames when a bad run reaches the chapter end.
BADFRAME_SPLIT_BURSTS_ACROSS_NEIGHBORS = False
# For single-frame repairs, skip a short lookahead/behind window before picking
# source to avoid using immediately adjacent frames that are often still unstable.
BADFRAME_SINGLE_FRAME_SOURCE_SKIP = 0
ENABLE_DESCRATCH_PLUGIN = True
GAMMA_DEFAULT = 1.0

def chapter_done(final_file):
    return final_file.exists() and final_file.stat().st_size > 100_000

def audio_mode(chapter):
    raw = chapter.get("audio")
    mode = str(raw).strip().lower() if raw is not None else "on"
    if mode in {"off", "false", "0", "no", "none"}:
        return "off"
    return "on"

def transcript_mode(archive_name, chapter_title):
    mode = get_transcript_mode_for_chapter(
        archive=str(archive_name or ""),
        chapter_title=str(chapter_title or ""),
    )
    return "on" if str(mode).strip().lower() == "on" else "off"

def title_selected(title, filters, exact=False):
    if not filters:
        return True
    text = str(title or "").strip().lower()
    for f in filters:
        needle = str(f or "").strip().lower()
        if not needle:
            continue
        if exact:
            if text == needle:
                return True
        elif needle in text:
            return True
    return False

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render delivery videos/clips from archive chapters.")
    p.add_argument(
        "--archive",
        action="append",
        default=[],
        help=(
            "Only process archive MKV stem(s) that contain this substring "
            "(case-insensitive). Repeatable."
        ),
    )
    p.add_argument(
        "--title",
        action="append",
        default=[],
        help="Only process chapter titles that contain this substring (case-insensitive). Repeatable.",
    )
    p.add_argument(
        "--title-exact",
        action="store_true",
        help="Match --title filters as exact chapter titles (case-insensitive) instead of substring match.",
    )
    p.add_argument(
        "--no-bob",
        action="store_true",
        help="Deprecated: bob output has been removed; render pipeline always renders non-bob output.",
    )
    p.add_argument(
        "--debug-extracted-frames",
        action="store_true",
        help=(
            "Burn local/global frame numbers into extracted.mkv for debugging. "
            f"Can also be enabled via {RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV}=1."
        ),
    )
    return p.parse_args(argv)

def _normalize_bad_repair_ranges(bad_source_frames=None, bad_repair_ranges=None):
    ranges = []
    if bad_repair_ranges is None:
        bad_source_frames = bad_source_frames or []
        frames = sorted({int(f) for f in bad_source_frames if int(f) >= 0})
        if not frames:
            return []
        start = prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
                continue
            ranges.append((start, prev, None))
            start = prev = f
        ranges.append((start, prev, None))
    else:
        for item in bad_repair_ranges:
            if len(item) < 2:
                continue
            try:
                a = int(item[0])
                b = int(item[1])
            except Exception:
                continue
            src = None
            if len(item) >= 3 and item[2] not in (None, ""):
                try:
                    src = int(item[2])
                except Exception:
                    src = None
            if b < a:
                a, b = b, a
            if b < 0:
                continue
            a = max(0, a)
            ranges.append((a, b, src))
        if not ranges:
            return []
    return ranges

def _resolve_badframe_repair_ranges(
    bad_source_frames=None,
    bad_repair_ranges=None,
    max_source_frame=None,
    source_clearance=0,
):
    ranges = _normalize_bad_repair_ranges(
        bad_source_frames=bad_source_frames,
        bad_repair_ranges=bad_repair_ranges,
    )
    if not ranges:
        return []

    bad_set = set()
    for a, b, _src in ranges:
        for f in range(a, b + 1):
            bad_set.add(f)

    max_allowed_src = None if max_source_frame is None else int(max_source_frame)

    clearance = max(0, int(source_clearance))

    def source_is_clear(src):
        if src in bad_set:
            return False
        if clearance <= 0:
            return True
        for f in range(int(src) - clearance, int(src) + clearance + 1):
            if f in bad_set:
                return False
        return True

    def choose_repair_source_after(b, extra_skip=0):
        src = b + 1 + max(0, int(extra_skip))
        while True:
            while src in bad_set:
                src += 1
            if max_allowed_src is not None and src > max_allowed_src:
                return None
            if source_is_clear(src):
                return src
            src += 1

    def choose_repair_source_before(a, extra_skip=0):
        src = a - 1 - max(0, int(extra_skip))
        while src >= 0:
            while src in bad_set and src >= 0:
                src -= 1
            if src < 0:
                return None
            if source_is_clear(src):
                return src
            src -= 1
        return None

    resolved_ranges = []
    for a, b, src_override in sorted(ranges, key=lambda x: (x[0], x[1])):
        src = src_override
        src_out_of_bounds = (
            max_allowed_src is not None and src is not None and src > max_allowed_src
        )
        if src is not None and src not in bad_set and not src_out_of_bounds:
            resolved_ranges.append((a, b, src))
            continue

        if src is not None and (src in bad_set or src_out_of_bounds):
            print(
                f"Badframe source override {src} is invalid for range {a}-{b}; "
                "falling back to auto neighbor source selection."
            )

        span = int(b) - int(a) + 1
        source_skip = BADFRAME_SINGLE_FRAME_SOURCE_SKIP if span == 1 else 0
        next_src = choose_repair_source_after(b, extra_skip=source_skip)

        src = next_src
        if src is None:
            prev_src = choose_repair_source_before(a, extra_skip=source_skip)
            if prev_src is None:
                print(
                    f"Unable to find clean source frame for bad range {a}-{b}; "
                    "leaving this range unrepaired."
                )
                continue
            src = prev_src
        resolved_ranges.append((int(a), int(b), int(src)))

    return _merge_badframe_repairs(resolved_ranges)

def _build_badframe_freezeframe_lines(resolved_ranges, frame_multiplier=1):
    if not resolved_ranges:
        return ""
    m = max(1, int(frame_multiplier))
    # Guardrail: never allow a source frame that is itself targeted as bad.
    bad_targets = set()
    for a, b, _src in resolved_ranges:
        bad_targets.update(range(int(a), int(b) + 1))

    fix_lines = ["c = last"]
    last_target_end = -1
    # Freeze contiguous bad-frame runs to one neighboring clean frame.
    for a, b, src in sorted(resolved_ranges, key=lambda x: (x[0], x[1])):
        ia, ib = int(a), int(b)
        if ia <= last_target_end:
            raise RuntimeError(
                f"Invalid overlapping FreezeFrame ranges: {ia}-{ib} overlaps prior target ending at {last_target_end}."
            )
        if int(src) in bad_targets:
            raise RuntimeError(
                f"Invalid FreezeFrame source {src} for bad range {a}-{b}: source is also bad."
            )
        out_a = ia * m
        out_b = ((ib + 1) * m) - 1
        out_src = int(src) * m
        fix_lines.append(f"c = c.FreezeFrame({out_a},{out_b},{out_src})")
        last_target_end = ib
    fix_lines.append("c")
    return "\n".join(fix_lines) + "\n"

def _normalize_gamma_value(raw, default=GAMMA_DEFAULT):
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    if not (value == value):
        value = float(default)
    return max(0.05, min(8.0, float(value)))

def _normalize_gamma_range_entries(gamma_ranges):
    out = []
    for idx, item in enumerate(list(gamma_ranges or [])):
        start = end = gamma = None
        if isinstance(item, dict):
            start = item.get("start_frame")
            end = item.get("end_frame")
            gamma = item.get("gamma")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, gamma = item[0], item[1], item[2]
        try:
            a = int(start)
            b = int(end)
        except Exception:
            continue
        if b <= a:
            continue
        g = _normalize_gamma_value(gamma, default=GAMMA_DEFAULT)
        out.append((a, b, g, idx))
    return out

def _resolve_gamma_segments_for_chapter(
    *,
    chapter_start_frame=0,
    chapter_end_frame=0,
    gamma_default=GAMMA_DEFAULT,
    gamma_ranges=None,
):
    chapter_start = int(chapter_start_frame)
    chapter_end = int(chapter_end_frame)
    chapter_len = max(1, chapter_end - chapter_start)

    default_gamma = _normalize_gamma_value(gamma_default, default=GAMMA_DEFAULT)
    raw_entries = _normalize_gamma_range_entries(gamma_ranges)
    entries = []
    for a, b, g, idx in raw_entries:
        ra = max(chapter_start, int(a))
        rb = min(chapter_end, int(b))
        if rb <= ra:
            continue
        entries.append((ra - chapter_start, rb - chapter_start, float(g), idx))

    boundaries = {0, chapter_len}
    for a, b, _g, _idx in entries:
        boundaries.add(int(a))
        boundaries.add(int(b))
    cuts = sorted(boundaries)

    segments = []
    for i in range(len(cuts) - 1):
        seg_a = int(cuts[i])
        seg_b = int(cuts[i + 1])
        if seg_b <= seg_a:
            continue
        gamma = float(default_gamma)
        winner_idx = -1
        for a, b, g, idx in entries:
            if a <= seg_a and seg_b <= b and idx >= winner_idx:
                gamma = float(g)
                winner_idx = idx
        if segments and segments[-1][1] == seg_a and abs(float(segments[-1][2]) - float(gamma)) < 1e-6:
            prev_a, _prev_b, prev_g = segments[-1]
            segments[-1] = (prev_a, seg_b, prev_g)
        else:
            segments.append((seg_a, seg_b, gamma))
    return segments

def _build_gamma_adjustment_lines(
    *,
    chapter_start_frame=0,
    chapter_end_frame=0,
    gamma_default=GAMMA_DEFAULT,
    gamma_ranges=None,
):
    segments = _resolve_gamma_segments_for_chapter(
        chapter_start_frame=chapter_start_frame,
        chapter_end_frame=chapter_end_frame,
        gamma_default=gamma_default,
        gamma_ranges=gamma_ranges,
    )
    chapter_len = max(1, int(chapter_end_frame) - int(chapter_start_frame))
    if not segments:
        return ""
    if all(abs(float(gamma) - 1.0) < 1e-6 for _a, _b, gamma in segments):
        return ""

    if len(segments) == 1 and int(segments[0][0]) == 0 and int(segments[0][1]) >= chapter_len:
        gamma = float(segments[0][2])
        if abs(gamma - 1.0) < 1e-6:
            return ""
        return (
            "c = last\n"
            f"c = c.SmoothLevels(16, {gamma:.4f}, 255, 16, 235, limiter=1, tvrange=true, dither=0)\n"
            "c\n"
        )

    out = [
        "c = last",
        "g_out = BlankClip(c, length=0)",
    ]
    for a, b, gamma in segments:
        ia = int(a)
        ib = int(b) - 1
        if ib < ia:
            continue
        if abs(float(gamma) - 1.0) < 1e-6:
            out.append(f"g_out = g_out ++ c.Trim({ia},{ib})")
        else:
            out.append(
                "g_out = g_out ++ "
                f"c.Trim({ia},{ib}).SmoothLevels(16, {float(gamma):.4f}, 255, 16, 235, limiter=1, tvrange=true, dither=0)"
            )
    out.append("c = g_out")
    out.append("c")
    return "\n".join(out) + "\n"

def build_badframe_prefilter_lines(bad_source_frames=None, bad_repair_ranges=None):
    resolved_ranges = _resolve_badframe_repair_ranges(
        bad_source_frames=bad_source_frames,
        bad_repair_ranges=bad_repair_ranges,
    )
    return _build_badframe_freezeframe_lines(resolved_ranges, frame_multiplier=1)

def build_badframe_postfilter_lines(bad_source_frames=None, bad_repair_ranges=None):
    resolved_ranges = _resolve_badframe_repair_ranges(
        bad_source_frames=bad_source_frames,
        bad_repair_ranges=bad_repair_ranges,
    )
    return _build_badframe_freezeframe_lines(
        resolved_ranges,
        frame_multiplier=BADFRAME_POST_QTGMC_MULTIPLIER,
    )

def cleanup_stale_subtitle_files(label, *paths):
    removed = []
    for path in paths:
        p = Path(path)
        if p.exists():
            p.unlink()
            removed.append(p.name)
    if removed:
        tag = str(label or "subtitle").strip() or "subtitle"
        print(f"Removed stale {tag} subtitle files: " + ", ".join(removed))

def cleanup_stale_dialogue_files(*paths):
    cleanup_stale_subtitle_files("dialogue", *paths)

def srt_to_ass(srt_path, ass_path, font="Calibri", fontsize=40):
    srt_path = Path(srt_path)
    ass_path = Path(ass_path)
    ass_header = f"""[Script Info]
Title: Converted from {srt_path.name}
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,1,0,2,10,10,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    content = srt_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\d+\n|\Z)", re.S)
    for idx, start, end, text in pattern.findall(content):
        text = text.strip().replace("\n", r"\N")
        start_parts = start.split(":")
        end_parts = end.split(":")
        start_ass = f"{int(start_parts[0])}:{int(start_parts[1]):02d}:{int(start_parts[2].split(',')[0]):02d}.{int(start_parts[2].split(',')[1])//10:02d}"
        end_ass = f"{int(end_parts[0])}:{int(end_parts[1]):02d}:{int(end_parts[2].split(',')[0]):02d}.{int(end_parts[2].split(',')[1])//10:02d}"
        lines.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")
    ass_path.write_text(ass_header + "\n".join(lines), encoding="utf-8")

def find_people_tsv(archive_name):
    path = METADATA_DIR / archive_name / "people.tsv"
    return path if path.exists() else None

def _merge_badframe_repairs(repairs):
    if not repairs:
        return []
    repairs = sorted(repairs, key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2]))
    merged = [repairs[0]]
    for a, b, src in repairs[1:]:
        la, lb, lsrc = merged[-1]
        if src == lsrc and a <= lb + 1:
            merged[-1] = (la, max(lb, b), lsrc)
        else:
            merged.append((a, b, src))
    return merged

def _bridge_bad_ranges(ranges):
    if not ranges:
        return []
    merged = [(int(ranges[0][0]), int(ranges[0][1]))]
    for a_raw, b_raw in ranges[1:]:
        a = int(a_raw)
        b = int(b_raw)
        la, lb = merged[-1]
        gap = a - lb - 1
        left_len = lb - la + 1
        right_len = b - a + 1
        should_bridge = False
        if gap <= BADFRAME_BRIDGE_ALWAYS_GAP:
            should_bridge = True
        elif gap <= BADFRAME_BRIDGE_SINGLETON_GAP and (left_len == 1 or right_len == 1):
            should_bridge = True
        if should_bridge:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged

def local_bad_frames_to_repairs(local_bad_frames, max_frame=None):
    frames = sorted({int(f) for f in (local_bad_frames or []) if int(f) >= 0})
    if not frames:
        return []
    contiguous = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        contiguous.append((start, prev))
        start = prev = f
    contiguous.append((start, prev))
    bridged = _bridge_bad_ranges(contiguous)
    out = [(a, b, None) for a, b in bridged]
    return _merge_badframe_repairs(out)

def chapter_global_frame_bounds(chapter):
    # Use exact rational math when raw ffmetadata ticks are available.
    return chapter_frame_bounds(chapter, fps_num=30000, fps_den=1001)

def chapter_exact_time_bounds(chapter):
    # Derive extraction time bounds from integer frame bounds to avoid
    # timestamp rounding drift (e.g., chapter start landing one frame early).
    s, e = chapter_global_frame_bounds(chapter)
    return (s * 1001.0 / 30000.0, e * 1001.0 / 30000.0)

def map_bad_ranges_to_chapter_local_frames(global_ranges, chapter):
    if not global_ranges:
        return []
    start, end = chapter_global_frame_bounds(chapter)  # [start, end)
    if end <= start:
        return []
    out = set()
    for a, b in global_ranges:
        lo = max(a, start)
        hi = min(b, end - 1)
        if hi < lo:
            continue
        for f in range(lo, hi + 1):
            out.add(f - start)
    return sorted(out)

def map_bad_repairs_to_chapter_local_ranges(global_repairs, chapter):
    if not global_repairs:
        return []
    start, end = chapter_global_frame_bounds(chapter)  # [start, end)
    if end <= start:
        return []
    out = []
    for a, b, source in global_repairs:
        lo = max(a, start)
        hi = min(b, end - 1)
        if hi < lo:
            continue
        local_source = None
        if source is not None:
            if start <= source <= end - 1:
                local_source = source - start
            else:
                print(
                    f"Badframe source override {source} is outside chapter bounds "
                    f"{start}-{end - 1}; falling back to auto source."
                )
        out.append((lo - start, hi - start, local_source))
    return _merge_badframe_repairs(out)


def _parse_sidecar_rows(tsv_path):
    rows = []
    for raw in Path(tsv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
        else:
            parts = [p.strip() for p in line.split(",")]
        rows.append(parts)
    return rows


def _frame_list_to_ranges(frames):
    ints = sorted({int(f) for f in (frames or []) if int(f) >= 0})
    if not ints:
        return []
    out = []
    start = prev = ints[0]
    for f in ints[1:]:
        if f == prev + 1:
            prev = f
            continue
        out.append((start, prev))
        start = prev = f
    out.append((start, prev))
    return out


def load_badframe_repairs(tsv_path, chapter_title=None, chapter_start_frame=None):
    rows = _parse_sidecar_rows(tsv_path)
    if not rows:
        return []

    header = [str(x).strip().lower() for x in rows[0]]
    data_rows = rows[1:]
    has_global_schema = ("frame" in header and "bad_frame" in header)
    has_local_schema = ("chapter" in header and "local_frame" in header and "bad_frame" in header)
    if not has_global_schema and not has_local_schema:
        raise ValueError(
            "Invalid frame_quality sidecar schema. "
            "Expected either: frame,bad_frame or chapter,local_frame,bad_frame."
        )

    if has_global_schema:
        idx_frame = header.index("frame")
        idx_bad = header.index("bad_frame")
        bad_frames = []
        for row in data_rows:
            if max(idx_frame, idx_bad) >= len(row):
                continue
            try:
                frame = int(row[idx_frame])
                bad = int(row[idx_bad])
            except Exception:
                continue
            if bad == 1 and frame >= 0:
                bad_frames.append(frame)
        return [(a, b, None) for a, b in _frame_list_to_ranges(bad_frames)]

    if chapter_title is None or chapter_start_frame is None:
        raise ValueError(
            "Local frame_quality sidecar requires chapter_title and chapter_start_frame."
        )

    idx_chapter = header.index("chapter")
    idx_local = header.index("local_frame")
    idx_bad = header.index("bad_frame")
    chapter_title_s = str(chapter_title).strip()
    ch_start = int(chapter_start_frame)
    bad_frames = []
    for row in data_rows:
        if max(idx_chapter, idx_local, idx_bad) >= len(row):
            continue
        try:
            ch = str(row[idx_chapter]).strip()
            local_f = int(row[idx_local])
            bad = int(row[idx_bad])
        except Exception:
            continue
        if ch != chapter_title_s:
            continue
        if bad == 1 and local_f >= 0:
            bad_frames.append(ch_start + local_f)
    return [(a, b, None) for a, b in _frame_list_to_ranges(bad_frames)]


def load_badframe_ranges(tsv_path, chapter_title=None, chapter_start_frame=None):
    repairs = load_badframe_repairs(
        tsv_path,
        chapter_title=chapter_title,
        chapter_start_frame=chapter_start_frame,
    )
    return [(int(a), int(b)) for a, b, _src in repairs]

def _parse_subtitle_ts(raw):
    text = str(raw or "").strip().replace(",", ".")
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            value = float(parts[0])
        elif len(parts) == 2:
            value = float(int(parts[0]) * 60 + float(parts[1]))
        elif len(parts) == 3:
            value = float(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
        else:
            return None
    except Exception:
        return None
    if not (value == value):
        return None
    return max(0.0, float(value))

def _to_ass_time(seconds):
    secs = max(0.0, float(seconds))
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    ss = int(s)
    cs = int(round((s - ss) * 100))
    if cs == 100:
        ss += 1
        cs = 0
    return f"{h}:{m:02d}:{ss:02d}.{cs:02d}"

def _to_srt_time(seconds):
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(ms):03d}"

def _to_vtt_time(seconds):
    return _to_srt_time(seconds).replace(",", ".")

def _load_people_tsv_entries(tsv_path):
    rows = []
    raw = Path(tsv_path).read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    for line in raw:
        text = str(line or "").strip()
        if not text:
            continue
        if text.lower().startswith("start"):
            continue
        parts = text.split("\t") if "\t" in text else text.split(",")
        if len(parts) < 3:
            continue
        start = _parse_subtitle_ts(parts[0])
        end = _parse_subtitle_ts(parts[1])
        people = ",".join(parts[2:]).strip()
        if start is None or end is None or end <= start or not people:
            continue
        rows.append((float(start), float(end), str(people)))
    return rows

def _clip_people_entries(entries, clip_start=None, clip_end=None):
    out = []
    for start_sec, end_sec, people in list(entries or []):
        start = float(start_sec)
        end = float(end_sec)
        if clip_start is not None:
            start -= float(clip_start)
            end -= float(clip_start)
        if clip_start is not None and clip_end is not None:
            duration = float(clip_end) - float(clip_start)
            if end <= 0 or start >= duration:
                continue
            if start < 0:
                start = 0.0
            if end > duration:
                end = duration
        if end <= start:
            continue
        out.append((float(start), float(end), str(people)))
    return out

def tsv_people_to_srt_vtt(tsv_path, srt_path, vtt_path, clip_start=None, clip_end=None):
    tsv_path = Path(tsv_path)
    srt_path = Path(srt_path)
    vtt_path = Path(vtt_path)
    entries = _clip_people_entries(
        _load_people_tsv_entries(tsv_path),
        clip_start=clip_start,
        clip_end=clip_end,
    )
    if not entries:
        return False

    srt_lines = []
    vtt_lines = ["WEBVTT", ""]
    for i, (start_sec, end_sec, people) in enumerate(entries, start=1):
        text = str(people).replace("|", "\n")
        srt_lines.extend(
            [
                str(i),
                f"{_to_srt_time(start_sec)} --> {_to_srt_time(end_sec)}",
                text,
                "",
            ]
        )
        vtt_lines.extend(
            [
                str(i),
                f"{_to_vtt_time(start_sec)} --> {_to_vtt_time(end_sec)}",
                text,
                "",
            ]
        )

    srt_path.write_text("\n".join(srt_lines).rstrip() + "\n", encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines).rstrip() + "\n", encoding="utf-8")
    return True

def tsv_people_to_ass(tsv_path, ass_path, font="Calibri", fontsize=36, clip_start=None, clip_end=None):
    tsv_path = Path(tsv_path)
    ass_path = Path(ass_path)
    ass_header = f"""[Script Info]
Title: People in frame ({tsv_path.name})
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: People,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,1,0,0,100,100,0,0,1,1,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    entries = _clip_people_entries(
        _load_people_tsv_entries(tsv_path),
        clip_start=clip_start,
        clip_end=clip_end,
    )
    events = []
    for start_sec, end_sec, people in entries:
        events.append(
            f"Dialogue: 0,{_to_ass_time(start_sec)},{_to_ass_time(end_sec)},People,,0,0,0,,{str(people).replace('|', ASS_NEWLINE)}"
        )

    if not events:
        return False

    ass_path.write_text(ass_header + "\n".join(events), encoding="utf-8")
    return True

def make_create_avs(
    temp_extracted: str,
    avs_filter_path: Path,
    bad_source_frames=None,
    bad_repair_ranges=None,
    chapter_start_frame=0,
    chapter_end_frame=0,
    gamma_default=GAMMA_DEFAULT,
    gamma_ranges=None,
    no_bob=False,
    source_clearance=0,
):
    # Full filter AVS: optional pre-filter FreezeFrame, then chapter filter import.
    chapter_len_frames = int(chapter_end_frame) - int(chapter_start_frame)
    if chapter_len_frames <= 0:
        chapter_len_frames = 1
    max_source_frame = chapter_len_frames - 1
    resolved_bad_repair_ranges = _resolve_badframe_repair_ranges(
        bad_source_frames=bad_source_frames or [],
        bad_repair_ranges=bad_repair_ranges,
        max_source_frame=max_source_frame,
        source_clearance=source_clearance,
    )
    prefilter_text = _build_badframe_freezeframe_lines(
        resolved_bad_repair_ranges,
        frame_multiplier=1,
    )
    # Guard against chapter filter scripts that still output bob cadence.
    # If filter output is approximately 2x chapter length, decimate to single-rate.
    cadence_guard_text = f"""c = last
expected_frames = {int(chapter_len_frames)}
if (c.FrameCount >= (expected_frames * 2 - 2) && c.FrameCount <= (expected_frames * 2 + 2)) {{
    c = c.SelectEven()
}}
"""
    gamma_adjust_text = _build_gamma_adjustment_lines(
        chapter_start_frame=chapter_start_frame,
        chapter_end_frame=chapter_end_frame,
        gamma_default=gamma_default,
        gamma_ranges=gamma_ranges,
    )
    no_bob_text = "c = last\nc\n"
    filter_import_path = Path(avs_filter_path).resolve().as_posix()
    descratch_lines = []
    if ENABLE_DESCRATCH_PLUGIN:
        for dll_name in ("Descratch64.dll", "Descratch32.dll"):
            dll_path = Path(QTGMC_DIR) / dll_name
            if dll_path.exists():
                descratch_lines.append(f'LoadPlugin("{dll_path.as_posix()}")')
                break  # only load one (64-bit preferred; skip 32-bit)
    descratch_block = ("\n".join(descratch_lines) + "\n") if descratch_lines else ""
    script_text = f'''
LoadPlugin("{QTGMC_DIR}/ffms2.dll") 
LoadPlugin("{QTGMC_DIR}/masktools2.dll") 
LoadPlugin("{QTGMC_DIR}/Rgtools.dll") 
LoadPlugin("{QTGMC_DIR}/mvtools2.dll") 
LoadPlugin("{QTGMC_DIR}/DePanEstimate.dll")
LoadPlugin("{QTGMC_DIR}/DePan.dll")
LoadPlugin("{QTGMC_DIR}/nnedi3.dll") 
LoadPlugin("{QTGMC_DIR}/yadifmod2.dll") 
LoadPlugin("{QTGMC_DIR}/fft3dfilter.dll") 
LoadPlugin("{QTGMC_DIR}/LoadDLL64.dll")
LoadPlugin("{QTGMC_DIR}/SmoothAdjust.dll")
{descratch_block}LoadDLL("{QTGMC_DIR}/libfftw3f-3.dll") 
Import("{QTGMC_DIR}/Zs_RF_Shared.avsi") 
Import("{QTGMC_DIR}/QTGMC.avsi") 
FFmpegSource2("{temp_extracted}", atrack=-1) 
chapter_start_frame = {int(chapter_start_frame)}
chapter_end_frame = {int(chapter_end_frame)}
{prefilter_text}
Import("{filter_import_path}")
{cadence_guard_text}
{gamma_adjust_text}
{no_bob_text}
'''
    import_marker = f'Import("{filter_import_path}")'
    imp_idx = script_text.find(import_marker)
    if imp_idx >= 0:
        post_import = script_text[imp_idx + len(import_marker):]
        if "FreezeFrame(" in post_import:
            raise RuntimeError(
                "Invalid AVS generation: FreezeFrame lines found after filter import."
            )
    return script_text

def make_freeze_only_avs(
    temp_extracted: str,
    bad_source_frames=None,
    bad_repair_ranges=None,
    chapter_start_frame=0,
    chapter_end_frame=0,
    source_clearance=0,
):
    chapter_len_frames = int(chapter_end_frame) - int(chapter_start_frame)
    if chapter_len_frames <= 0:
        chapter_len_frames = 1
    max_source_frame = chapter_len_frames - 1
    resolved_bad_repair_ranges = _resolve_badframe_repair_ranges(
        bad_source_frames=bad_source_frames or [],
        bad_repair_ranges=bad_repair_ranges,
        max_source_frame=max_source_frame,
        source_clearance=source_clearance,
    )
    freeze_text = _build_badframe_freezeframe_lines(
        resolved_bad_repair_ranges,
        frame_multiplier=1,
    )
    if not freeze_text:
        freeze_text = "c = last\nc\n"
    return f'''
LoadPlugin("{QTGMC_DIR}/ffms2.dll")
FFmpegSource2("{temp_extracted}", atrack=-1)
chapter_start_frame = {int(chapter_start_frame)}
chapter_end_frame = {int(chapter_end_frame)}
{freeze_text}
'''

def make_gamma_only_avs(
    temp_extracted: str,
    chapter_start_frame=0,
    chapter_end_frame=0,
    gamma_default=GAMMA_DEFAULT,
    gamma_ranges=None,
):
    gamma_adjust_text = _build_gamma_adjustment_lines(
        chapter_start_frame=chapter_start_frame,
        chapter_end_frame=chapter_end_frame,
        gamma_default=gamma_default,
        gamma_ranges=gamma_ranges,
    )
    if not gamma_adjust_text:
        gamma_adjust_text = "c = last\nc\n"
    return f'''
LoadPlugin("{QTGMC_DIR}/ffms2.dll")
LoadPlugin("{QTGMC_DIR}/SmoothAdjust.dll")
FFmpegSource2("{temp_extracted}", atrack=-1)
chapter_start_frame = {int(chapter_start_frame)}
chapter_end_frame = {int(chapter_end_frame)}
{gamma_adjust_text}
'''

def make_extract_audio(temp_extracted, temp_transcript):
    return [FFMPEG_BIN,
        "-nostdin", "-v", "error",
        "-i", str(temp_extracted),
        "-vn",
        "-af", "highpass=f=120,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=13,aresample=16000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-y", str(temp_transcript)]

def debug_extracted_frames_enabled(args):
    env_raw = str(os.environ.get(RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV, "")).strip().lower()
    env_on = env_raw in {"1", "true", "yes", "on"}
    return bool(getattr(args, "debug_extracted_frames", False) or env_on)

def probe_video_frame_count(path):
    p = Path(path)
    if not p.exists():
        return 0
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(p),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return 0
    for raw in str(out or "").splitlines():
        token = str(raw or "").strip()
        if not token or token.upper() == "N/A":
            continue
        try:
            v = int(token)
        except Exception:
            continue
        if v > 0:
            return int(v)
    return 0

def assert_expected_frame_count(path, expected_frames, context_label):
    exp = max(0, int(expected_frames))
    if exp <= 0:
        return
    actual = probe_video_frame_count(path)
    if actual <= 0:
        print(f"WARNING: Unable to probe frame count for {context_label}: {path}")
        return
    # Allow tiny tolerance for muxer boundary effects.
    if abs(int(actual) - int(exp)) <= 1:
        return
    hint = ""
    if abs(int(actual) - int(exp * 2)) <= 2:
        hint = " (looks like accidental bob/double-rate output)"
    raise RuntimeError(
        f"Frame-count drift detected for {context_label}: expected {exp}, got {actual}.{hint}"
    )

def _subtitle_io(subtitle_tracks):
    input_args = []
    output_args = []
    if not subtitle_tracks:
        return input_args, output_args

    for sub in subtitle_tracks:
        input_args += ["-i", str(sub["path"])]

    for i in range(len(subtitle_tracks)):
        output_args += ["-map", f"{i + 1}:s:0"]

    output_args += ["-c:s", "mov_text"]

    for i, sub in enumerate(subtitle_tracks):
        output_args += [f"-metadata:s:s:{i}", "language=eng"]
        title = sub.get("title")
        if title:
            output_args += [f"-metadata:s:s:{i}", f"title={title}"]
        if sub.get("forced"):
            output_args += [f"-disposition:s:{i}", "forced"]
    return input_args, output_args

def build_filmed_comment(author, creation_time, location, archive_tape_title, start_hms, end_hms):
    author_text = "" if author is None else str(author).strip()
    if not author_text or author_text.lower() in {"none", "null"}:
        head = f"Filmed on {creation_time} at {location}"
    else:
        head = f"Filmed by {author_text} on {creation_time} at {location}"
    return f"{head}, original tape {archive_tape_title} @ {start_hms}-{end_hms} "

def make_encode_final_x265(temp_qtgmc, subtitle_tracks, final_file, author, title, archive_tape_title, start_hms, end_hms, creation_time, location, include_audio=True):
    subtitle_tracks = subtitle_tracks or []
    sub_inputs, sub_outputs = _subtitle_io(subtitle_tracks)
    comment = build_filmed_comment(author, creation_time, location, archive_tape_title, start_hms, end_hms)
    cmd = [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_qtgmc),
        *sub_inputs,
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-pix_fmt", "yuv420p",
        "-fps_mode:v:0", "passthrough",
        "-c:v", "libx265", "-crf", "20", "-preset", "slow",
        "-profile:v", "main", "-level", "4.0",
        "-x265-params", "log-level=0",
        "-tag:v", "hvc1", "-brand", "mp42",
        "-map", "0:v:0",
    ]
    if include_audio:
        cmd += [
            "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "1",
            "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-map", "0:a:0?",
        ]
    else:
        cmd += ["-an"]
    cmd += [
        *sub_outputs,
        "-metadata", f"title={title}",
        "-metadata", f"comment={comment}",
        "-metadata", f"creation_time={creation_time}",
        "-metadata", f"location={location}",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart+use_metadata_tags",
        "-y", str(final_file),
    ]
    if include_audio:
        cmd += ["-metadata:s:a:0", "language=eng"]
    return cmd

def make_encode_final_x264(temp_qtgmc, subtitle_tracks, final_file, author, title, archive_tape_title, start_hms, end_hms, creation_time, location, include_audio=True):
    subtitle_tracks = subtitle_tracks or []
    sub_inputs, sub_outputs = _subtitle_io(subtitle_tracks)
    comment = build_filmed_comment(author, creation_time, location, archive_tape_title, start_hms, end_hms)
    cmd = [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_qtgmc),
        *sub_inputs,
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-pix_fmt", "yuv420p",
        "-fps_mode:v:0", "passthrough",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-level", "4.0", "-tune", "grain",
        "-map", "0:v:0",
    ]
    if include_audio:
        cmd += [
            "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "1",
            "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-map", "0:a:0?",
        ]
    else:
        cmd += ["-an"]
    cmd += [
        *sub_outputs,
        "-metadata", f"title={title}",
        "-metadata", f"comment={comment}",
        "-metadata", f"creation_time={creation_time}",
        "-metadata", f"location={location}",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart+use_metadata_tags",
        "-y", str(final_file),
    ]
    if include_audio:
        cmd += ["-metadata:s:a:0", "language=eng"]
    return cmd

def make_render_avs_ffv1(temp_avs, temp_audio_src, temp_out):
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_avs),
        "-i", str(temp_audio_src),
        "-pix_fmt", "yuv422p",
        "-fps_mode:v:0", "passthrough",
        "-map", "0:v:0", "-c:v", "ffv1",
        "-level", "3", "-coder", "1", "-context", "1",
        "-map", "1:a:0?", "-c:a", "copy",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-y", str(temp_out)]

def make_deinterlace(temp_avs, temp_extracted, temp_qtgmc):
    return make_render_avs_ffv1(temp_avs, temp_extracted, temp_qtgmc)

def make_deinterlace_ffmpeg_fallback(temp_extracted, temp_qtgmc, no_bob=False):
    # Cross-platform fallback when AviSynth/QTGMC is unavailable.
    # Bob output has been removed; always emit one frame per input frame.
    bwdif_mode = "send_frame"
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_extracted),
        "-vf", f"bwdif=mode={bwdif_mode}:parity=auto:deint=interlaced",
        "-pix_fmt", "yuv422p",
        "-fps_mode:v:0", "passthrough",
        "-map", "0:v:0", "-c:v", "ffv1",
        "-level", "3", "-coder", "1", "-context", "1",
        "-map", "0:a:0?", "-c:a", "copy",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-y", str(temp_qtgmc)]

def transcribe_audio(model, temp_transcript, final_srt, final_vtt, final_dir):
    if get_writer is None:
        raise RuntimeError("Whisper is unavailable. Install whisper to generate transcripts.")
    prompt_text = (
      ""
    )
    srt_writer = get_writer("srt", final_dir)
    vtt_writer = get_writer("vtt", final_dir)
    result = model.transcribe(str(temp_transcript), word_timestamps=True, language="en", fp16=False, prompt=prompt_text)
    srt_writer(result, str(final_srt))
    vtt_writer(result, str(final_vtt))

def _run_with_args(args):
    model = None
    rebuild_selected = bool(args.title)
    debug_extracted_frames = debug_extracted_frames_enabled(args)
    if debug_extracted_frames:
        print(
            "Debug extracted-frame overlay enabled: "
            f"{RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV}=1 or --debug-extracted-frames"
        )

    archive_filters = [str(x or "").strip().lower() for x in (args.archive or []) if str(x or "").strip()]
    for src in ARCHIVE_DIR.glob("*.mkv"):
        if archive_filters:
            stem_text = src.stem.strip().lower()
            if not any(f in stem_text for f in archive_filters):
                continue
        archive_name = src.stem
        chapters_file = METADATA_DIR / archive_name / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name}: no metadata found {chapters_file}")
            continue
        _settings_path, _render_settings = load_render_settings(archive_name, create=True)

        ffm, chapters = parse_chapters(chapters_file)
        if not chapters:
            print(f"No chapters for {src.name}")
            continue

        for ch in chapters:
            ch["duration"] = float(ch.get("end", 0)) - float(ch.get("start", 0))
        chapters.sort(key=lambda x: x["duration"])
        chapters = [ch for ch in chapters if title_selected(ch.get("title"), args.title, exact=bool(args.title_exact))]
        if args.title and not chapters:
            print(f"Skipping {src.name}: no chapters matched --title filter(s).")
            continue

        cur_count = 1
        total_chapters = len(chapters)

        for i, ch in enumerate(chapters):
            title = ch.get("title")
            start_sec = ch["start"]
            end_sec = ch["end"]
            extract_start_sec, extract_end_sec = chapter_exact_time_bounds(ch)
            chapter_start_frame, chapter_end_frame = chapter_global_frame_bounds(ch)

            final_dir = VIDEOS_DIR if ch["duration"] >= 200 else CLIPS_DIR
            final_file = final_dir / f"{safe(title)}.mp4"
            final_srt = final_dir / f"{safe(title)}.srt"
            final_vtt = final_dir / f"{safe(title)}.vtt"
            final_ass = final_dir / f"{safe(title)}.ass"
            people_srt = final_dir / f"{safe(title)}.people.srt"
            people_vtt = final_dir / f"{safe(title)}.people.vtt"
            people_ass = final_dir / f"{safe(title)}.people.ass"
            people_tsv = find_people_tsv(archive_name)
            include_audio = audio_mode(ch) == "on"
            transcribe_dialogue = include_audio and transcript_mode(archive_name, title) == "on"

            if not transcribe_dialogue:
                cleanup_stale_dialogue_files(final_srt, final_vtt, final_ass)

            if chapter_done(final_file) and not rebuild_selected:
                print(f"Skipping existing chapter: {title}")
                cur_count += 1
                continue
            if chapter_done(final_file) and rebuild_selected:
                print(f"Rebuilding matched chapter: {title}")

            # inline temp path creation
            temp_dir = final_dir / f"{safe(title)}_temp"
            temp_dir.mkdir(exist_ok=True)
            extracted = temp_dir / "extracted.mkv"
            repaired_extracted = temp_dir / "repaired_extracted.mkv"
            qtgmc = temp_dir / "qtgmc.mkv"
            audio = temp_dir / "audio.wav"
            avs = temp_dir / "script.avs"
            freeze_avs = temp_dir / "freeze.avs"
            filter_script = METADATA_DIR / archive_name / "filter.avs"
            chapter_filter_script = METADATA_DIR / archive_name / f"{title}.avs"
            if chapter_filter_script.exists():
                filter_script = chapter_filter_script

            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            date = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            progress = f"({cur_count} of {total_chapters} chapters) [{date}]"
            print(f"Processing: {src.name} {progress}")
            print(
                f"Chapter bounds (full): {title} | "
                f"{extract_start_sec:.3f}s-{extract_end_sec:.3f}s "
                f"(frames {chapter_start_frame}-{max(chapter_start_frame, chapter_end_frame - 1)})"
            )
            chapter_len = max(0, int(chapter_end_frame) - int(chapter_start_frame))

            try:
                print(f"Extracting chapter...")
                run(
                    make_extract_chapter(
                        src,
                        extract_start_sec,
                        extract_end_sec,
                        extracted,
                        start_frame=chapter_start_frame,
                        end_frame=chapter_end_frame,
                        debug_frame_numbers=debug_extracted_frames,
                    )
                )
                assert_expected_frame_count(
                    extracted,
                    chapter_len,
                    f"extracted chapter '{title}'",
                )

                print(f"Applying video filters...")
                if sys.platform == "win32":
                    if filter_script.exists():
                        global_bad = get_bad_frames_for_chapter(archive_name, str(title))
                        manual_source_frames_global = [
                            f for f in global_bad if chapter_start_frame <= int(f) < chapter_end_frame
                        ]
                        manual_source_frames = [
                            int(f) - int(chapter_start_frame) for f in manual_source_frames_global
                        ]
                        manual_repairs = local_bad_frames_to_repairs(manual_source_frames)
                        marked_count = len(set(int(f) for f in manual_source_frames))
                        repaired_count = sum((int(b) - int(a) + 1) for a, b, _src in manual_repairs)
                        freeze_input = extracted
                        if manual_source_frames_global:
                            print(
                                f"Render settings bad frame(s): {len(manual_source_frames)} -> "
                                + ",".join(str(f) for f in manual_source_frames[:12])
                                + ("..." if len(manual_source_frames) > 12 else "")
                            )
                            if repaired_count > marked_count:
                                print(
                                    "Expanded freeze target coverage by "
                                    f"{repaired_count - marked_count} frame(s) via gap bridging "
                                    f"(always<={BADFRAME_BRIDGE_ALWAYS_GAP}, "
                                    f"singleton<={BADFRAME_BRIDGE_SINGLETON_GAP})."
                                )
                            freeze_script = make_freeze_only_avs(
                                extracted,
                                bad_source_frames=manual_source_frames,
                                bad_repair_ranges=manual_repairs,
                                chapter_start_frame=chapter_start_frame,
                                chapter_end_frame=chapter_end_frame,
                                source_clearance=BADFRAME_SOURCE_CLEARANCE,
                            )
                            freeze_lines = freeze_script.count("FreezeFrame(")
                            print(
                                "AVS freeze stage: "
                                f"freeze_lines={freeze_lines}, "
                                f"source_clearance={BADFRAME_SOURCE_CLEARANCE}"
                            )
                            freeze_avs.write_text(freeze_script, encoding="ascii")
                            run(make_render_avs_ffv1(freeze_avs, extracted, repaired_extracted))
                            assert_expected_frame_count(
                                repaired_extracted,
                                chapter_len,
                                f"repaired extracted chapter '{title}'",
                            )
                            freeze_input = repaired_extracted
                        else:
                            print("No bad frames listed in render_settings; no freeze-frame repairs applied.")

                        gamma_profile = get_gamma_profile_for_chapter(
                            archive=str(archive_name or ""),
                            chapter_title=str(title or ""),
                            ch_start=chapter_start_frame,
                            ch_end=chapter_end_frame,
                        )
                        gamma_default = float(gamma_profile.get("default_gamma", GAMMA_DEFAULT))
                        gamma_ranges = list(gamma_profile.get("ranges", []))

                        script = make_create_avs(
                            freeze_input,
                            filter_script,
                            bad_source_frames=[],
                            bad_repair_ranges=[],
                            chapter_start_frame=chapter_start_frame,
                            chapter_end_frame=chapter_end_frame,
                            gamma_default=gamma_default,
                            gamma_ranges=gamma_ranges,
                            no_bob=args.no_bob,
                            source_clearance=0,
                        )
                        freeze_count = script.count("FreezeFrame(")
                        filter_has_qtgmc = False
                        try:
                            filter_has_qtgmc = "QTGMC(" in filter_script.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        except Exception:
                            pass
                        print(
                            "AVS pipeline: "
                                f"freeze_lines={freeze_count}, "
                                f"filter_script={filter_script.name}, "
                                f"filter_has_qtgmc={filter_has_qtgmc}, "
                                f"filter_input={Path(freeze_input).name}"
                            )
                        avs.write_text(script, encoding="ascii")
                        run(make_deinterlace(avs, freeze_input, qtgmc))
                        assert_expected_frame_count(
                            qtgmc,
                            chapter_len,
                            f"qtgmc chapter '{title}'",
                        )
                    else:
                        print("Skipping since there's no filter script for this archive...")
                        shutil.copy(extracted, qtgmc)
                elif os.environ.get("TEST_ENV"):
                    print("Skipping deinterlacing for test run...")
                    shutil.copy(extracted, qtgmc)
                else:
                    print(
                        "AviSynth/QTGMC is Windows-only. "
                        f"Using FFmpeg bwdif fallback on {sys.platform}."
                    )
                    if filter_script.exists():
                        print(
                            f"Skipping AviSynth filter script on this platform: {filter_script.name}"
                        )
                    run(make_deinterlace_ffmpeg_fallback(extracted, qtgmc, no_bob=args.no_bob))
                    assert_expected_frame_count(
                        qtgmc,
                        chapter_len,
                        f"fallback chapter '{title}'",
                    )

                subtitle_tracks = []

                if transcribe_dialogue:
                    print(f"Transcribing audio...")
                    if whisper is None:
                        raise RuntimeError(
                            "Whisper is unavailable. Install whisper, or set "
                            "archive_settings.transcript=off (or chapter override) in render_settings.json."
                        )
                    if model is None:
                        model = whisper.load_model("turbo", download_root=WHISPER_MODEL_DIR)
                    run(make_extract_audio(extracted, audio))
                    transcribe_audio(model, audio, final_srt, final_vtt, final_dir)
                    srt_to_ass(final_srt, final_ass)
                    subtitle_tracks.append({"path": final_ass, "title": "Dialogue", "forced": False})
                elif include_audio:
                    print("Skipping dialogue transcription (render_settings transcript=off).")
                else:
                    print("Skipping audio and transcription (AUDIO=off).")

                if people_tsv:
                    wrote_people_text = tsv_people_to_srt_vtt(
                        people_tsv,
                        people_srt,
                        people_vtt,
                        clip_start=extract_start_sec,
                        clip_end=extract_end_sec,
                    )
                    wrote_people_ass = tsv_people_to_ass(
                        people_tsv,
                        people_ass,
                        clip_start=extract_start_sec,
                        clip_end=extract_end_sec,
                    )
                    if wrote_people_text and wrote_people_ass:
                        subtitle_tracks.append({"path": people_ass, "title": "People", "forced": False})
                    else:
                        cleanup_stale_subtitle_files("people", people_srt, people_vtt, people_ass)
                        print(f"People TSV had no entries: {people_tsv}")
                else:
                    cleanup_stale_subtitle_files("people", people_srt, people_vtt, people_ass)

                print(f"Final encoding...")
                author = ch.get("author", ffm.get("author"))
                archive_tape_title = ffm.get("title")
                start_hms = format_hms(start_sec)
                end_hms = format_hms(end_sec)
                ctime = ch.get("creation_time")
                location = ch.get("location")

                cmd = make_encode_final_x264(
                    qtgmc, subtitle_tracks, final_file, author, title, archive_tape_title,
                    start_hms, end_hms, ctime, location, include_audio=include_audio
                )
                run(cmd)

            finally:
                os.chdir(original_cwd)
                keep_temp = str(os.getenv("RENDER_KEEP_TEMP", "0")).strip().lower() in {
                    "1", "true", "yes", "on"
                }
                if keep_temp:
                    print(f"Keeping temp dir for inspection: {temp_dir}")
                else:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            cur_count += 1

        print(f"All done")


def run_make_videos(
    *,
    title_filters=None,
    no_bob=False,
):
    args = argparse.Namespace(
        archive=[],
        title=list(title_filters or []),
        title_exact=False,
        no_bob=bool(no_bob),
    )
    _run_with_args(args)


def main(argv=None):
    args = parse_args(argv)
    _run_with_args(args)

if __name__ == "__main__":
    main()
