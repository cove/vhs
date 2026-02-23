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
BADFRAME_PAD_BEFORE_DEFAULT = 2
BADFRAME_PAD_AFTER_DEFAULT = 0
BADFRAME_POST_QTGMC_MULTIPLIER = 2

def auto_badframe_pad(span):
    # Minimize repaired-frame count while still protecting QTGMC temporal context.
    # Single-frame glitches usually do not need extra pre-pad.
    if span <= 1:
        return 0, 0
    if span <= 3:
        return 1, 0
    return BADFRAME_PAD_BEFORE_DEFAULT, BADFRAME_PAD_AFTER_DEFAULT

def chapter_done(final_file):
    return final_file.exists() and final_file.stat().st_size > 100_000

def audio_mode(chapter):
    raw = chapter.get("audio")
    mode = str(raw).strip().lower() if raw is not None else "on"
    if mode in {"off", "false", "0", "no", "none"}:
        return "off"
    return "on"

def transcript_mode(chapter):
    raw = chapter.get("transcript")
    mode = str(raw).strip().lower() if raw is not None else "on"
    if mode in {"off", "false", "0", "no", "skip", "none"}:
        return "off"
    if mode in {"on", "true", "1", "yes", "force", "auto"}:
        return "on"
    return "on"

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
        help="Deprecated: bob output has been removed; step_6 always renders non-bob output.",
    )
    p.add_argument(
        "--frame-quality-tsv",
        default="",
        help=(
            "Deprecated and ignored. step_6 now reads per-chapter BAD_FRAMES from "
            "metadata/<archive>/chapters.ffmetadata."
        ),
    )
    p.add_argument(
        "--frame-quality-archive",
        default="",
        help=(
            "Deprecated and ignored."
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

    def choose_repair_source_at_or_after(floor_frame, b):
        src = max(int(floor_frame), b + 1)
        while src in bad_set:
            src += 1
        if max_allowed_src is not None and src > max_allowed_src:
            return None
        return src

    def choose_repair_source_before(a):
        src = a - 1
        if max_allowed_src is not None:
            src = min(src, max_allowed_src)
        while src >= 0 and src in bad_set:
            src -= 1
        if src < 0:
            return None
        return src

    resolved_ranges = []
    # Keep replacement sources strictly forward in source time:
    # - never use a frame that has already played in output ([0..b])
    # - never move backward once a later source frame has been used
    last_used_src = -1
    for a, b, src_override in sorted(ranges, key=lambda x: (x[0], x[1])):
        min_src = max(last_used_src, b + 1)
        src = src_override
        src_out_of_bounds = (
            max_allowed_src is not None and src is not None and src > max_allowed_src
        )
        if src is None or src < min_src or src in bad_set or src_out_of_bounds:
            if src is not None and (src < min_src or src in bad_set or src_out_of_bounds):
                print(
                    f"Badframe source override {src} is invalid for range {a}-{b}; "
                    f"using forward source >= {min_src}."
                )
            src = choose_repair_source_at_or_after(min_src, b)
            if src is None:
                # Chapter/timeline-bound fallback: if no forward clean frame exists,
                # use the nearest previous clean frame rather than risking a bad-frame clamp.
                src = choose_repair_source_before(a)
                if src is None:
                    print(
                        f"Unable to find any clean source frame for bad range {a}-{b}; "
                        "leaving this range unrepaired."
                    )
                    continue
                print(
                    f"No forward clean source within bounds for bad range {a}-{b}; "
                    f"falling back to previous clean frame {src}."
                )
        last_used_src = max(last_used_src, src)
        resolved_ranges.append((a, b, src))
    return resolved_ranges

def _build_badframe_freezeframe_lines(resolved_ranges, frame_multiplier=1):
    if not resolved_ranges:
        return ""
    m = max(1, int(frame_multiplier))

    fix_lines = ["c = last"]
    # Freeze contiguous bad-frame runs to one neighboring clean frame.
    for a, b, src in sorted(resolved_ranges, key=lambda x: (x[0], x[1]), reverse=True):
        out_a = a * m
        out_b = ((b + 1) * m) - 1
        out_src = src * m
        fix_lines.append(f"c = c.FreezeFrame({out_a},{out_b},{out_src})")
    fix_lines.append("c")
    return "\n".join(fix_lines) + "\n"

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

def cleanup_stale_dialogue_files(*paths):
    removed = []
    for path in paths:
        p = Path(path)
        if p.exists():
            p.unlink()
            removed.append(p.name)
    if removed:
        print("Removed stale dialogue transcript files: " + ", ".join(removed))

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

def find_frame_quality_tsv(archive_name):
    path = METADATA_DIR / archive_name / "frame_quality.tsv"
    return path if path.exists() else None

def resolve_quality_sidecar_tsv(archive_name, override_path=None, override_archive=""):
    if override_path and (not override_archive or archive_name == override_archive):
        return Path(override_path)
    return find_frame_quality_tsv(archive_name)

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

def _read_frame_quality_header(tsv_path):
    if not tsv_path:
        return [], None, []
    rows = Path(tsv_path).read_text(encoding="utf-8-sig").splitlines()
    first_data = None
    for line in rows:
        s = line.strip()
        if s and not s.startswith("#"):
            first_data = s
            break
    if not first_data:
        return rows, None, []
    cols = [c.strip().lower() for c in first_data.split("\t")]
    return rows, first_data, cols

def frame_quality_schema(tsv_path):
    rows, first_data, cols = _read_frame_quality_header(tsv_path)
    if not first_data:
        return None
    if "frame" in cols and "bad_frame" in cols:
        return "global"
    if "local_frame" in cols and "bad_frame" in cols:
        return "local"
    raise ValueError(
        f"Expected frame_quality.tsv format in {tsv_path} "
        "(required columns: frame,bad_frame or local_frame,bad_frame)."
    )

def load_badframe_repairs(tsv_path, chapter_title=None, chapter_start_frame=None):
    if not tsv_path:
        return []
    rows, first_data, cols = _read_frame_quality_header(tsv_path)
    if not first_data:
        return []
    schema = frame_quality_schema(tsv_path)

    idx_bad = cols.index("bad_frame")
    idx_frame = cols.index("frame") if "frame" in cols else None
    idx_local_frame = cols.index("local_frame") if "local_frame" in cols else None
    idx_chapter = cols.index("chapter") if "chapter" in cols else None

    local_title = str(chapter_title or "").strip().lower()
    local_start = int(chapter_start_frame or 0)
    if schema == "local" and chapter_start_frame is None:
        raise ValueError(
            f"Local frame_quality sidecar requires chapter_start_frame: {tsv_path}"
        )
    if schema == "local" and idx_chapter is not None and not local_title:
        raise ValueError(
            f"Local frame_quality sidecar with chapter column requires chapter_title: {tsv_path}"
        )

    bad_frames = []
    for line in rows:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("\t")]
        low = [p.lower() for p in parts]
        if low and low[0] == "frame":
            continue
        idxs = [idx_bad]
        if idx_frame is not None:
            idxs.append(idx_frame)
        if idx_local_frame is not None:
            idxs.append(idx_local_frame)
        if idx_chapter is not None:
            idxs.append(idx_chapter)
        if max(idxs) >= len(parts):
            continue
        try:
            is_bad = int(parts[idx_bad]) == 1
        except Exception:
            continue
        if not is_bad:
            continue
        if schema == "global":
            try:
                frame = int(parts[idx_frame])
            except Exception:
                continue
        else:
            if idx_chapter is not None and local_title:
                row_chapter = str(parts[idx_chapter]).strip().lower()
                if row_chapter != local_title:
                    continue
            try:
                frame = local_start + int(parts[idx_local_frame])
            except Exception:
                continue
        bad_frames.append(frame)
    bad_frames = sorted(set(bad_frames))
    if not bad_frames:
        return []
    out_exact = []
    start = prev = bad_frames[0]
    for f in bad_frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            out_exact.append((start, prev, None))
            start = prev = f
    out_exact.append((start, prev, None))
    return _merge_badframe_repairs(out_exact)

def load_badframe_ranges(tsv_path, chapter_title=None, chapter_start_frame=None):
    return [
        (a, b)
        for (a, b, _src) in load_badframe_repairs(
            tsv_path,
            chapter_title=chapter_title,
            chapter_start_frame=chapter_start_frame,
        )
    ]

def local_bad_frames_to_repairs(local_bad_frames):
    frames = sorted({int(f) for f in (local_bad_frames or []) if int(f) >= 0})
    if not frames:
        return []
    out = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        out.append((start, prev, None))
        start = prev = f
    out.append((start, prev, None))
    return _merge_badframe_repairs(out)

def chapter_global_frame_bounds(chapter):
    # Chapters are parsed to seconds in common.parse_chapters; convert back to
    # source frame indices using archive cadence (30000/1001).
    s = int(round(float(chapter.get("start", 0.0)) * 30000.0 / 1001.0))
    e = int(round(float(chapter.get("end", 0.0)) * 30000.0 / 1001.0))
    if e < s:
        e = s
    return s, e

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
    def parse_ts(ts):
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")
        if len(parts) == 1:
            h = 0
            m = 0
            s = float(parts[0])
        elif len(parts) == 2:
            h = 0
            m = int(parts[0])
            s = float(parts[1])
        else:
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
        return h * 3600 + m * 60 + s

    def to_ass_time(seconds):
        if seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        ss = int(s)
        cs = int(round((s - ss) * 100))
        if cs == 100:
            ss += 1
            cs = 0
        return f"{h}:{m:02d}:{ss:02d}.{cs:02d}"

    lines = []
    # utf-8-sig strips BOM that can appear on the header line
    raw = tsv_path.read_text(encoding="utf-8-sig").splitlines()
    for line in raw:
        if not line.strip():
            continue
        if line.lower().startswith("start"):
            continue
        if "\t" in line:
            parts = line.split("\t")
        else:
            parts = line.split(",")
        if len(parts) < 3:
            continue
        start = parts[0].strip()
        end = parts[1].strip()
        people = ",".join(parts[2:]).strip()
        if not start or not end or not people:
            continue
        lines.append((start, end, people))

    if not lines:
        return False

    events = []
    for start, end, people in lines:
        start_sec = parse_ts(start)
        end_sec = parse_ts(end)
        if clip_start is not None:
            start_sec -= float(clip_start)
            end_sec -= float(clip_start)

        if clip_start is not None and clip_end is not None:
            duration = float(clip_end) - float(clip_start)
            if end_sec <= 0 or start_sec >= duration:
                continue
            if start_sec < 0:
                start_sec = 0.0
            if end_sec > duration:
                end_sec = duration

        events.append(
            f"Dialogue: 0,{to_ass_time(start_sec)},{to_ass_time(end_sec)},People,,0,0,0,,{people.replace('|', ASS_NEWLINE)}"
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
    no_bob=False,
):
    chapter_len_frames = int(chapter_end_frame) - int(chapter_start_frame)
    max_source_frame = chapter_len_frames - 1
    resolved_bad_repair_ranges = _resolve_badframe_repair_ranges(
        bad_source_frames=bad_source_frames or [],
        bad_repair_ranges=bad_repair_ranges,
        max_source_frame=max_source_frame,
    )
    prefilter_text = _build_badframe_freezeframe_lines(
        resolved_bad_repair_ranges,
        frame_multiplier=1,
    )
    postfilter_text = _build_badframe_freezeframe_lines(
        resolved_bad_repair_ranges,
        frame_multiplier=BADFRAME_POST_QTGMC_MULTIPLIER,
    )
    # Bob output has been removed; always keep one output frame per source frame.
    no_bob_text = "c = last\nc = c.SelectEven()\nc\n"
    filter_import_path = Path(avs_filter_path).resolve().as_posix()
    return f'''
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
LoadDLL("{QTGMC_DIR}/libfftw3f-3.dll") 
Import("{QTGMC_DIR}/Zs_RF_Shared.avsi") 
Import("{QTGMC_DIR}/QTGMC.avsi") 
FFmpegSource2("{temp_extracted}", atrack=-1) 
chapter_start_frame = {int(chapter_start_frame)}
chapter_end_frame = {int(chapter_end_frame)}
{prefilter_text}
Import("{filter_import_path}")
{postfilter_text}
{no_bob_text}
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

def make_extract_chapter(src, start, end, dest):
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(src),
        "-ss", f"{start}", "-to", f"{end}",
        "-force_key_frames", "0",
        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-y", str(dest)]

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

def make_deinterlace(temp_avs, temp_extracted, temp_qtgmc):
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_avs),
        "-i", str(temp_extracted),
        "-pix_fmt", "yuv422p",
        "-map", "0:v:0", "-c:v", "ffv1",
        "-level", "3", "-coder", "1", "-context", "1",
        "-map", "0:a", "-c:a", "copy",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-y", str(temp_qtgmc)]

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
    quality_arg = str(getattr(args, "frame_quality_tsv", "") or "").strip()
    if quality_arg:
        print(
            "WARNING: --frame-quality-tsv/--frame-quality-archive are deprecated and ignored. "
            "Using BAD_FRAMES in chapters.ffmetadata."
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
            people_ass = final_dir / f"{safe(title)}.people.ass"
            people_tsv = find_people_tsv(archive_name)
            include_audio = audio_mode(ch) == "on"
            transcribe_dialogue = include_audio and transcript_mode(ch) == "on"

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
            qtgmc = temp_dir / "qtgmc.mkv"
            audio = temp_dir / "audio.wav"
            avs = temp_dir / "script.avs"
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

            try:
                print(f"Extracting chapter...")
                run(make_extract_chapter(src, extract_start_sec, extract_end_sec, extracted))

                print(f"Applying video filters...")
                if sys.platform == "win32":
                    if filter_script.exists():
                        chapter_len = max(0, int(chapter_end_frame) - int(chapter_start_frame))
                        if "bad_frames" not in ch:
                            print(
                                f"WARNING: chapter '{title}' has no BAD_FRAMES metadata; "
                                "rendering without freeze-frame repair for this chapter."
                            )
                        local_bad = parse_bad_frames_csv(ch.get("bad_frames", ""))
                        manual_source_frames = [
                            f for f in local_bad if 0 <= int(f) < max(1, chapter_len)
                        ]
                        manual_repairs = local_bad_frames_to_repairs(manual_source_frames)
                        if manual_source_frames:
                            print(
                                f"Chapter metadata bad frame(s): {len(manual_source_frames)} -> "
                                + ",".join(str(f) for f in manual_source_frames[:12])
                                + ("..." if len(manual_source_frames) > 12 else "")
                            )
                        else:
                            print("No chapter bad frames listed; no freeze-frame repairs applied.")

                        script = make_create_avs(
                            extracted,
                            filter_script,
                            bad_source_frames=manual_source_frames,
                            bad_repair_ranges=manual_repairs,
                            chapter_start_frame=chapter_start_frame,
                            chapter_end_frame=chapter_end_frame,
                            no_bob=args.no_bob,
                        )
                        avs.write_text(script, encoding="ascii")
                        run(make_deinterlace(avs, extracted, qtgmc))
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

                subtitle_tracks = []

                if transcribe_dialogue:
                    print(f"Transcribing audio...")
                    if whisper is None:
                        raise RuntimeError(
                            "Whisper is unavailable. Install whisper, or set TRANSCRIPT=off for this chapter."
                        )
                    if model is None:
                        model = whisper.load_model("turbo", download_root=WHISPER_MODEL_DIR)
                    run(make_extract_audio(extracted, audio))
                    transcribe_audio(model, audio, final_srt, final_vtt, final_dir)
                    srt_to_ass(final_srt, final_ass)
                    subtitle_tracks.append({"path": final_ass, "title": "Dialogue", "forced": True})
                elif include_audio:
                    print("Skipping dialogue transcription (TRANSCRIPT=off).")
                else:
                    print("Skipping audio and transcription (AUDIO=off).")

                if people_tsv:
                    if tsv_people_to_ass(people_tsv, people_ass, clip_start=start_sec, clip_end=end_sec):
                        subtitle_tracks.append({"path": people_ass, "title": "People", "forced": False})
                    else:
                        print(f"People TSV had no entries: {people_tsv}")

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
                shutil.rmtree(temp_dir, ignore_errors=True)

            cur_count += 1

        print(f"All done")


def run_make_videos(
    *,
    title_filters=None,
    no_bob=False,
    frame_quality_tsv="",
    frame_quality_archive="",
):
    args = argparse.Namespace(
        archive=[],
        title=list(title_filters or []),
        title_exact=False,
        no_bob=bool(no_bob),
        frame_quality_tsv=str(frame_quality_tsv) if frame_quality_tsv else "",
        frame_quality_archive=str(frame_quality_archive or ""),
    )
    _run_with_args(args)


def main(argv=None):
    args = parse_args(argv)
    _run_with_args(args)

if __name__ == "__main__":
    main()
