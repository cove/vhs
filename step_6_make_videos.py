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
BADFRAME_MONOTONIC_NEARBY_WINDOW = 30

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

def title_selected(title, filters):
    if not filters:
        return True
    text = str(title or "").strip().lower()
    for f in filters:
        needle = str(f or "").strip().lower()
        if needle and needle in text:
            return True
    return False

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render delivery videos/clips from archive chapters.")
    p.add_argument(
        "--title",
        action="append",
        default=[],
        help="Only process chapter titles that contain this substring (case-insensitive). Repeatable.",
    )
    return p.parse_args(argv)

def build_badframe_prefilter_lines(bad_source_frames=None, bad_repair_ranges=None):
    ranges = []
    if bad_repair_ranges is None:
        bad_source_frames = bad_source_frames or []
        frames = sorted({int(f) for f in bad_source_frames if int(f) >= 0})
        if not frames:
            return ""
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
            return ""

    bad_set = set()
    for a, b, _src in ranges:
        for f in range(a, b + 1):
            bad_set.add(f)

    def choose_repair_source_nearest(a, b):
        # Pick nearest clean source frame outside [a,b], scanning both sides.
        # Ties favor the previous frame to preserve motion direction.
        d = 1
        while True:
            left = a - d
            if left >= 0 and left not in bad_set:
                return left
            right = b + d
            if right not in bad_set:
                return right
            d += 1

    def choose_repair_source_at_or_after(floor_frame, b):
        src = max(int(floor_frame), b + 1)
        while src in bad_set:
            src += 1
        return src

    resolved_ranges = []
    prev_auto = None  # (a, b, src)
    for a, b, src_override in sorted(ranges, key=lambda x: (x[0], x[1])):
        src = src_override
        auto_selected = src is None or src < 0 or src in bad_set
        if auto_selected:
            src = choose_repair_source_nearest(a, b)
            if prev_auto is not None:
                _pa, prev_b, prev_src = prev_auto
                nearby = (a - prev_b) <= BADFRAME_MONOTONIC_NEARBY_WINDOW
                prev_is_future = prev_src > prev_b
                switched_to_past = src < a
                if nearby and (src < prev_src or (prev_is_future and switched_to_past)):
                    src = choose_repair_source_at_or_after(prev_src, b)
            prev_auto = (a, b, src)
        resolved_ranges.append((a, b, src))

    fix_lines = ["c = last"]
    # Freeze contiguous bad-frame runs to one neighboring clean frame.
    for a, b, src in sorted(resolved_ranges, key=lambda x: (x[0], x[1]), reverse=True):
        fix_lines.append(f"c = c.FreezeFrame({a},{b},{src})")
    fix_lines.append("c")
    return "\n".join(fix_lines) + "\n"

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

def find_badframes_tsv(archive_name):
    path = METADATA_DIR / archive_name / "badframes.tsv"
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

def load_badframe_repairs(tsv_path):
    if not tsv_path:
        return []

    def parse_bool_token(text):
        t = str(text or "").strip().lower()
        if t in {"1", "true", "yes", "y", "on"}:
            return True
        if t in {"", "0", "false", "no", "n", "off"}:
            return False
        return None

    out = []
    header = None
    raw = Path(tsv_path).read_text(encoding="utf-8-sig").splitlines()
    for line in raw:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "\t" in s:
            parts = [p.strip() for p in s.split("\t")]
        else:
            parts = [p.strip() for p in s.split(",")]
        if len(parts) < 2:
            continue

        low_parts = [p.strip().lower() for p in parts]
        if header is None and low_parts and low_parts[0].startswith("start"):
            start_idx = next((i for i, p in enumerate(low_parts) if p in {"start", "start_frame", "startframe"}), 0)
            end_idx = next((i for i, p in enumerate(low_parts) if p in {"end", "end_frame", "endframe"}), 1)
            note_idx = next((i for i, p in enumerate(low_parts) if p in {"note", "notes", "comment"}), None)
            no_pad_idx = next((i for i, p in enumerate(low_parts) if p in {"no_pad", "nopad", "no-pad"}), None)
            source_idx = next(
                (
                    i
                    for i, p in enumerate(low_parts)
                    if p in {
                        "source_frame",
                        "source",
                        "src_frame",
                        "repair_frame",
                        "replace_frame",
                        "replace_with",
                        "use_frame",
                    }
                ),
                None,
            )
            header = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "note_idx": note_idx,
                "no_pad_idx": no_pad_idx,
                "source_idx": source_idx,
            }
            continue

        start_idx = 0 if header is None else header["start_idx"]
        end_idx = 1 if header is None else header["end_idx"]
        if max(start_idx, end_idx) >= len(parts):
            continue
        try:
            a = int(parts[start_idx])
            b = int(parts[end_idx])
        except Exception:
            continue

        source = None
        source_text_as_note = ""
        source_idx = None if header is None else header["source_idx"]
        if source_idx is not None and source_idx < len(parts):
            source_raw = parts[source_idx].strip()
            if source_raw and source_raw.lower() not in {"auto", "none", "null"}:
                try:
                    source = int(source_raw)
                except Exception:
                    # Be forgiving if rows omit an empty source column and place note text
                    # in the source_frame position.
                    source_text_as_note = source_raw
                    print(f"Ignoring invalid source_frame value '{source_raw}' in {Path(tsv_path).name}")

        no_pad_explicit = None
        no_pad_idx = None if header is None else header["no_pad_idx"]
        if no_pad_idx is not None and no_pad_idx < len(parts):
            no_pad_raw = parts[no_pad_idx].strip()
            parsed_no_pad = parse_bool_token(no_pad_raw)
            if parsed_no_pad is None:
                print(f"Ignoring invalid no_pad value '{no_pad_raw}' in {Path(tsv_path).name}")
            else:
                no_pad_explicit = parsed_no_pad

        note = ""
        if header is None:
            if len(parts) >= 3:
                note = ",".join(parts[2:]).strip().lower()
        else:
            note_fields = []
            note_idx = header["note_idx"]
            if note_idx is not None and note_idx < len(parts):
                note_fields.append(parts[note_idx].strip())
            if source_text_as_note:
                note_fields.append(source_text_as_note)
            used = {start_idx, end_idx}
            if note_idx is not None:
                used.add(note_idx)
            if no_pad_idx is not None:
                used.add(no_pad_idx)
            if source_idx is not None:
                used.add(source_idx)
            for idx, part in enumerate(parts):
                if idx in used:
                    continue
                txt = part.strip()
                if txt:
                    note_fields.append(txt)
            note = ",".join(note_fields).strip().lower()

        if b < a:
            a, b = b, a
        if b < 0:
            continue
        a = max(0, a)
        span = b - a + 1
        allow_long = ("allow_long" in note) or ("allow-long" in note)
        if span > BADFRAME_MAX_SPAN_DEFAULT and not allow_long:
            print(
                f"Skipping suspicious badframe range {a}-{b} (span {span}); "
                "add note allow_long to keep it."
            )
            continue
        pad_before, pad_after = auto_badframe_pad(span)
        no_pad_from_note = ("no_pad" in note) or ("nopad" in note)
        use_no_pad = no_pad_from_note if no_pad_explicit is None else no_pad_explicit
        if use_no_pad:
            pad_before = 0
            pad_after = 0
        m = re.search(r"pad\s*=\s*(\d+)", note)
        if m:
            pad_before = int(m.group(1))
            pad_after = int(m.group(1))
        m = re.search(r"pad_before\s*=\s*(\d+)", note)
        if m:
            pad_before = int(m.group(1))
        m = re.search(r"pad_after\s*=\s*(\d+)", note)
        if m:
            pad_after = int(m.group(1))

        a = max(0, a - max(0, int(pad_before)))
        b = b + max(0, int(pad_after))
        out.append((a, b, source))

    return _merge_badframe_repairs(out)

def load_badframe_ranges(tsv_path):
    return [(a, b) for (a, b, _src) in load_badframe_repairs(tsv_path)]

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
):
    filter_text = Path(avs_filter_path).read_text(encoding="utf-8")
    prefilter_text = build_badframe_prefilter_lines(
        bad_source_frames or [],
        bad_repair_ranges=bad_repair_ranges,
    )
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
{filter_text}
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

def main(argv=None):
    args = parse_args(argv)
    model = None
    rebuild_selected = bool(args.title)

    for src in ARCHIVE_DIR.glob("*.mkv"):
        archive_name = src.stem
        chapters_file = METADATA_DIR / archive_name / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name}: no metadata found {chapters_file}")
            continue

        ffm, chapters = parse_chapters(chapters_file)
        if not chapters:
            print(f"No chapters for {src.name}")
            continue

        badframes_tsv = find_badframes_tsv(archive_name)
        archive_badframe_repairs = load_badframe_repairs(badframes_tsv) if badframes_tsv else []
        archive_badframe_ranges = [(a, b) for (a, b, _src) in archive_badframe_repairs]
        if badframes_tsv:
            if archive_badframe_repairs:
                explicit_count = sum(1 for (_a, _b, src) in archive_badframe_repairs if src is not None)
                print(
                    f"Loaded bad frame sidecar: {badframes_tsv.name} ({len(archive_badframe_repairs)} range(s), "
                    f"{explicit_count} with source override)"
                )
            else:
                print(f"Bad frame sidecar is present but empty: {badframes_tsv.name}")

        for ch in chapters:
            ch["duration"] = float(ch.get("end", 0)) - float(ch.get("start", 0))
        chapters.sort(key=lambda x: x["duration"])
        chapters = [ch for ch in chapters if title_selected(ch.get("title"), args.title)]
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

            try:
                print(f"Extracting chapter...")
                run(make_extract_chapter(src, extract_start_sec, extract_end_sec, extracted))

                print(f"Applying video filters...")
                if sys.platform == "win32":
                    if filter_script.exists():
                        manual_repairs = map_bad_repairs_to_chapter_local_ranges(archive_badframe_repairs, ch)
                        manual_source_frames = map_bad_ranges_to_chapter_local_frames(archive_badframe_ranges, ch)
                        if manual_source_frames:
                            print(
                                f"Sidecar source bad frame(s): {len(manual_source_frames)} -> "
                                + ",".join(str(f) for f in manual_source_frames[:12])
                                + ("..." if len(manual_source_frames) > 12 else "")
                            )
                        if manual_source_frames:
                            print(
                                f"Applying bad-frame repair at {len(manual_source_frames)} source frame(s): "
                                + ",".join(str(f) for f in manual_source_frames[:12])
                                + ("..." if len(manual_source_frames) > 12 else "")
                            )
                        manual_override_ranges = [r for r in manual_repairs if r[2] is not None]
                        if manual_override_ranges:
                            preview = ",".join(
                                f"{a}-{b}->{src}" for (a, b, src) in manual_override_ranges[:8]
                            )
                            if len(manual_override_ranges) > 8:
                                preview += "..."
                            print(
                                f"Applying explicit bad-frame source overrides: "
                                f"{len(manual_override_ranges)} range(s) [{preview}]"
                            )

                        script = make_create_avs(
                            extracted,
                            filter_script,
                            bad_source_frames=manual_source_frames,
                            bad_repair_ranges=manual_repairs,
                            chapter_start_frame=chapter_start_frame,
                            chapter_end_frame=chapter_end_frame,
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
                    raise RuntimeError("Unsupported platform for QTGMC: " + sys.platform)

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

if __name__ == "__main__":
    main()
