#
# Common configuration and utility functions for all scripts:
# - Defines platform-specific binary paths (FFmpeg, mediainfo, Whisper, legacy b3sum).
# - Provides SHA3-256 checksum helpers for manifests and verification.
# - Sets up project directories for archives, videos, clips, and metadata.
# - Provides safe filename sanitization, HMS formatting, and subprocess wrappers.
# - Reads FFmetadata chapters and parses them into Python dicts.
# - Maintains a metadata_by_title cache for fast lookup.
# - Checks if chapter files are done based on size and existence.
# - Measures media duration via ffprobe.
#
import os, shutil, subprocess, sys
import hashlib
from dataclasses import replace as dataclass_replace
from pathlib import Path

# ---------------------------------------------------------
# Base Paths
# ---------------------------------------------------------

BASE = Path(__file__).parent.resolve()
FFMPEG_DIR = None

def _resolve_command(cmd_name, bundled_path=None):
    """
    Resolve an executable command, preferring a bundled path, then PATH lookup.
    Returns (command, parent_dir_or_none).
    """
    if bundled_path is not None:
        p = Path(bundled_path)
        if p.exists():
            p = p.resolve()
            return p, p.parent

    found = shutil.which(cmd_name)
    if found:
        p = Path(found)
        return p, p.parent

    # Keep a simple command token fallback so subprocess can still try PATH.
    return Path(cmd_name), None

def _command_exists(cmd):
    cmd_text = str(cmd)
    p = Path(cmd_text)
    if p.is_absolute() or "/" in cmd_text or "\\" in cmd_text:
        return p.exists()
    return shutil.which(cmd_text) is not None

if sys.platform == "win32":
    FFMPEG_DIR = BASE / "software" / "Windows" / "FFmpeg-QTGMC Easy 2025.01.11"
    FFMPEG_BIN = FFMPEG_DIR / "ffmpeg.exe"
    FFPROBE_BIN = FFMPEG_DIR / "ffprobe.exe"
    B3SUM_BIN = BASE / "bin" / "b3sum_windows_x64_bin.exe"
    MEDIAINFO_BIN = BASE / "bin" / "MediaInfo.exe"
    WHISPER_MODEL_DIR = BASE / "models" / "WhisperModel"
elif sys.platform == "darwin":
    FFMPEG_DIR = BASE / "bin"
    FFMPEG_BIN = FFMPEG_DIR / "ffmpeg-8.0.1.darwin.arm64"
    FFPROBE_BIN = FFMPEG_DIR / "ffprobe-8.0.1.darwin.arm64"
    B3SUM_BIN = BASE / "bin" / "b3sum"
    MEDIAINFO_BIN = "mediainfo"
    WHISPER_MODEL_DIR = BASE / "models" / "WhisperModel"
elif sys.platform.startswith("linux"):
    ffmpeg_override = os.getenv("FFMPEG_BIN")
    ffprobe_override = os.getenv("FFPROBE_BIN")
    b3sum_override = os.getenv("B3SUM_BIN")
    mediainfo_override = os.getenv("MEDIAINFO_BIN")

    if ffmpeg_override:
        FFMPEG_BIN = Path(ffmpeg_override)
        ffmpeg_dir = (
            Path(ffmpeg_override).parent
            if (Path(ffmpeg_override).is_absolute() or "/" in ffmpeg_override or "\\" in ffmpeg_override)
            else None
        )
    else:
        FFMPEG_BIN, ffmpeg_dir = _resolve_command("ffmpeg", BASE / "bin" / "ffmpeg")

    if ffprobe_override:
        FFPROBE_BIN = Path(ffprobe_override)
        ffprobe_dir = (
            Path(ffprobe_override).parent
            if (Path(ffprobe_override).is_absolute() or "/" in ffprobe_override or "\\" in ffprobe_override)
            else None
        )
    else:
        FFPROBE_BIN, ffprobe_dir = _resolve_command("ffprobe", BASE / "bin" / "ffprobe")

    if b3sum_override:
        B3SUM_BIN = Path(b3sum_override)
    else:
        B3SUM_BIN, _ = _resolve_command("b3sum", BASE / "bin" / "b3sum")

    if mediainfo_override:
        mediainfo_cmd = Path(mediainfo_override)
    else:
        mediainfo_cmd, _ = _resolve_command("mediainfo")

    MEDIAINFO_BIN = mediainfo_cmd
    WHISPER_MODEL_DIR = BASE / "models" / "WhisperModel"
    # Prefer ffmpeg folder for PATH augmentation; fall back to ffprobe folder.
    FFMPEG_DIR = ffmpeg_dir or ffprobe_dir
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

# ---------------------------------------------------------
# Project Directories (shared between scripts)
# ---------------------------------------------------------
ARCHIVE_DIR = VIDEOS_DIR = CLIPS_DIR = None

if os.getenv("TEST_ENV") == "1":
    base_dir = BASE / "test"
    METADATA_DIR = base_dir / "metadata"
    ARCHIVE_DIR = base_dir / "Archive"
    VIDEOS_DIR = base_dir / "Videos"
    CLIPS_DIR = base_dir / "Clips"
    DRIVE_DIR = base_dir.resolve()
else:
    base_dir = BASE.parent
    METADATA_DIR = BASE / "metadata"
    ARCHIVE_DIR = base_dir / "Archive"
    VIDEOS_DIR = base_dir / "Videos"
    CLIPS_DIR = base_dir / "Clips"
    DRIVE_DIR = base_dir.resolve()

for _dir in (VIDEOS_DIR, CLIPS_DIR):
    _dir.mkdir(exist_ok=True)

QTGMC_DIR = FFMPEG_DIR
ARCHIVE_CHECKSUM_FILE = ARCHIVE_DIR / "00-archive-manifest-sha3-256sums.txt"
DRIVE_CHECKSUM_FILE = ARCHIVE_DIR / "00-drive-manifest-sha3-256sums.txt"
LEGACY_ARCHIVE_CHECKSUM_FILE = ARCHIVE_DIR / "00-archive-manifest-blake3sums.txt"
LEGACY_DRIVE_CHECKSUM_FILE = ARCHIVE_DIR / "00-drive-manifest-blake3sums.txt"

# Add FFmpeg binaries early to PATH so all scripts inherit it
if FFMPEG_DIR:
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

# ---------------------------------------------------------
# Shared FFmpeg Settings
# ---------------------------------------------------------

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def run(cmd, cwd=None):
    print("Command: " + " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

def make_extract_chapter(
    src,
    start,
    end,
    dest,
    start_frame=None,
    end_frame=None,
    debug_frame_numbers=False,
):
    """
    Frame-exact chapter extraction command builder.
    Uses select+setpts for exact frame slicing and optional drawtext overlay.
    """
    if start_frame is None or end_frame is None:
        raise ValueError("make_extract_chapter requires start_frame and end_frame.")
    s_frame = int(start_frame)
    e_frame = int(end_frame)
    if e_frame <= s_frame:
        e_frame = s_frame + 1

    vf_filters = [
        f"select='between(n\\,{s_frame}\\,{e_frame - 1})'",
        "setpts=N/FRAME_RATE/TB",
    ]
    if bool(debug_frame_numbers):
        local_label = "%{eif\\:n\\:d}"
        global_label = f"%{{eif\\:n+{s_frame}\\:d}}"
        font_expr = ""
        win_font = Path("C:/Windows/Fonts/consola.ttf")
        if win_font.exists():
            font_expr = "fontfile='C\\:/Windows/Fonts/consola.ttf'"
        vf_filters.append(
            "drawtext="
            + f"text='local={local_label} global={global_label}'"
            + (f":{font_expr}" if font_expr else "")
            + ":x=16:y=16:fontsize=24:"
            + "fontcolor=white:box=1:boxcolor=black@0.55:borderw=2"
        )
    vf_select = ",".join(vf_filters)

    af_trim = f"atrim=start={float(start):.6f}:end={float(end):.6f},asetpts=PTS-STARTPTS"
    return [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(src),
        "-vf", vf_select,
        "-af", af_trim,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-fps_mode:v:0", "passthrough",
        "-c:v", "ffv1",
        "-level", "3", "-coder", "1", "-context", "1",
        "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-y", str(dest),
    ]

def resolve_path(path_value, base_dir=None):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else BASE
    return (base / path).resolve()

def resolve_optional_path(path_value, default_path, base_dir=None):
    text = str(path_value or "").strip()
    if text:
        return resolve_path(text, base_dir=base_dir)
    return Path(default_path)

def require_non_empty(text, field_name):
    value = str(text or "").strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
    return value

def apply_config_overrides(config, **overrides):
    cleaned = {k: v for k, v in overrides.items() if v is not None}
    if not cleaned:
        return config
    return dataclass_replace(config, **cleaned)

def read_ffmetadata_title(path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("[CHAPTER]"):
                break
            if line.startswith("title="):
                return line.split("=", 1)[1].strip()
    return ""

from fractions import Fraction

def _parse_timebase_fraction(text):
    raw = str(text or "").strip()
    if "/" in raw:
        num_s, den_s = raw.split("/", 1)
        num = int(num_s)
        den = int(den_s)
    else:
        num = int(raw)
        den = 1
    if den == 0:
        raise ValueError("timebase denominator cannot be zero")
    if den < 0:
        num = -num
        den = -den
    return int(num), int(den)

def _round_fraction_nearest_int(frac):
    frac = Fraction(frac)
    if frac >= 0:
        return int((frac.numerator * 2 + frac.denominator) // (2 * frac.denominator))
    pos = -frac
    return -int((pos.numerator * 2 + pos.denominator) // (2 * pos.denominator))

def chapter_frame_bounds(chapter, fps_num=30000, fps_den=1001):
    """
    Return chapter [start_frame, end_frame) using exact rational math when raw
    ffmetadata ticks/timebase are available. Falls back to float seconds.
    """
    fps = Fraction(int(fps_num), int(fps_den))
    try:
        s_raw = int(chapter.get("start_raw"))
        e_raw = int(chapter.get("end_raw"))
        tb_num = int(chapter.get("timebase_num"))
        tb_den = int(chapter.get("timebase_den"))
        tb = Fraction(tb_num, tb_den)
        s = _round_fraction_nearest_int(Fraction(s_raw) * tb * fps)
        e = _round_fraction_nearest_int(Fraction(e_raw) * tb * fps)
    except Exception:
        s = int(round(float(chapter.get("start", 0.0)) * float(fps_num) / float(fps_den)))
        e = int(round(float(chapter.get("end", 0.0)) * float(fps_num) / float(fps_den)))
    if e < s:
        e = s
    return int(s), int(e)

def parse_chapters(path):
    chapters = []
    ffmetadata = {}
    cur = {}
    in_chapter = False
    seen_chapter = False

    def finalize(ch):
        tb_num = 1
        tb_den = 1
        if "timebase" in ch:
            tb_num, tb_den = _parse_timebase_fraction(ch["timebase"])
        tb = Fraction(int(tb_num), int(tb_den))
        ch["timebase_num"] = int(tb_num)
        ch["timebase_den"] = int(tb_den)

        if "start" in ch:
            s_raw = int(ch["start"])
            ch["start_raw"] = int(s_raw)
            ch["start"] = float(Fraction(s_raw) * tb)
        else:
            ch["start"] = float(ch.get("start", 0.0))

        if "end" in ch:
            e_raw = int(ch["end"])
            ch["end_raw"] = int(e_raw)
            ch["end"] = float(Fraction(e_raw) * tb)
        else:
            ch["end"] = float(ch.get("end", 0.0))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # global ffmetadata before any chapter
        if not seen_chapter and "=" in line and not line.startswith(("[", ";")):
            k, v = line.split("=", 1)
            ffmetadata[k.strip().lower()] = v.strip()
            continue

        if line == "[CHAPTER]":
            # finish previous chapter
            if cur and in_chapter:
                finalize(cur)
                chapters.append(cur)
            cur = {}
            in_chapter = True
            seen_chapter = True
            continue

        if in_chapter and "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()

    # finalize last chapter
    if cur and in_chapter:
        finalize(cur)
        chapters.append(cur)

    return ffmetadata, chapters

def parse_bad_frames_csv(text):
    vals = []
    seen = set()
    for raw in str(text or "").split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            fid = int(token)
        except Exception:
            continue
        if fid < 0 or fid in seen:
            continue
        seen.add(fid)
        vals.append(fid)
    vals.sort()
    return vals

def format_bad_frames_csv(frame_ids):
    vals = sorted({int(x) for x in (frame_ids or []) if int(x) >= 0})
    return ",".join(str(v) for v in vals)

def load_bad_frames_by_chapter(path):
    bad_by_title = {}
    _ffm, chapters = parse_chapters(Path(path))
    for ch in chapters:
        title = str(ch.get("title", "")).strip()
        if not title:
            continue
        bad_by_title[title] = parse_bad_frames_csv(ch.get("bad_frames", ""))
    return bad_by_title

_FRAME_LIST_KEYS = (
    "bad_frames",
    "bad_frame_override",
    "good_frame_override",
)

def _normalize_frame_list_key(key):
    k = str(key or "").strip().lower()
    return k if k in _FRAME_LIST_KEYS else ""

def update_chapter_frame_lists_in_ffmetadata(path, chapter_frame_lists):
    """
    Update chapter frame-list metadata lines in chapters.ffmetadata in-place.
    chapter_frame_lists:
      {
        chapter_title: {
          "BAD_FRAMES": [...],
          "BAD_FRAME_OVERRIDE": [...],
          "GOOD_FRAME_OVERRIDE": [...],
        }
      }
    Empty lists remove the corresponding key from that chapter block.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"chapters.ffmetadata not found: {p}")

    def _norm_title(text):
        return " ".join(str(text or "").strip().lower().split())

    pending = {}
    for raw_title, raw_fields in (chapter_frame_lists or {}).items():
        nk = _norm_title(raw_title)
        if not nk:
            continue
        field_map = {}
        for raw_key, vals in dict(raw_fields or {}).items():
            key = _normalize_frame_list_key(raw_key)
            if not key:
                continue
            field_map[key] = list(vals or [])
        if field_map:
            pending[nk] = field_map
    if not pending:
        return 0

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    i = 0
    touched = 0

    while i < len(lines):
        line = lines[i]
        if line.strip() != "[CHAPTER]":
            out.append(line)
            i += 1
            continue

        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip() != "[CHAPTER]":
            block.append(lines[i])
            i += 1

        title = ""
        for bline in block:
            s = bline.strip()
            if "=" in s and not s.startswith(";"):
                k, v = s.split("=", 1)
                if k.strip().lower() == "title":
                    title = v.strip()
                    break

        nk = _norm_title(title)
        updates = pending.get(nk)
        should_update = updates is not None
        remove_keys = set(updates.keys()) if should_update else set()

        title_idx = -1
        cleaned = []
        for bline in block:
            s = bline.strip()
            if "=" in s and not s.startswith(";"):
                k, _v = s.split("=", 1)
                key = k.strip().lower()
                if key == "title":
                    title_idx = len(cleaned)
                if should_update and key in remove_keys:
                    continue
            cleaned.append(bline)

        if should_update:
            insert_at = title_idx + 1 if title_idx >= 0 else len(cleaned)
            for key in _FRAME_LIST_KEYS:
                if key not in updates:
                    continue
                csv = format_bad_frames_csv(updates[key])
                if csv:
                    cleaned.insert(insert_at, f"{key.upper()}={csv}")
                    insert_at += 1
            touched += 1

        out.extend(cleaned)

    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return touched

def update_chapter_bad_frames_in_ffmetadata(path, chapter_bad_frames):
    """
    Update BAD_FRAMES lines in chapters.ffmetadata in-place.
    chapter_bad_frames: {chapter_title: [global_frame_ids]}.
    """
    mapped = {
        title: {"BAD_FRAMES": list(vals or [])}
        for title, vals in (chapter_bad_frames or {}).items()
    }
    return update_chapter_frame_lists_in_ffmetadata(path, mapped)

metadata_by_title = {}
def load_all_metadata():
    for dirpath in METADATA_DIR.glob("*"):
        chapters_file = dirpath / "chapters.ffmetadata"
        if not chapters_file.exists():
            continue

        # Parse chapters metadata
        global_meta, chapters = parse_chapters(chapters_file)

        # Load comments.txt if it exists
        comments_file = dirpath / "comments.txt"
        comments = []
        if comments_file.exists():
            with comments_file.open("r", encoding="utf-8") as f:
                comments = [line.strip() for line in f if line.strip()]

        # Populate metadata_by_title
        for chap_data in chapters:
            ch_title = chap_data.get("title").strip()
            entry = {
                "global": global_meta,
                "chapter": chap_data,
                "path": chapters_file,
                "comments": comments  # new field
            }
            metadata_by_title[ch_title] = entry

def get_metadata_for_video(title):
    entry = metadata_by_title.get(title)
    if entry:
        return entry["global"], entry["chapter"]
    return None, None

def is_chapter_done(final_file):
    if not final_file.exists():
        return False

    if final_file.stat().st_size < 100_000:
        return False

    return True

def duration(path):
    try:
        out = subprocess.check_output(
            [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True,
        ).strip()
        return float(out) if out else 0
    except:
        return 0

# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def ensure_ffmpeg_exists():
    if not _command_exists(FFMPEG_BIN):
        raise FileNotFoundError(f"FFmpeg not found at {FFMPEG_BIN}")

# ---------------------------------------------------------
# SHA3-256 Checksums
# ---------------------------------------------------------

def sha3sum_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha3_256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def write_sha3_manifest(root_dir, manifest_path, relative_base=None, ignore_fn=None):
    root_dir = Path(root_dir)
    manifest_path = Path(manifest_path)
    relative_base = Path(relative_base) if relative_base else root_dir

    manifest_path.unlink(missing_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "a", encoding="utf-8") as out:
        for file_path in root_dir.rglob("*"):
            if ignore_fn and ignore_fn(file_path):
                continue

            if not file_path.is_file():
                continue

            if file_path.resolve() == manifest_path.resolve():
                continue

            digest = sha3sum_file(file_path)
            rel_path = file_path.relative_to(relative_base)
            out.write(f"{digest}  {rel_path}\n")

    print("Checksums written to:", manifest_path)

def verify_sha3_manifest(root_dir, manifest_path):
    root_dir = Path(root_dir)
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    failures = 0
    total = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                print(f"Skipping malformed line: {raw_line.rstrip()}")
                failures += 1
                continue

            expected, rel_path = parts
            rel_path = rel_path.lstrip("*")
            target = root_dir / rel_path

            if not target.exists():
                print(f"MISSING: {rel_path}")
                failures += 1
                continue

            actual = sha3sum_file(target)
            total += 1

            if actual.lower() != expected.lower():
                print(f"MISMATCH: {rel_path}")
                failures += 1

    if failures == 0:
        print("ALL FILES VERIFIED - CHECKSUMS MATCH!")
        return 0

    print(f"{failures} FILES FAILED VERIFICATION")
    return 1

def verify_blake3_manifest(root_dir, manifest_path):
    if not _command_exists(B3SUM_BIN):
        print(f"ERROR: b3sum not found at {B3SUM_BIN}")
        return 1

    r = subprocess.run(
        [str(B3SUM_BIN), "-c", str(manifest_path)],
        cwd=Path(root_dir),
        capture_output=True,
        text=True,
    )
    print(r.stdout or r.stderr)

    if r.returncode == 0:
        print("ALL FILES VERIFIED - CHECKSUMS MATCH!")
    else:
        print("SOME FILES FAILED VERIFICATION!")

    return r.returncode

def detect_manifest_algo(manifest_path):
    name = Path(manifest_path).name.lower()
    if "blake3" in name:
        return "blake3"
    if "sha3" in name:
        return "sha3"
    return None

def verify_manifest(root_dir, manifest_path, algo="auto"):
    algo = (algo or "auto").lower()
    manifest_path = Path(manifest_path)

    if algo == "auto":
        algo = detect_manifest_algo(manifest_path) or "sha3"

    if algo == "blake3":
        return verify_blake3_manifest(root_dir, manifest_path)

    if algo == "sha3":
        return verify_sha3_manifest(root_dir, manifest_path)

    print(f"Unknown checksum algorithm: {algo}")
    return 1
