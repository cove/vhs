#
# Common configuration and utility functions for all scripts:
# - Defines platform-specific binary paths (FFmpeg, b3sum, mediainfo, Whisper).
# - Sets up project directories for archives, videos, clips, and metadata.
# - Provides safe filename sanitization, HMS formatting, and subprocess wrappers.
# - Reads FFmetadata chapters and parses them into Python dicts.
# - Maintains a metadata_by_title cache for fast lookup.
# - Checks if chapter files are done based on size and existence.
# - Measures media duration via ffprobe.
#
import os, subprocess, sys
from pathlib import Path

# ---------------------------------------------------------
# Base Paths
# ---------------------------------------------------------

BASE = Path(__file__).parent.resolve()
FFMPEG_DIR = None

if sys.platform == "win32":
    FFMPEG_DIR = BASE / "software" / "Windows" / "FFmpeg-QTGMC Easy 2025.01.11"
    FFMPEG_BIN = FFMPEG_DIR / "ffmpeg.exe"
    FFPROBE_BIN = FFMPEG_DIR / "ffprobe.exe"
    B3SUM_BIN = BASE / "bin" / "b3sum_windows_x64_bin.exe"
    MEDIAINFO_BIN = BASE / "bin" / "mediainfo.exe"
    WHISPER_MODEL_DIR = BASE / "models" / "WhisperModel"
elif sys.platform == "darwin":
    FFMPEG_DIR = BASE / "bin"
    FFMPEG_BIN = FFMPEG_DIR / "ffmpeg-8.0.1.darwin.arm64"
    FFPROBE_BIN = FFMPEG_DIR / "ffprobe-8.0.1.darwin.arm64"
    B3SUM_BIN = BASE / "bin" / "b3sum"
    MEDIAINFO_BIN = "mediainfo"
    WHISPER_MODEL_DIR = BASE / "models" / "WhisperModel"
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
    base_dir = BASE.parent.parent
    METADATA_DIR = BASE / "metadata"
    ARCHIVE_DIR = base_dir / "Archive"
    VIDEOS_DIR = base_dir / "Videos"
    CLIPS_DIR = base_dir / "Clips"
    DRIVE_DIR = base_dir.resolve()

for _dir in (VIDEOS_DIR, CLIPS_DIR):
    _dir.mkdir(exist_ok=True)

QTGMC_DIR = FFMPEG_DIR
ARCHIVE_CHECKSUM_FILE = ARCHIVE_DIR / "00-archive-manifest-blake3sums.txt"
DRIVE_CHECKSUM_FILE = ARCHIVE_DIR / "00-drive-manifest-blake3sums.txt"

# Add FFmpeg binaries early to PATH so all scripts inherit it
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

def parse_chapters(path):
    chapters = []
    ffmetadata = {}
    cur = {}
    in_chapter = False
    seen_chapter = False

    def finalize(ch):
        tb = Fraction(1, 1)
        if "timebase" in ch:
            num, den = ch["timebase"].split("/", 1)
            tb = Fraction(int(num), int(den))

        if "start" in ch:
            s = int(ch["start"])
            ch["start"] = float(round(Fraction(s) * tb, 3))

        if "end" in ch:
            e = int(ch["end"])
            ch["end"] = float(round(Fraction(e) * tb, 3))

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
    if not FFMPEG_BIN.exists():
        raise FileNotFoundError(f"FFmpeg not found at {FFMPEG_BIN}")
