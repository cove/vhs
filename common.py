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
