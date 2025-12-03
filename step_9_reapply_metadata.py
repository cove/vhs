import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"

ARCHIVE = BASE.parent / "Archive"
CLIPS = BASE.parent / "Clips"
VIDEOS = BASE.parent / "Videos"
SUBTITLES = BASE.parent / "Subtitles"
MEDIA_METADATA = BASE / "media_metadata"

metadata_by_uuid = {}
metadata_by_title = {}

def parse_chapters(path, title=None, uuid=None):
    global_meta = {}
    chapters = {}
    current = None
    in_chapter = False
    seen_chapter = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Global metadata section
        if not seen_chapter and "=" in line and not line.startswith(("[", ";")):
            k, v = line.split("=", 1)
            global_meta[k.strip().lower()] = v.strip()
            continue

        # Chapter start
        if line == "[CHAPTER]":
            seen_chapter = True
            if current:
                # Insert previous chapter before starting next
                chap_title = current.get("title", f"chapter_{len(chapters)+1}")
                chapters[chap_title] = current
            current = {}
            in_chapter = True
            continue

        # Chapter body
        if in_chapter and "=" in line:
            k, v = line.split("=", 1)
            current[k.lower()] = v.strip()

    # Add last chapter
    if current:
        chap_title = current.get("title", f"chapter_{len(chapters)+1}")
        chapters[chap_title] = current

    # --- Selection logic ---
    # 1. Select by UUID
    if uuid:
        for chap_title, chap_data in chapters.items():
            if chap_data.get("uuid") == uuid:
                return global_meta, chap_data
        return global_meta, None

    # 2. Select by title
    if title:
        return global_meta, chapters.get(title)

    # 3. No filters → return all chapters
    return global_meta, chapters

def load_all_metadata():
    for dirpath in MEDIA_METADATA.glob("*"):
        chapters_file = dirpath / "chapters.ffmetadata"
        if not chapters_file.exists():
            continue

        ffm, chapters = parse_chapters(chapters_file)

        # record uuid of archive
        archive_uuid = ffm.get("uuid", "").strip()
        archive_title = ffm.get("title", "").strip()

        entry = {
            "global": ffm,
            "chapters": chapters,
            "path": chapters_file
        }

        if archive_uuid:
            metadata_by_uuid[archive_uuid] = entry
        if archive_title:
            metadata_by_title[archive_title.lower()] = entry

def ffprobe_metadata_field(path, key):
    try:
        out = subprocess.check_output([
            FFMPEG,
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", f"stream_tags={key}",
            "-of", "default=nw=1:nk=1",
            str(path)
        ], text=True).strip()
        return out or ""
    except Exception:
        return ""

def load_metadata_for_video(video_path):
    # ---- A. Extract UUID from the video (if present)
    vid_uuid = ffprobe_metadata_field(video_path, "com.apple.quicktime.uuid")

    if vid_uuid and vid_uuid in metadata_by_uuid:
        entry = metadata_by_uuid[vid_uuid]
        # find matching chapter by uuid
        for ch in entry["chapters"]:
            if ch.get("uuid", "") == vid_uuid:
                return entry["global"], ch

    # ---- B. If no UUID match, try title
    title = ffprobe_metadata_field(video_path, "title").lower()
    if title in metadata_by_title:
        entry = metadata_by_title[title]
        for ch in entry["chapters"]:
            if ch.get("title", "").lower() == title:
                return entry["global"], ch

    return None, None

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

load_all_metadata()
print(f"Loaded metadata for {len(metadata_by_uuid)} UUIDs and {len(metadata_by_title)} titles")

all_videos = (f for folder in [VIDEOS, CLIPS] for f in folder.glob("*.mp4"))
for src in all_videos:
    title = src.stem
    subtitle_file = SUBTITLES / f"{title}.vtt"
    in_progress = src.with_suffix(".avs")

    if src.name.endswith(".subtitle_temp.mp4") or in_progress.exists():
        continue

    if not subtitle_file.exists():
        print(f"No subtitles found for {src.name}, skipping")
        continue

    final_file = src
    temp_file = src.with_suffix(".subtitle_temp.mp4")

    ffm, ch = load_metadata_for_video(src)
    if not ch:
        print(f"No chapters for {src.name}")
        continue

    ctime = ch.get("creation_time", "")
    uuid = ch.get("uuid", "")
    location = ch.get("location", "")
    date = ch.get("date", "")
    tape_id = ch.get("tape_id", "")
    videographer = ch.get("videographer", "")
    genre = ch.get("genre", "")
    start_hms = ch.get("start", "0")
    end_hms = ch.get("end", "0")

    cmd = [
        FFMPEG,
        "-v", "warning",
        "-i", str(src),
        "-i", str(subtitle_file),
        "-map", "0:v",
        "-map", "0:a",
        "-map", "1",
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text"
    ]

    if location:
        iso6709 = location.rstrip("/") + "/"
        cmd += [
            "-metadata", f"com.apple.quicktime.location.ISO6709={iso6709}",
        ]

    cmd += [
        "-metadata", f"title={title}",
        "-metadata", f"comment=Extracted chapter from archive_file=\"{src.name}\", time_range={start_hms}-{end_hms}",
        "-metadata", f"creation_time={ctime}",
        "-metadata", f"com.apple.quicktime.creationdate={ctime}",
        "-metadata", f"com.apple.quicktime.uuid={uuid}",
        "-metadata", f"date={ctime}",
        "-metadata", f"genre={genre}",
        "-metadata", f"videographer={videographer}",
        "-metadata", f"tape_id={tape_id}"
        "-metadata:s:v:0", "language=eng",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "default",
        "-y", str(temp_file)
    ]

    print(f"Applying subtitles to {src.name}")
    run(cmd)

print("All done")
