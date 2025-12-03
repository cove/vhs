import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"

ARCHIVE = BASE.parent / "Archive"
CLIPS = BASE.parent / "Clips"
VIDEOS = BASE.parent / "Videos"
SUBTITLES = BASE.parent / "Subtitles"

def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_chapter(path, title_name, uuid=None):
    ffmetadata = {}
    cur = {}
    in_chapter = False
    seen_chapter = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        # Global metadata
        if not seen_chapter and "=" in line and not line.startswith(("[", ";")):
            k, v = line.split("=", 1)
            ffmetadata[k.strip().lower()] = v.strip()
            continue

        # Chapter start
        if line == "[CHAPTER]":
            seen_chapter = True
            if cur and in_chapter:
                cur_title = cur.get("title", "")
                cur_uuid = cur.get("uuid", "")
                if cur_title == title_name and (uuid is None or cur_uuid == uuid):
                    return ffmetadata, cur
            cur = {}
            in_chapter = True
            continue

        # Chapter key=value
        if in_chapter and "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()

        # End of chapter block
        if in_chapter and not line and cur:
            cur_title = cur.get("title", "")
            cur_uuid = cur.get("uuid", "")
            if cur_title == title_name and (uuid is None or cur_uuid == uuid):
                return ffmetadata, cur
            cur = {}
            in_chapter = False

    # Last chapter
    if cur and in_chapter:
        cur_title = cur.get("title", "")
        cur_uuid = cur.get("uuid", "")
        if cur_title == title_name and (uuid is None or cur_uuid == uuid):
            return ffmetadata, cur

    # Not found
    return ffmetadata, None

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

    chapters_file = BASE / "media_metadata" / title / "chapters.ffmetadata"
    ffmetadata, ch = parse_chapter(chapters_file, title)

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
