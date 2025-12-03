import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"

ARCHIVE = BASE.parent / "Archive"
CLIPS = BASE.parent / "Clips"
VIDEOS = BASE.parent / "Videos"
SUBTITLES = BASE.parent / "Subtitles"
MEDIA_METADATA = BASE / "media_metadata"

metadata_by_title = {}

def parse_chapters(path):
    global_meta = {}
    chapters = {}
    current = None
    seen_chapter = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Global metadata
        if not seen_chapter and "=" in line and not line.startswith(("[", ";")):
            k, v = line.split("=", 1)
            global_meta[k.strip().lower()] = v.strip()
            continue

        # Chapter start
        if line == "[CHAPTER]":
            seen_chapter = True
            if current:
                chap_title = current.get("title", f"chapter_{len(chapters)+1}")
                chapters[chap_title] = current
            current = {}
            continue

        # Chapter body
        if "=" in line and current is not None:
            k, v = line.split("=", 1)
            current[k.lower()] = v.strip()

    # Add last chapter
    if current:
        chap_title = current.get("title", f"chapter_{len(chapters)+1}")
        chapters[chap_title] = current

    return global_meta, chapters

def load_all_metadata():
    for dirpath in MEDIA_METADATA.glob("*"):
        chapters_file = dirpath / "chapters.ffmetadata"
        if not chapters_file.exists():
            continue

        global_meta, chapters = parse_chapters(chapters_file)

        for chap_title, chap_data in chapters.items():
            ch_title = chap_data.get("title", "").strip()

            entry = {
                "global": global_meta,
                "chapter": chap_data,   # <-- only this chapter
                "path": chapters_file
            }

            metadata_by_title[ch_title.lower()] = entry

def load_metadata_for_video(video_path):
    title = video_path.stem.lower()

    if title in metadata_by_title:
        entry = metadata_by_title[title]
        ch = entry["chapter"]
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
print(f"Loaded metadata for {len(metadata_by_title)} titles")

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
        print(f"No chapters for {title}")
        continue

    ctime = ch.get("creation_time", "")
    location = ch.get("location", "")
    date = ffm.get("date", "")
    tape_id = ffm.get("tape_id", "")
    videographer = ffm.get("videographer", "")
    genre = ffm.get("genre", "")
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
        "-metadata", f"comment=Extracted chapter from \"{src.name}\" @ {start_hms}-{end_hms}",
        "-metadata", f"creation_time={ctime}",
        "-metadata", f"com.apple.quicktime.creationdate={ctime}",
        "-metadata", f"date={ctime}",
        "-metadata", f"genre={genre}",
        "-metadata", f"videographer={videographer}",
        "-metadata", f"tape_id={tape_id}",
        "-metadata:s:v:0", "language=eng",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "default",
        "-y", str(temp_file)
    ]

    print(f"Reapplying metadata to {src.name}")
    run(cmd)
    final_file.replace(temp_file)


print("All done")
