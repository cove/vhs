import subprocess, sys, os, re
from pathlib import Path
import whisper
from whisper.utils import get_writer

BASE = Path(__file__).parent.resolve()
FFMPEG_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
QTGMC_DIR = FFMPEG_DIR

ARCHIVE = BASE.parent / "Archive"
VIDEOS = BASE.parent / "Videos"
VIDEOS.mkdir(exist_ok=True)
CLIPS = BASE.parent / "Clips"
CLIPS.mkdir(exist_ok=True)
SUBTITLES = BASE.parent / "Subtitles"
SUBTITLES.mkdir(exist_ok=True)

os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg.exe not found at {FFMPEG}")
    sys.exit(1)

# --- Helper Functions ---
def run(cmd, cwd=None):
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def parse_chapters(path):
    chapters = []
    ffmetadata = {}
    cur = {}
    in_chapter = False
    seen_chapter = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not seen_chapter and "=" in line and not line.startswith(("[", ";")):
            k, v = line.split("=", 1)
            ffmetadata[k.strip().lower()] = v.strip()
            continue
        if line == "[CHAPTER]":
            seen_chapter = True
            if cur and in_chapter:
                chapters.append(cur)
            cur = {}
            in_chapter = True
            continue
        if in_chapter and "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()
        if in_chapter and not line and cur:
            chapters.append(cur)
            cur = {}
            in_chapter = False
    if cur and in_chapter:
        chapters.append(cur)
    return ffmetadata, chapters

for src in ARCHIVE.glob("*.mkv"):
    prefix = "_".join(src.stem.rsplit("_", 2)[:2])
    chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
    if not chapters_file.exists():
        print(f"Skipping {src.name} — no metadata")
        continue

    ffmetadata, chapters = parse_chapters(chapters_file)
    if not chapters:
        print(f"No chapters for {src.name}")
        continue

    for ch in chapters:
        start = int(ch.get("start", 0))
        end = int(ch.get("end", 0))
        ch["duration"] = end - start
    chapters.sort(key=lambda x: x["duration"])

    print(f"Processing: {src.name} ({len(chapters)} chapters)")

    for i, ch in enumerate(chapters):
        title = ch.get("title", f"Chapter {i+1}")
        start_sec, end_sec = int(ch["start"]), int(ch["end"])
        duration = ch.get("duration")
        ctime = ch.get("creation_time", "")
        location = ch.get("location", "")
        uuid = ch.get("uuid", "")
        start_hms = format_hms(start_sec)
        end_hms = format_hms(end_sec)
        videographer = ch.get("videographer", ffmetadata.get('videographer', ''))

        final_dir = VIDEOS
        if duration < 200:
            final_dir = CLIPS
        archive_file = final_dir / f"{safe(title)}_archive.mkv"
        final_file = final_dir / f"{safe(title)}.mp4"

        if final_file.exists() and final_file.stat().st_size > 100_000:
            print(f"  Skipping existing chapter: {title}")
            continue

        temp_raw = final_dir / f"{safe(title)}_temp_raw.mkv"
        print(f"Extracting chapter: {title}")
        run([FFMPEG, "-v", "warning", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
             "-i", str(src), "-map", "0:v", "-map", "0:a", "-c", "copy",
             "-avoid_negative_ts", "make_zero", "-y", str(temp_raw)])

        print(f"Transcribing audio: {title}")
        model = whisper.load_model("large-v3")
        vtt_writer = get_writer("vtt", str(SUBTITLES))
        result = model.transcribe(str(temp_raw), language="en", fp16=False)
        final_vtt = SUBTITLES / f"{safe(title)}.vtt"
        vtt_writer(result, str(final_vtt))

        temp_avs = final_dir / f"{safe(title)}_temp.avs"
        avs_script = f'''
SetFilterMTMode("DEFAULT_MT_MODE", 2)
LoadPlugin("{QTGMC_DIR}/ffms2.dll") 
LoadPlugin("{QTGMC_DIR}/masktools2.dll") 
LoadPlugin("{QTGMC_DIR}/Rgtools.dll") 
LoadPlugin("{QTGMC_DIR}/mvtools2.dll") 
LoadPlugin("{QTGMC_DIR}/nnedi3.dll") 
LoadPlugin("{QTGMC_DIR}/yadifmod2.dll") 
LoadPlugin("{QTGMC_DIR}/fft3dfilter.dll") 
LoadPlugin("{QTGMC_DIR}/LoadDLL64.dll") 
LoadDLL("{QTGMC_DIR}/libfftw3f-3.dll") 
Import("{QTGMC_DIR}/Zs_RF_Shared.avsi") 
Import("{QTGMC_DIR}/QTGMC.avsi") 
FFmpegSource2("{temp_raw}", atrack=-1) 
AssumeFPS(30000,1001) 
ConvertToYV12(matrix="Rec601") 
#QTGMC(Preset="Slow",FPSDivisor=2,SourceMatch=3,Lossless=2,Sharpness=0.1)
Crop(4, 2, -8, -10)
LanczosResize(640,480) 
Prefetch()'''
        temp_avs.write_text(avs_script, encoding="ascii")

        print(f"Encoding: {final_file.name}")
        cmd = [
            FFMPEG,
            "-i", str(temp_raw), "-i", str(temp_avs), "-i", str(final_vtt),
            "-metadata", f"title={title}",
            "-metadata", f"comment=Chapter from file {src.name} @ {start_hms}-{end_hms}",
            "-metadata", f"creation_time={ctime}",
            "-metadata", f"date={ctime}",
            "-metadata", f"genre={ffmetadata.get('genre', '')}",
            "-metadata", f"videographer={videographer}",
            "-metadata", f"tape_id={ffmetadata.get('tape_id', '')}",
            "-metadata", f"com.apple.quicktime.creationdate={ctime}",
            "-metadata", f"com.apple.quicktime.uuid={uuid}",
        ]

        if location:
            iso6709 = location.rstrip("/") + "/"
            cmd += [
                "-metadata", f"com.apple.quicktime.location.ISO6709={iso6709}"
            ]

        cmd += ["-map", "0:v", "-map", "0:a", "-map", "1", "-map_metadata", "-1",
                "-tag:v", "hvc1",
                "-brand", "mp42",
                "-f", "mp4",
                "-c:v", "libx265",
                "-preset", "veryslow",
                "-crf", "16",
                "-profile:v", "main10",
                "-pix_fmt", "yuv420p10le",
                "-x265-params", "merange=57",
                "-x265-params", "psy-rd=2.0",
                "-x265-params", "aq-mode=3",
                "-x265-params", "aq-strength=1.0",
                "-x265-params", "bframes=8",
                "-x265-params", "keyint=600",
                "-x265-params", "rc-lookahead=80",
                "-x265-params", "no-sao=0",
                "-x265-params", "no-strong-intra-smoothing=0",
                "-x265-params", "deblock=-1:-1",
                "-x265-params", "ref=6",
                "-c:a", "aac", "-b:a", "48k", "-ac", "1",
                "-af", "highpass=f=80",
                "-af", "lowpass=f=14000",
                "-af", "afftdn=nf=-28",
                "-af", "dynaudnorm=g=15",
                "-c:s", "mov_text",
                "-metadata:s:s:0", "language=eng",
                "-disposition:s:0", "forced",
                "-metadata:s:a:0", "language=eng",
                "-movflags", "+faststart+write_colr+use_metadata_tags",
                "-y", str(final_file)]
        run(cmd, cwd=final_dir)

        for each in [temp_raw, temp_avs, Path(f"{safe(title)}_temp_raw.mkv.ffindex")]:
            each.unlink(missing_ok=True)

        print(f"  Done {final_file.name}")

print("All done")
