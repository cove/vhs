import glob
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

        final_dir = VIDEOS
        if duration < 200:
            final_dir = CLIPS
        final_file = final_dir / f"{safe(title)}.mp4"
        archive_file = final_dir / f"{safe(title)}_archive.mkv"

        print(f"Extracting chapter: {title} ({format_hms(start_sec)} - {format_hms(end_sec)}) ")
        temp_extracted = final_dir / f"{safe(title)}_extracted.mkv"
        run([FFMPEG, "-v", "warning", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
             "-i", str(src), "-map", "0:v", "-map", "0:a", "-c", "copy",
             "-avoid_negative_ts", "make_zero", "-y", str(temp_extracted)])

        temp_avs = final_dir / f"{safe(title)}.avs"
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
FFmpegSource2("{temp_extracted}", atrack=-1) 
AssumeFPS(30000,1001) 
ConvertToYV12(matrix="Rec601") 
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3)
Crop(4,2,-8,-10)
LanczosResize(640,480)
ConvertToYV12(interlaced=false)
# Videos tend to be a little over saturated
Tweak(sat=0.8)
Prefetch()'''
        temp_avs.write_text(avs_script, encoding="ascii")

        print(f"Deinterlacing chapter: {title}")
        temp_qtgmc = final_dir / f"{safe(title)}_qtgmc.mkv"
        run([FFMPEG, "-v", "warning", "-i", str(temp_avs), "-i", str(temp_extracted),
            "-pix_fmt", "yuv422p",
            "-color_primaries:v", "6",
            "-color_trc:v", "6",
            "-colorspace:v", "5",
            "-color_range:v", "1",
            "-map", "0:v:0", "-c:v", "ffv1",
            "-level", "3", "-g", "1", "-coder", "1", "-context", "1",
            "-slices", "24", "-slicecrc", "1",
            "-map", "0:a", "-c:a", "copy",
            "-y", str(temp_qtgmc)])

        final_vtt = SUBTITLES / f"{title}.vtt"
        print(f"Transcribing audio: {final_file.name} to {final_vtt.name}")
        temp_transcript = final_dir / f"{safe(title)}_transcript.wav"
        subprocess.run([
            FFMPEG, "-v", "warning",
            "-i", str(temp_extracted),
            "-vn",
            "-af", "highpass=f=120,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=13,aresample=16000",
            "-c:a", "pcm_s16le",
            "-y",
            str(temp_transcript)
        ], check=True)

        model = whisper.load_model("turbo")
        vtt_writer = get_writer("vtt", str(SUBTITLES))
        result = model.transcribe(str(temp_transcript), language="en", fp16=False)
        vtt_writer(result, str(final_vtt))

        print(f"Final encoding: {final_file.name}")
        run([FFMPEG, "-v", "warning",
             "-i", str(temp_qtgmc.name),
             "-i", str(final_vtt),
             "-map_metadata", "-1",
             "-map_chapters", "-1",
             "-c:v", "libx265", "-crf", "18", "-preset", "veryslow",
             "-pix_fmt", "yuv420p10le",
             "-x265-params", "no-open-gop=1:bframes=8",
             "-c:a", "aac", "-b:a", "48k", "-ac", "1",
             "-af", "highpass=f=80,lowpass=f=14000",
             "-tag:v", "hvc1", "-brand", "mp42",
             "-map", "0:v:0",
             "-map", "0:a:0",
             "-map", "1:s:0",
             "-c:s", "mov_text",
             "-metadata:s:s:0", "language=eng",
             "-disposition:s:0", "forced",
             "-metadata:s:a:0", "language=eng",
             "-movflags", "+faststart+write_colr+use_metadata_tags",
             "-y", str(final_file)], cwd=final_dir)

        [p.unlink() for p in Path(final_dir).rglob('*.ffindex')]
        #temp_extracted.with_suffix(".ffindex").unlink(missing_ok=True)
        temp_extracted.unlink(missing_ok=True)
        temp_qtgmc.unlink(missing_ok=True)
        temp_avs.unlink(missing_ok=True)
        temp_transcript.unlink(missing_ok=True)

        print(f"  Done  {final_file.name}")

print("All done")

