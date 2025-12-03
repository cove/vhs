import subprocess
import sys
import os
from pathlib import Path
import psutil
import concurrent.futures
import whisper
from whisper.utils import get_writer

BASE = Path(__file__).parent.resolve()
FFMPEG_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE = BASE / "bin" / "ffprobe.exe"

QTGMC_DIR = FFMPEG_DIR

ARCHIVE = BASE.parent / "Archive"
VIDEOS = BASE.parent / "Videos"
CLIPS = BASE.parent / "Clips"
SUBTITLES = BASE.parent / "Subtitles"

for d in [VIDEOS, CLIPS, SUBTITLES]:
    d.mkdir(exist_ok=True)

os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg.exe not found at {FFMPEG}")
    sys.exit(1)

def run(cmd,cwd=None,cpu_list=None):
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd)
    if cpu_list:
        try:
            p = psutil.Process(proc.pid)
            p.cpu_affinity(cpu_list)
        except Exception as e:
            print(f"Warning: failed to set CPU affinity: {e}", file=sys.stderr)

    retcode = proc.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd)


def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_video_duration(path):
    try:
        out = subprocess.check_output([
            FFPROBE,
            "-v", "error", "-threads", "1",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ], text=True)
        return float(out.strip())
    except Exception:
        return None

def is_chapter_done(final_file, expected_duration):
    if not final_file.exists():
        return False
    actual_duration = get_video_duration(final_file)
    if actual_duration is None:
        return False
    # allow small rounding difference (0.5s)
    return abs(actual_duration - expected_duration) < 0.5

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

def extract_chapter(src, start, end, dest):
    run([FFMPEG,
         "-nostdin",
         "-v", "error",
         "-f", "matroska",
         "-i", str(src),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-r", "30000 / 1001",
         "-pix_fmt", "yuv422p",
         "-color_primaries:v", "6",
         "-color_trc:v", "6",
         "-colorspace:v", "5",
         "-color_range:v", "1",
         "-ac", "1",
         "-map", "0:v", "-map", "0:a", "-c", "copy",
         "-avoid_negative_ts", "make_zero", "-y", str(dest)])

def create_avs(temp_extracted, avs_path):
    avs_script = f'''
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
Tweak(sat=0.85)
'''
    avs_path.write_text(avs_script, encoding="ascii")

def deinterlace(temp_avs, temp_extracted, temp_qtgmc, cpuset=None):
    run([FFMPEG,
         "-nostdin",
         "-v", "warning",
         "-threads", "1",
         "-f", "matroska",
         "-i", str(temp_avs.name),
         "-i", str(temp_extracted.name),
         "-r", "30000 / 1001",
         "-pix_fmt", "yuv422p",
         "-color_primaries:v", "6",
         "-color_trc:v", "6",
         "-colorspace:v", "5",
         "-color_range:v", "1",
         "-map", "0:v:0", "-c:v", "ffv1",
         "-level", "3", "-g", "1", "-coder", "1", "-context", "1",
         "-slices", "24", "-slicecrc", "1",
         "-ac", "1",
         "-map", "0:a", "-c:a", "copy",
         "-y", str(temp_qtgmc)], temp_qtgmc.parent, cpuset)

def extract_audio(temp_extracted, temp_transcript, cpuset=None):
    run([
        FFMPEG, "-nostdin", "-v", "warning", "-threads", "1",
        "-i", str(temp_extracted),
        "-vn",
        "-af", "highpass=f=120,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=13,aresample=16000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-y",
        str(temp_transcript)
    ], temp_extracted.parent, cpuset)

def transcribe_audio(model, temp_transcript, final_vtt):
    vtt_writer = get_writer("vtt", str(SUBTITLES))
    result = model.transcribe(str(temp_transcript), language="en", fp16=False)
    vtt_writer(result, str(final_vtt))

def encode_final(temp_qtgmc, final_vtt, final_file, title, ffmetadata, start_hms, end_hms, ctime, location, cpuset=None):
    cmd = [FFMPEG,
           "-nostdin",
           "-v",
           "warning",
           "-threads", "1",
           "-i", str(temp_qtgmc),
           "-i", str(final_vtt),
           "-map_metadata", "-1",
           "-map_chapters", "-1",
           "-c:v", "libx265", "-crf", "18", "-preset", "veryslow",
           "-r", "30000 / 1001",
           "-pix_fmt", "yuv420p10le",
           "-x265-params", "no-open-gop=1:bframes=8",
           "-c:a", "aac", "-b:a", "48k", "-ac", "1",
           "-af", "highpass=f=80,lowpass=f=14000,loudnorm=I=-16:TP=-1.5:LRA=11",
           "-tag:v", "hvc1", "-brand", "mp42",
           "-map", "0:v:0", "-map", "0:a:0", "-map", "1:s:0",
           "-c:s", "mov_text",
           "-metadata:s:s:0", "language=eng",
           "-disposition:s:0", "forced",
           "-metadata:s:a:0", "language=eng",
           "-metadata", f"title={title}",
           "-metadata", f"comment=Chapter from archive {ffmetadata.get('title','')} @ {start_hms}-{end_hms}",
           "-metadata", f"creation_time={ctime}",
           "-metadata", f"com.apple.quicktime.creationdate={ctime}",
           "-metadata", f"date={ctime}",
           "-metadata", f"genre={ffmetadata.get('genre','')}",
           "-metadata", f"videographer={ffmetadata.get('videographer','')}",
           "-metadata", f"tape_id={ffmetadata.get('tape_id','')}"
    ]
    if location:
        iso6709 = location.rstrip("/") + "/"
        cmd += ["-metadata", f"com.apple.quicktime.location.ISO6709={iso6709}"]
    cmd += ["-movflags", "+faststart+write_colr+use_metadata_tags", "-y", str(final_file)]
    run(cmd, temp_qtgmc.parent, cpuset)

def cleanup_temp_files(*files):
    for f in files:
        f.unlink(missing_ok=True)
        f.with_suffix(".ffindex").unlink(missing_ok=True)

def process_chapter(chapter_job, cpuset):
    p = psutil.Process()
    p.cpu_affinity(cpuset)

    model, src, ffmetadata, ch, i = chapter_job
    title = ch.get("title", f"Chapter {i+1}")
    start_sec, end_sec = int(ch["start"]), int(ch["end"])
    duration = end_sec - start_sec
    ctime = ch.get("creation_time", "")
    location = ch.get("location", "")

    final_dir = VIDEOS if duration >= 200 else CLIPS
    final_file = final_dir / f"{safe(title)}.mp4"
    if is_chapter_done(final_file, duration):
        return f"Skipped existing: {title}"

    print(f"Extracting chapter: {title} ({format_hms(start_sec)} - {format_hms(end_sec)})")
    temp_extracted = final_dir / f"{safe(title)}_extracted.mkv"
    extract_chapter(src, start_sec, end_sec, temp_extracted)

    print(f"Deinterlacing chapter: {title}")
    temp_avs = final_dir / f"{safe(title)}.avs"
    create_avs(temp_extracted.name, temp_avs)

    temp_qtgmc = final_dir / f"{safe(title)}_qtgmc.mkv"
    deinterlace(temp_avs, temp_extracted, temp_qtgmc, cpuset)

    print(f"Transcribing chapter: {title}")
    temp_transcript = final_dir / f"{safe(title)}_transcript.wav"
    final_vtt = SUBTITLES / f"{safe(title)}.vtt"
    extract_audio(temp_extracted, temp_transcript, cpuset)
    transcribe_audio(model, temp_transcript, final_vtt)

    print(f"Final encoding chapter: {final_file.name}")
    start_hms = format_hms(start_sec)
    end_hms = format_hms(end_sec)
    encode_final(temp_qtgmc, final_vtt, final_file, title, ffmetadata, start_hms, end_hms, ctime, location, cpuset)

    cleanup_temp_files(temp_extracted, temp_qtgmc, temp_avs, temp_transcript)
    return f"Done: {final_file.name}"

def main():
    model = whisper.load_model("turbo")
    chapter_jobs = []

    # Load all metadata upfront
    for src in ARCHIVE.glob("*.mkv"):
        prefix = "_".join(src.stem.rsplit("_", 2)[:2])
        chapters_file = BASE / "metadata" / prefix / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name} — no metadata")
            continue
        ffmetadata, chapters = parse_chapters(chapters_file)
        if not chapters:
            print(f"No chapters for {src.name}")
            continue
        for i, ch in enumerate(chapters):
            start = int(ch.get("start", 0))
            end = int(ch.get("end", 0))
            ch["duration"] = end - start
            chapter_jobs.append((model, src, ffmetadata, ch, i))

    chapter_jobs.sort(key=lambda x: x[3]["duration"])

    cpus_real = psutil.cpu_count(logical=False)
    worker_count = int(cpus_real/2)
    cpus_logical = psutil.cpu_count(logical=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for idx, job in enumerate(chapter_jobs):
            cpuset = [(idx * 2) % cpus_logical, (idx * 2 + 1) % cpus_logical]
            # submit job with CPU pinning handled inside process_chapter
            futures.append(executor.submit(process_chapter, job, cpuset))
        for f in concurrent.futures.as_completed(futures):
            print(f.result())

if __name__ == "__main__":
    main()
    print("All done")

