import sys, os, subprocess, shutil
from pathlib import Path
import psutil
import concurrent.futures
import whisper
from whisper.utils import get_writer
import gc

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

def run(cmd, cpuset=None):
    proc = subprocess.Popen([str(c) for c in cmd])
    if cpuset:
        try:
            p = psutil.Process(proc.pid)
            p.cpu_affinity(cpuset)
            for child in p.children(recursive=True):
                child.cpu_affinity(cpuset)

        except Exception as e:
            print(f"Warning: failed to set CPU affinity: {e}", file=sys.stderr)

    retcode = proc.wait()
    if retcode != 0:
        print(f"ERROR: {cmd} = {retcode}")
        raise subprocess.CalledProcessError(retcode, cmd)

def load_whisper_prompt():
    media_dir = BASE / "media"
    comments_lines = []

    if media_dir.exists():
        for subdir in media_dir.iterdir():
            if subdir.is_dir():
                comments_file = subdir / "comments.txt"
                if comments_file.exists():
                    with comments_file.open("r", encoding="utf-8") as f:
                        comments_lines.extend([line.strip() for line in f if line.strip()])

    extra_hints = [
        "Glenda", "Terry", "Terrance", "Bennett", "Terrance J. Bennett", "Tara",
        "Buddy", "Morgan", "Morgie", "Asia", "Hazel", "Poppyfields Dr", "Ponies",
        "Davis", "Uncle Al", "Jacky", "Dory", "Ralph", "Monica", "Gene", "Michael",
        "Anat", "Kim", "Rhett", "Butler", "Lance", "Jan", "Beau Brummell", "Michael",
        "Davis", "Allan", "Peter", "Pasadena", "Altadena", "Christmas carols", "Parties",
        "Jingle Bells", "Rudolph the Red-Nosed Reindeer", "Wedding", "Johnny Appleseed",
        "Christmas Eva", "Christmas Day", "Birthdays", "School Plays", "Easter", "Arizona",
        "Swim & Tennis Club", "Pool", "I Love You", "Football", "Catch", "Santa", "Santa Claws"
    ]

    prompt_str = ", ".join(comments_lines + extra_hints)
    return prompt_str

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def format_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def is_chapter_done(final_file):
    if not final_file.exists():
        return False

    if final_file.stat().st_size < 100_000:
        return False

    return True

def calculate_worker_count(gb_per_worker):
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    mem_based = max(1, int(total_ram_gb // gb_per_worker))
    cpu_based = psutil.cpu_count(logical=False) or 1
    return max(1, min(mem_based, cpu_based))

def parse_chapters(path):
    chapters = []
    ffmetadata = {}
    cur = {}
    in_chapter = False
    seen_chapter = False
    timebase_num = 1
    timebase_den = 1

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

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
            timebase_num = 1
            timebase_den = 1
            continue

        if in_chapter and "=" in line:
            k, v = line.split("=", 1)
            k = k.lower()
            v = v.strip()
            cur[k] = v

            if k == "timebase":
                num, den = v.split("/", 1)
                timebase_num = int(num)
                timebase_den = int(den)

            elif k in ("start", "end"):
                cur[k] = int(v)

                seconds = cur[k] * (timebase_num / timebase_den)
                cur[k + "_seconds"] = round(seconds, 3)

    if cur and in_chapter:
        chapters.append(cur)

    return ffmetadata, chapters

def extract_chapter(src, start, end, dest):
    run([FFMPEG,
        "-nostdin",
        "-v", "warning",
        "-guess_layout_max", "0",
        "-channel_layout", "mono",
        "-i", str(src),
        "-ss", f"{start}", "-to", f"{end}",
        "-pix_fmt", "yuv422p",
        "-color_primaries:v", "6",
        "-color_trc:v", "6",
        "-colorspace:v", "5",
        "-color_range:v", "1",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
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
Tweak(sat=0.8)
'''
    avs_path.write_text(avs_script, encoding="ascii")

def deinterlace(temp_avs, temp_extracted, temp_qtgmc, cpuset=None):
    run([FFMPEG,
         "-nostdin",
         "-v", "error",
         "-guess_layout_max", "0",
         "-channel_layout", "mono",
         "-i", str(temp_avs),
         "-i", str(temp_extracted),
         "-r", "30000 / 1001",
         "-pix_fmt", "yuv422p",
         "-color_primaries:v", "6",
         "-color_trc:v", "6",
         "-colorspace:v", "5",
         "-color_range:v", "1",
         "-threads", "1",
         "-map", "0:v:0", "-c:v", "ffv1",
         "-level", "3", "-g", "1", "-coder", "1", "-context", "1",
         "-slices", "24", "-slicecrc", "1",
         "-map", "0:a", "-c:a", "copy",
         "-y", str(temp_qtgmc)], cpuset)

def extract_audio(temp_extracted, temp_transcript, cpuset=None):
    run([
        FFMPEG, "-nostdin", "-v", "warning",
        "-guess_layout_max", "0",
        "-channel_layout", "mono",
        "-i", str(temp_extracted),
        "-vn",
        "-af", "highpass=f=120,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=13,aresample=16000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-y",
        str(temp_transcript)
    ], cpuset)

def transcribe_audio(temp_transcript, final_vtt):
    try:
        model = whisper.load_model("turbo")
        vtt_writer = get_writer("vtt", str(SUBTITLES))
        prompt_text = load_whisper_prompt()
        result = model.transcribe(str(temp_transcript), prompt=prompt_text, language="en", fp16=False)
        vtt_writer(result, str(final_vtt))
    finally:
        del model
        gc.collect()

def encode_final(temp_qtgmc, final_vtt, final_file, title, ffmetadata, start_hms, end_hms, ctime, location, cpuset=None):
    cmd = [FFMPEG,
           "-nostdin",
           "-v", "error",
           "-guess_layout_max", "0",
           "-channel_layout", "mono",
           "-i", str(temp_qtgmc),
           "-i", str(final_vtt),
           "-map_metadata", "-1",
           "-map_chapters", "-1",
           "-threads", "1",
           "-thread_type", "frame",
           "-c:v", "libx265", "-crf", "18", "-preset", "veryslow",
           "-r", "30000 / 1001",
           "-pix_fmt", "yuv420p10le",
           "-x265-params", "no-open-gop=1:bframes=8:pools=none",
           "-c:a", "aac", "-b:a", "48k", "-ac", "1",
           "-af", "highpass=f=80,lowpass=f=14000,loudnorm=I=-16:TP=-1.5:LRA=11",
           "-tag:v", "hvc1", "-brand", "mp42",
           "-map", "0:v:0", "-map", "0:a:0", "-map", "1:s:0",
           "-c:s", "mov_text",
           "-metadata:s:s:0", "language=eng",
           "-disposition:s:0", "forced",
           "-metadata:s:a:0", "language=eng",
           "-metadata", f"title={title}",
           "-metadata", f"comment=Chapter from archive {ffmetadata.get('title', '')} @ {start_hms}-{end_hms}",
           "-metadata", f"creation_time={ctime}",
           "-metadata", f"date={ctime}",
           "-metadata", f"location={location}",
           "-metadata", f"genre={ffmetadata.get('genre', '')}",
           "-metadata", f"videographer={ffmetadata.get('videographer', '')}",
           "-metadata", f"tape_id={ffmetadata.get('tape_id', '')}",
           "-movflags", "+faststart+write_colr+use_metadata_tags", "-y", str(final_file)]
    run(cmd)

def process_chapter(chapter_job, cpuset):
    p = psutil.Process()
    p.cpu_affinity(cpuset)

    src, ffmetadata, ch, i = chapter_job
    title = ch.get("title", f"Chapter {i+1}")
    start_sec, end_sec = float(ch["start"]), float(ch["end"])
    duration = end_sec - start_sec
    ctime = ch.get("creation_time", "")
    location = ch.get("location", "")

    final_dir = VIDEOS if duration >= 200.0 else CLIPS
    final_file = final_dir / f"{safe(title)}.mp4"

    if is_chapter_done(final_file):
        print(f"Skipped existing: {title}")
        return

    # Create dedicated temp directory inside the output folder
    temp_dir = final_dir / f"{safe(title)}_temp"
    temp_dir.mkdir(exist_ok=True)

    temp_extracted   = temp_dir / "extracted.mkv"
    temp_qtgmc       = temp_dir / "qtgmc.mkv"
    temp_transcript  = temp_dir / "audio.wav"
    temp_avs         = temp_dir / "script.avs"

    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        print(f"Extracting chapter: {title} ({format_hms(start_sec)} - {format_hms(end_sec)})")
        extract_chapter(src, start_sec, end_sec, temp_extracted)

        print(f"Deinterlacing chapter: {title}")
        create_avs(temp_extracted, temp_avs)
        deinterlace(temp_avs, temp_extracted, temp_qtgmc, cpuset)

        print(f"Transcribing chapter: {title}")
        final_vtt = SUBTITLES / f"{safe(title)}.vtt"
        extract_audio(temp_extracted, temp_transcript, cpuset)
        transcribe_audio(temp_transcript, final_vtt)

        print(f"Final encoding chapter: {final_file.name}")
        start_hms = format_hms(start_sec)
        end_hms = format_hms(end_sec)
        encode_final(temp_qtgmc, final_vtt, final_file, title, ffmetadata, start_hms, end_hms, ctime, location, cpuset)

        print(f"Done: {final_file.name}")

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
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
            chapter_jobs.append((src, ffmetadata, ch, i))

    chapter_jobs.sort(key=lambda x: x[2]["duration"])

    cpus_logical = psutil.cpu_count(logical=True)
    worker_count = calculate_worker_count(gb_per_worker=3)
    print(f"Worker count: {worker_count}, CPU count: {cpus_logical}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for idx, job in enumerate(chapter_jobs):
            # 2 neighboring logical CPUs per worker should be on the same core
            cpuset = [(idx * 2) % cpus_logical, (idx * 2 + 1) % cpus_logical]
            futures.append(executor.submit(process_chapter, job, cpuset))
        for f in concurrent.futures.as_completed(futures):
            _ = f.result()

if __name__ == "__main__":
    main()
    print("All done")

