import subprocess, sys, os, re
from pathlib import Path
import whisper
from whisper.utils import get_writer

# --- Paths / Environment ---
BASE = Path(__file__).parent.resolve()
FFMPEG_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
FFMPEG = FFMPEG_DIR / "ffmpeg.exe"
QTGMC_DIR = FFMPEG_DIR

ARCHIVE = BASE.parent / "Archive"
VIDEOS = BASE.parent / "Videos"
VIDEOS.mkdir(exist_ok=True)
CLIPS = BASE.parent / "Clips"
CLIPS.mkdir(exist_ok=True)

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

def srt_to_ass(srt_path, ass_path, font="Calibri", fontsize=40):
    srt_path = Path(srt_path)
    ass_path = Path(ass_path)
    ass_header = f"""[Script Info]
Title: Converted from {srt_path.name}
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    content = srt_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\d+\n|\Z)", re.S)
    for idx, start, end, text in pattern.findall(content):
        text = text.strip().replace("\n", r"\N")
        start_parts = start.split(":")
        end_parts = end.split(":")
        start_ass = f"{int(start_parts[0])}:{int(start_parts[1]):02d}:{int(start_parts[2].split(',')[0]):02d}.{int(start_parts[2].split(',')[1])//10:02d}"
        end_ass = f"{int(end_parts[0])}:{int(end_parts[1]):02d}:{int(end_parts[2].split(',')[0]):02d}.{int(end_parts[2].split(',')[1])//10:02d}"
        lines.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")
    ass_path.write_text(ass_header + "\n".join(lines), encoding="utf-8")

# --- Load Whisper model ---
model = whisper.load_model("turbo")
srt_writer = get_writer("srt", str(CLIPS))

# --- Main ---
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

        final_dir = VIDEOS
        if duration < 200:
            final_dir = CLIPS

        archive_file = final_dir / f"{safe(title)}_archive.mkv"

        # --- Extract chapter ---
        temp_raw = final_dir / f"{safe(title)}_temp_raw.mkv"
        print(f"Extracting chapter: {title}")
        run([FFMPEG, "-v", "warning", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
             "-i", str(src), "-map", "0:v", "-map", "0:a", "-c", "copy",
             "-avoid_negative_ts", "make_zero", "-y", str(temp_raw)])

        avs_file = final_dir / f"{safe(title)}_temp.avs"
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
QTGMC(Preset="Very Slow",FPSDivisor=2,EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3) 
Crop(0,0,-2,-6) 
LanczosResize(640,480) 
Prefetch()'''
        avs_file.write_text(avs_script, encoding="ascii")

        # --- QTGMC → FFV1 (archive) ---
        print(f"Applying deinterlacing to chapter: {title}")
        temp_qtgmc = final_dir / f"{safe(title)}_temp_qtgmc.mkv"
        run([FFMPEG, "-v", "warning", "-i", str(avs_file), "-i", str(temp_raw),
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

        # --- Whisper transcription ---
        final_file = final_dir / f"{safe(title)}.mp4"
        print(f"Transcribing audio: {title}")
        result = model.transcribe(str(temp_qtgmc), language="en", fp16=False)
        temp_srt = final_dir / f"{final_dir.stem}_subtitles.srt"
        temp_ass = final_dir / f"{final_dir.stem}_subtitles.ass"
        srt_writer(result, str(temp_srt))
        srt_to_ass(temp_srt, temp_ass)

        # --- Encode MP4 with subtitles burned in ---
        print(f"Encoding subtitles and final metadata: {final_file.name}")
        cmd = [
            FFMPEG, "-v", "warning",
            "-i", str(temp_qtgmc),
            "-metadata", f"title={title}",
            "-metadata", f"comment=Chapter from file {src.name} ({ffmetadata.get('uuid', '')}) @ {start_hms}-{end_hms} )",
            "-metadata", f"creation_time={ctime}",
            "-metadata", f"com.apple.quicktime.creationdate={ctime}",
            "-metadata", f"com.apple.quicktime.uuid={uuid}",
            "-metadata", f"date={ctime}",
            "-metadata", f"genre={ffmetadata.get('genre', '')}",
            "-metadata", f"videographer={ffmetadata.get('videographer', '')}",
            "-metadata", f"tape_id={ffmetadata.get('tape_id', '')}",
        ]

        if location:
            iso6709 = location.rstrip("/") + "/"
            cmd += [
                "-metadata", f"com.apple.quicktime.location.ISO6709={iso6709}"
            ]

        cmd += [
                "-map", "0:v", "-map_metadata", "-1",
                "-vf", f"ass={temp_ass}",
                "-c:v", "libx265",
                "-preset", "veryslow",
                "-crf", "16",
                "-profile:v", "main10",
                "-pix_fmt", "yuv420p10le",
                "-x265-params",
                "merange=57:psy-rd=2.0:aq-mode=3:aq-strength=1.0:bframes=8:keyint=600:rc-lookahead=80:no-sao=0:no-strong-intra-smoothing=0",
                "-x265-params", "deblock=-1:-1",
                "-x265-params", "ref=6",
                "-tag:v", "hvc1",
                "-movflags", "+faststart+write_colr+use_metadata_tags",
                "-brand", "mp42",
                "-c:a", "aac", "-b:a", "48k", "-ac", "1",
                "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=g=15",
                "-y", str(final_file)]
        run(cmd)

        temp_raw.unlink(missing_ok=True)
        temp_srt.unlink(missing_ok=True)
        temp_ass.unlink(missing_ok=True)

        print(f"  Done → {final_file.name}")

print("All done")
