import subprocess, sys, os
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

from pathlib import Path

def srt_to_ass(srt_path, ass_path, font="Calibri", fontsize=14):
    """
    Convert an SRT file to a styled ASS subtitle file.

    Parameters:
        srt_path: Path to input SRT file
        ass_path: Path to output ASS file
        font: font name
        fontsize: font size
    """
    srt_path = Path(srt_path)
    ass_path = Path(ass_path)

    # ASS header
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
    with srt_path.open("r", encoding="utf-8") as f:
        content = f.read()

    import re
    pattern = re.compile(r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\d+\n|\Z)", re.S)
    matches = pattern.findall(content)

    for idx, start, end, text in matches:
        # replace newlines in SRT with \N for ASS
        text = text.strip().replace("\n", r"\N")
        # convert SRT time to ASS time (replace , with .)
        start_ass = start.replace(",", ".")[:-1]  # remove last digit of ms to fit ASS format
        end_ass = end.replace(",", ".")[:-1]
        # ASS time format is H:MM:SS.cs (centiseconds)
        start_parts = start.split(":")
        end_parts = end.split(":")
        start_ass = f"{int(start_parts[0])}:{int(start_parts[1]):02d}:{int(start_parts[2].split(',')[0]):02d}.{int(start_parts[2].split(',')[1])//10:02d}"
        end_ass = f"{int(end_parts[0])}:{int(end_parts[1]):02d}:{int(end_parts[2].split(',')[0]):02d}.{int(end_parts[2].split(',')[1])//10:02d}"

        lines.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")

    ass_path.write_text(ass_header + "\n".join(lines), encoding="utf-8")


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

# --- Load Whisper once ---
model = whisper.load_model("turbo")
srt_writer = get_writer("srt", str(CLIPS))

# --- Main processing ---
for src in ARCHIVE.glob("*.mkv"):
    name = src.stem
    prefix = "_".join(name.rsplit("_", 2)[:2])
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

    print(f"Processing: {src.name} ({len(chapters)} chapters, shortest first)")

    for i, ch in enumerate(chapters):
        title = ch.get("title", f"Chapter {i+1}")
        start_sec = int(ch["start"])
        end_sec = int(ch["end"])
        ctime = ch.get("creation_time", "")
        date = ctime[:4]
        location = ch.get("location", "")
        filter_avs = ch.get("filter_avs", "")
        uuid = ch.get("uuid", "")
        duration = ch.get("duration")

        final_dir = VIDEOS if duration >= 200 else CLIPS
        final = final_dir / f"{safe(title)}.mp4"
        if final.exists() and final.stat().st_size < 100_000:
            print(f"  Skipping {final.name}")
            continue

        temp_raw = final_dir / f"temp_raw_{i+1:02d}.mkv"
        avs_file = final_dir / f"qtgmc_{i+1:02d}.avs"

        # --- Extract chapter ---
        print(f"Extracting: {final.name}")
        run([
            FFMPEG, "-v", "warning",
            "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
            "-i", str(src),
            "-map", "0:v", "-map", "0:a",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-y", str(temp_raw)
        ])

        # --- Whisper transcription ---
        print(f"Transcribing: {final.name}")
        result = model.transcribe(str(temp_raw), language="en", fp16=False)

        temp_srt = final_dir / f"{final.stem}_subtitles.srt"
        temp_ass = final_dir / f"{final.stem}_subtitles.ass"

        srt_writer(result, str(temp_srt))

        start_hms = format_hms(start_sec)
        end_hms = format_hms(end_sec)

        print(f"Restoring: {final.name}")

        srt_to_ass(temp_srt, temp_ass)
        avs_script = f'''
SetFilterMTMode("DEFAULT_MT_MODE", 2)
LoadPlugin("{QTGMC_DIR}/VSFilterMod.dll")
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
FFmpegSource2("{temp_raw.name}", atrack=-1)
AssumeFPS(30000,1001)
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",FPSDivisor=2,EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3)
Crop(0,0,-2,-6)
LanczosResize(640,480)
{filter_avs}
VSFilter("{temp_ass}")
Prefetch()
'''
        avs_file.write_text(avs_script, encoding="ascii")

        # --- QTGMC encode ---
        temp_qtgmc = final_dir / f"{final.stem}_qtgmc_temp.mp4"
        cmd = [
            FFMPEG, "-v", "warning",
            "-i", str(avs_file), "-i", str(temp_raw),
            "-map", "0:v", "-map", "1:a", "-map_metadata", "-1",
            "-metadata", f"title={title}",
            "-metadata", f"comment=Chapter from file {src.name} ({ffmetadata.get('uuid', '')}) @ {start_hms}-{end_hms} )",
            "-metadata", f"creation_time={ctime}",
            "-metadata", f"com.apple.quicktime.creationdate={ctime}",
            "-metadata", f"com.apple.quicktime.uuid={uuid}",
            "-metadata", f"date={date}",
            "-metadata", f"genre={ffmetadata.get('genre', '')}",
            "-metadata", f"videographer={ffmetadata.get('videographer', '')}",
            "-metadata", f"tape_id={ffmetadata.get('tape_id', '')}",
        ]

        if location:
            iso6709 = location.rstrip("/") + "/"
            cmd += ["-metadata", f"com.apple.quicktime.location.ISO6709={iso6709}"]
        cmd += [
            "-c:v", "libx265",
            "-preset", "veryslow",
            "-crf", "16",
            "-profile:v", "main10",
            "-pix_fmt", "yuv420p10le",
            "-x265-params", "merange=57:psy-rd=2.0:aq-mode=3:aq-strength=1.0:bframes=8:keyint=600:rc-lookahead=80:no-sao=0:no-strong-intra-smoothing=0",
            "-x265-params", "deblock=-1:-1",
            "-x265-params", "ref=6",
            "-tag:v", "hvc1",
            "-movflags", "+faststart+write_colr+use_metadata_tags",
            "-brand", "mp42",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1",
            "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=g=15",
            "-y", str(temp_qtgmc)
        ]
        run(cmd, cwd=final_dir)
        temp_raw.unlink(missing_ok=True)
        avs_file.unlink(missing_ok=True)

        temp_qtgmc.unlink(missing_ok=True)
        temp_srt.unlink(missing_ok=True)

        print(f"  Done → {final.name}\n")

    print(f"Finished: {src.name}\n")

print("All done")
