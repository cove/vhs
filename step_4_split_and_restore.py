#!/usr/bin/env python3
# vhs_c_qtgmc_parallel.py
# Processes all tapes in ../Archive/ — you control how many run at once

from random import random
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
from random import uniform

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
QTGMC_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
ARCHIVE = BASE.parent / "Archive"
OUTPUT = BASE.parent / "Videos"
MAX_PARALLEL = 8
HARDWARE_ACCEL = True

def random_delay():
    delay = uniform(5, 30)   # 5–30 seconds — perfect spread
    print(f"   → Stagger delay: {delay:.1f}s")
    sleep(delay)
    
def run(cmd, cwd=None):
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)

def parse_chapters(path):
    chapters, cur = [], {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line == "[CHAPTER]":
            if cur:
                chapters.append(cur)
            cur = {}
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()
    if cur:
        chapters.append(cur)
    return chapters

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def process_single_file(src_path):
    src = Path(src_path)
    name = src.stem
    prefix = "_".join(name.rsplit("_", 2)[:2])
    out_dir = OUTPUT
    out_dir.mkdir(exist_ok=True)

    chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
    if not chapters_file.exists():
        print(f"Skipping {src.name} — no metadata")
        return

    random_delay()
    
    chapters = parse_chapters(chapters_file)
    print(f"Processing: {src.name} ({len(chapters)} chapters)")

    for i, ch in enumerate(chapters, 1):
        title = ch.get("title", f"chapter_{i}")
        start, end = ch.get("start"), ch.get("end")
        ctime = ch.get("creation_time", "")

        final = out_dir / f"{safe(title)}.mp4"
        if final.exists():
            print(f"  Skipping {final.name}")
            continue

        temp_raw = out_dir / f"temp_raw_{i:02d}_{safe(title)}.mkv"
        avs_file = out_dir / f"qtgmc_{i:02d}_{safe(title)}.avs"

        run([FFMPEG, "-v", "error",
             "-ss", start, "-to", end,
             "-i", src,
             "-map", "0:v", "-map", "0:a",
             "-c", "copy", "-avoid_negative_ts", "make_zero",
             "-y", temp_raw], cwd=out_dir)

        avs_file.write_text(f'''
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
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3)
Levels(16, 1.10, 235, 0, 255, coring=false)
ColorYUV(off_u=-6, off_v=+2)
MergeChroma(Blur(0.8))
Tweak(sat=1.25, bright=2)
Crop(0,0,-2,-6)
LanczosResize(640,480)
''', encoding="ascii")
#           "-c:v", "libx265", 

        cmd = [
            FFMPEG, "-v", "error", "-i", avs_file, "-i", temp_raw,
            "-map", "0:v", "-map", "1:a?",
            "-map_metadata", "-1",
            "-metadata", f"title={title}",
            "-metadata", f"creation_time={ctime}",
            "-metadata", f"com.apple.quicktime.creationdate={ctime}"]

        if HARDWARE_ACCEL == True:
           cmd += ["-c:v", "h264_amf",
            "-quality", "quality",
            "-usage", "1",
            "-rc", "2",
            "-qp_i", "20", "-qp_p", "22", "-qp_b", "24"]
        else:
            cmd += ["-c:v", "libx265", "-preset", "medium", "-crf", "18",
            "-profile:v", "main10", "-pix_fmt", "yuv420p10le"]

        cmd += [
            "-tag:v", "hvc1", "-brand", "mp42",
            "-c:a", "aac", "-b:a", "48k", "-ac", "1",
            "-af", "highpass=f=80,lowpass=f=14000,acompressor",
            "-movflags", "+faststart+write_colr",
            "-y", final]
        run(cmd, cwd=out_dir)

        for p in [temp_raw, temp_raw.with_suffix(".mkv.ffindex"), avs_file]:
            p.unlink(missing_ok=True)

    print(f"Finished: {src.name}")

if __name__ == "__main__":
    if not FFMPEG.exists():
        print(f"ERROR: ffmpeg not found at {FFMPEG}")
        sys.exit(1)

    mkv_files = list(ARCHIVE.glob("bennett*.mkv"))
    if not mkv_files:
        print("No files found")
        sys.exit(0)

    print(f"Starting {len(mkv_files)} files — {MAX_PARALLEL} at a time\n")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [executor.submit(process_single_file, str(f)) for f in mkv_files]
        for future in as_completed(futures):
            future.result()  # raise exception if any

    print("\nAll done")
