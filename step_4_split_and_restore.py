#!/usr/bin/env python3
# vhs_c_qtgmc_parallel.py
# Parallel VHS processing using QTGMC with cleanup on Ctrl-C

import signal
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
QTGMC_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
ARCHIVE = BASE.parent / "Archive"
OUTPUT = BASE.parent / "Videos"
MAX_PARALLEL = 10
USE_HEVC_AMF_ACCEL = True

# Track all temp files and current final output
CLEAN_UP_ON_ABORT_FILES = set()

def cleanup_temp_files():
    for p in CLEAN_UP_ON_ABORT_FILES:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    CLEAN_UP_ON_ABORT_FILES.clear()

def handle_sigint(signum, frame):
    print("\nCtrl-C detected, cleaning up all temporary and current files...")
    cleanup_temp_files()
    sys.exit(1)

signal.signal(signal.SIGINT, handle_sigint)

def run(cmd, cwd=None):
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)

def parse_chapters(path):
    chapters, cur = [], {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line == "[CHAPTER]":
            if cur: chapters.append(cur)
            cur = {}
        elif "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()
    if cur: chapters.append(cur)
    return chapters

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def process_single_file(src_path):
    src = Path(src_path)
    prefix = "_".join(src.stem.rsplit("_", 2)[:2])
    out_dir = OUTPUT
    out_dir.mkdir(exist_ok=True)

    chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
    if not chapters_file.exists():
        print(f"Skipping {src.name} — no metadata")
        return

    chapters = parse_chapters(chapters_file)
    print(f"Processing: {src.name} ({len(chapters)} chapters)")

    for i, ch in enumerate(chapters, 1):
        title = ch.get("title", f"chapter_{i}")
        start, end, ctime = ch.get("start"), ch.get("end"), ch.get("creation_time", "")
        final = out_dir / f"{safe(title)}.mp4"

        if final.exists():
            print(f"  Skipping {final.name}")
            continue

        temp_raw = out_dir / f"temp_raw_{i:02d}_{safe(title)}.mkv"
        avs_file = out_dir / f"qtgmc_{i:02d}_{safe(title)}.avs"

        # Track all current files for cleanup
        CLEAN_UP_ON_ABORT_FILES = [temp_raw, temp_raw.with_suffix(".mkv.ffindex"), avs_file, final]

        # Extract chapter segment
        run([FFMPEG, "-v", "warning", "-ss", start, "-to", end, "-i", src,
             "-map", "0:v", "-map", "0:a", "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", temp_raw],
            cwd=out_dir)

        # Write AVS script
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
Crop(0,0,-2,-6)
LanczosResize(640,480)
''', encoding="ascii")

        # Prepare final encoding command
        cmd = [FFMPEG, "-v", "warning", "-probesize", "50M", "-i", avs_file, "-i", temp_raw,
               "-map", "0:v", "-map", "1:a", "-map_metadata", "-1",
               "-metadata", f"title={title}", "-metadata", f"creation_time={ctime}",
               "-metadata", f"description=Chapter {title} from VHS tape {src.name}"]

        if USE_HEVC_AMF_ACCEL:
            cmd += [
                "-c:v", "hevc_amf",
                
                # Highest quality mode
                "-usage", "transcoding",              # = "0" — best quality
                "-quality", "quality",                # highest AMF preset
                
                # Best rate control for viewing quality
                "-rc", "vbr_latency",                 # ← VBR Latency = best balance 2025
                "-qvbr_quality_level", "14",          # ← 14 = visually lossless for VHS
                "-header_insertion_spacing", "0",     # smooth playback
                
                # GOP & B-frames — perfect for 59.94 fps
                "-g", "600",                          # 10 sec GOP at 59.94 fps
                "-bf", "3",                           # 3 B-frames = excellent compression
                
                # 10-bit + Apple compatibility
                "-profile:v", "main10",
                "-pix_fmt", "yuv420p10le",
                "-tag:v", "hvc1",                     # Apple QuickTime / iOS
                "-movflags", "+faststart+write_colr",
                "-brand", "mp42",
                
                # Optional: force sane QP bounds (prevents spikes)
                "-min_qp_i", "14", "-max_qp_i", "24",
                "-min_qp_p", "16", "-max_qp_p", "28",
                "-min_qp_b", "18", "-max_qp_b", "30",
                
                # Bonus 2025 tweaks (only if your driver supports)
                "-preanalysis", "1",                  # look-ahead = better motion
                "-enforce_hrd", "1",                  # smoother bitrate
            ]
        else:
            cmd += [
                "-c:v", "libx265",
                "-preset", "slow",
                "-crf", "18",
                "-profile:v", "main10",
                "-pix_fmt", "yuv420p10le"
            ]

        cmd += ["-brand", "mp42", "-c:a", "aac", "-b:a", "48k", "-ac", "1",
                "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=g=15",
                "-movflags", "+faststart+write_colr", "-y", final]

        run(cmd, cwd=out_dir)

        # Clean up finished files
        CLEAN_UP_ON_ABORT_FILES.clear()
        for p in [temp_raw, temp_raw.with_suffix(".mkv.ffindex"), avs_file]:
            Path(p).unlink(missing_ok=True)

    print(f"Finished: {src.name}")

if __name__ == "__main__":
    if not FFMPEG.exists():
        sys.exit(f"ERROR: ffmpeg not found at {FFMPEG}")

    mkv_files = list(ARCHIVE.glob("bennett*.mkv"))
    if not mkv_files:
        sys.exit("No files found")

    print(f"Starting {len(mkv_files)} files — {MAX_PARALLEL} at a time\n")

    try:
        with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            futures = [executor.submit(process_single_file, str(f)) for f in mkv_files]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught, cleaning up all temporary and current files...")
        cleanup_temp_files()
        sys.exit(1)

    print("\nAll done")
