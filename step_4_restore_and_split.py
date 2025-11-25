#!/usr/bin/env python3

import sys
import subprocess
import tempfile
import configparser
from pathlib import Path

FFMPEG = "software/FFmpeg-QTGMC Easy 2025.01.11/ffmpeg.exe"
QTGMC_DIR = "software/FFmpeg-QTGMC Easy 2025.01.11"

for src_path in sys.argv[1:]:
    src = Path(src_path).resolve()
    name = src.stem
    out_dir = src.parent / f"{name}_chapters"
    out_dir.mkdir(exist_ok=True)

    prefix = ''.join(c for c in name if c.isdigit() or c.isalpha())[:10]
    meta_dir = Path(__file__).parent / "media_metadata" / prefix

    chapters_file = meta_dir / "chapstes.ffmetadata"
    cfg = configparser.ConfigParser(delimiters='=', comment_prefixes=';', interpolation=None)
    cfg.optionxform = str
    cfg.read(str(chapters_file), encoding="utf-8")

    chapters = []
    for section in cfg.sections():
        if section == "FFMETADATA1": continue
        title = cfg[section].get("title", "Untitled").strip()
        ctime = cfg[section].get("creation_time", "").strip() or None
        start = float(cfg[section].get("START", 0))
        end = float(cfg[section].get("END", 999999))
        chapters.append((title, ctime, start, end))

    for i, (title, ctime, start, end) in enumerate(chapters):
        num = f"{i+1:02d}"
        safe_title = ''.join(c if c not in r'<>:"/\|?*' else '-' for c in title)
        final = out_dir / f"{num} - {safe_title}.mp4"
        temp_raw = Path(tempfile.gettempdir()) / f"temp_{num}.mkv"

        subprocess.run([
            FFMPEG, "-v", "error",
            "-ss", str(start), "-to", str(end),
            "-i", str(src),
            "-map", "0:v", "-map", "0:a?",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-y", str(temp_raw)
        ], check=True)

        avs = f'''
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
ConvertToYV12(matrix="Rec601")
QTGMC(preset="Faster")
Crop(0,0,-2,-6)
LanczosResize(640,480)
Return Last
'''
        avs_file = Path(tempfile.gettempdir()) / f"qtgmc_{num}.avs"
        avs_file.write_text(avs, encoding="ascii")

        subprocess.run([
            FFMPEG,
            "-i", str(avs_file), "-i", str(temp_raw),
            "-map", "0:v", "-map", "1:a?",
            "-metadata", f"title={title}",
            "-metadata", f"creation_time={ctime or ''}",
            "-c:v", "libx265", "-preset", "slow", "-crf", "18",
            "-x265-params", "profile=main10",
            "-tag:v", "hvc1",  # Apple QuickTime compatibility
            "-c:a", "aac", "-b:a", "48k",
            "-af", "highpass=f=80,lowpass=f=14000,acompressor",
            "-movflags", "+faststart",
            "-y", str(final)
        ], check=True)

        temp_raw.unlink(missing_ok=True)
        avs_file.unlink(missing_ok=True)

    print(f"Done: {out_dir}")

print("All finished.")
