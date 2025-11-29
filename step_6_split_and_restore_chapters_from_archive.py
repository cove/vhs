#!/usr/bin/env python3
# vhs_c_csv_split_mkv_to_mp4.py
# Split .mkv → chapter MKVs → QTGMC → final .mp4

import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
QTGMC_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
ARCHIVE = BASE.parent / "Archive"
VIDEOS = BASE.parent / "Videos"
VIDEOS.mkdir(exist_ok=True)

THREADS = 8

if not FFMPEG.exists():
    print(f"ERROR: ffmpeg.exe not found at {FFMPEG}")
    sys.exit(1)

def run(cmd, cwd=None):
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def parse_chapters(path):
    chapters = []
    cur = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "[CHAPTER]":
            if cur: chapters.append(cur)
            cur = {}
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            cur[k.lower()] = v.strip()
    if cur: chapters.append(cur)
    return chapters

def main():
    for src in ARCHIVE.glob("*.mkv"):
        name = src.stem
        prefix = "_".join(name.rsplit("_", 2)[:2])
        chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name} — no metadata")
            continue

        chapters = parse_chapters(chapters_file)
        if not chapters:
            print(f"No chapters for {src.name}")
            continue

        print(f"Processing: {src.name} ({len(chapters)} chapters)")

        # Build CSV segment list
        csv_lines = []
        for ch in chapters:
            title = ch.get("title", "Untitled")
            safe_title = safe(title)
            start_ms = int(ch["start"])
            end_ms = int(ch["end"])
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000
            # Intermediate = MKV, final = MP4
            csv_lines.append(f"{VIDEOS}/{safe_title}.mkv,{start_sec:.3f},{end_sec:.3f}")

        csv_file = VIDEOS / f"{src.stem}_segments.csv"
        csv_file.write_text("\n".join(csv_lines), encoding="utf-8")

        # One FFmpeg command — splits into chapter MKVs
        run([
            FFMPEG, "-v", "error",
            "-i", str(src),
            "-map", "0", "-c", "copy",
            "-f", "segment",
            "-segment_list", str(csv_file),
            "-segment_list_type", "csv",
            "-reset_timestamps", "1",
            "-y", "-segment_list_entry_prefix" f"{VIDEOS}/%s"
        ])

        # Process each chapter MKV → final MP4
        for chapter_mkv in VIDEOS.glob("*.mkv"):
            if not chapter_mkv.name.endswith(".mkv"):
                continue
            title = chapter_mkv.stem
            final = VIDEOS / f"{safe(title)}.mp4"

            if final.exists():
                print(f"  Skipping {final.name}")
                chapter_mkv.unlink(missing_ok=True)
                continue

            print(f"  Encoding: {title}")

            avs = VIDEOS / "qtgmc.avs"
            avs.write_text(f'''
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
FFmpegSource2("{chapter_mkv.name}", atrack=-1)
AssumeFPS(30000,1001)
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3)
Levels(16, 1.10, 235, 0, 255, coring=false)
ColorYUV(off_u=-6, off_v=+2)
MergeChroma(Blur(0.8))
Tweak(sat=1.25, bright=2)
Crop(0,0,-2,-6)
LanczosResize(640,480)
Prefetch()
''', encoding="ascii")

            run([
                FFMPEG, "-i", avs, "-i", chapter_mkv,
                "-map", "0:v", "-map", "1:a?", "-map_metadata", "-1",
                "-metadata", f"title={title}",
                "-c:v", "libx265", "-preset", "fast", "-crf", "18",
                "-profile:v", "main10", "-pix_fmt", "yuv420p10le",
                "-tag:v", "hvc1", "-movflags", "+faststart+write_colr", "-brand", "mp42",
                "-c:a", "aac", "-b:a", "48k",
                "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=g=15",
                "-threads", str(THREADS),
                "-y", final
            ], cwd=VIDEOS)

            chapter_mkv.unlink(missing_ok=True)
            avs.unlink(missing_ok=True)

        csv_file.unlink(missing_ok=True)
        print(f"Finished: {src.name}\n")

    print("All done — perfect chapters in ../Videos")

if __name__ == "__main__":
    main()
