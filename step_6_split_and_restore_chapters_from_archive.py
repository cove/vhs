import sys
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
MKVMERGE = BASE / "bin" / "mkvmerge.exe"
QTGMC_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
ARCHIVE = BASE.parent / "Archive"
OUTPUT = BASE.parent  # final files go up one level
THREADS = 8

if not MKVMERGE.exists():
    print(f"ERROR: mkvmerge.exe not found at {MKVMERGE}")
    sys.exit(1)

def run(cmd, cwd=None):
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

def safe(s):
    return s.translate(str.maketrans(r'<>:"/\|?*', "_________"))

def main():
    for src in ARCHIVE.glob("*.mkv"):
        name = src.stem
        prefix = "_".join(name.rsplit("_", 2)[:2])
        chapters_file = BASE / "media_metadata" / prefix / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name} — no metadata")
            continue

        chapter_dir = src.parent / f"{name}_chapters"
        chapter_dir.mkdir(exist_ok=True)

        print(f"Splitting: {src.name}")

        # Split with mkvmerge — 100% perfect chapter titles & timing
        run([
            MKVMERGE, "-o", str(chapter_dir / "%title%.mkv"),
            "--split", "chapters:all",
            str(src)
        ], cwd=chapter_dir)

        # Process each chapter
        for chapter_mkv in chapter_dir.glob("*.mkv"):
            title = chapter_mkv.stem
            final = OUTPUT / f"{safe(title)}.mp4"

            if final.exists():
                print(f"  Skipping {final.name}")
                chapter_mkv.unlink(missing_ok=True)
                continue

            print(f"  Processing: {title}")

            avs = chapter_dir / "qtgmc.avs"
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
            ], cwd=chapter_dir)

            chapter_mkv.unlink(missing_ok=True)
            avs.unlink(missing_ok=True)

        # Move MP4s up one level and delete chapter folder
        for mp4 in chapter_dir.glob("*.mp4"):
            mp4.replace(OUTPUT / mp4.name)
        chapter_dir.rmdir()

        print(f"Finished: {src.name}\n")

    print("All done — perfect chapters in ../")

if __name__ == "__main__":
    main()
