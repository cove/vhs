from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess, sys
import psutil

BASE = Path(__file__).parent.resolve()
FFMPEG = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11" / "ffmpeg.exe"
QTGMC_DIR = BASE / "software" / "FFmpeg-QTGMC Easy 2025.01.11"
ARCHIVE = BASE.parent / "Archive"
OUTPUT = BASE.parent / "Videos"
MAX_PARALLEL = 8
USE_HEVC_AMF_ACCEL = True
PHYSICAL_CORES = list(range(0, psutil.cpu_count(logical=False), 2))

def run(cmd, cwd=None):
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

def parse_chapters(path):
    chapters, cur = [], {}
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

def safe(s): return s.translate(str.maketrans(r'<>:"/\|?*', "________"))

def encode_and_remux(job):
    job_id, src_path, prefix, chap_idx, chap = job
    core = PHYSICAL_CORES[job_id % len(PHYSICAL_CORES)]
    psutil.Process().cpu_affinity([core])
    src = Path(src_path)
    out_dir = OUTPUT
    out_dir.mkdir(exist_ok=True)
    title = chap.get("title", f"Chapter {chap_idx}")
    start, end = chap["start"], chap["end"]
    ctime = chap.get("creation_time", "")
    location = chap.get("location")
    final = out_dir / f"{safe(title)}.mp4"
    tmp_raw = out_dir / f"temp_raw_c{chap_idx:02d}_j{job_id}.mkv"
    avs = out_dir / f"qtgmc_c{chap_idx:02d}_j{job_id}.avs"

    if not final.exists():
        # extract chapter
        run([FFMPEG, "-v", "error", "-ss", start, "-to", end, "-i", src,
             "-map", "0:v", "-map", "0:a", "-c", "copy",
             "-avoid_negative_ts", "make_zero", "-y", tmp_raw], cwd=out_dir)

        # create QTGMC script
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
FFmpegSource2("{tmp_raw}", atrack=-1)
AssumeFPS(30000,1001)
ConvertToYV12(matrix="Rec601")
QTGMC(Preset="Very Slow",EZKeepGrain=1.0,Sharpness=1.2,SourceMatch=3,Lossless=2,TR2=3)
Crop(0,0,-2,-6)
LanczosResize(640,480)
''', encoding="ascii")

        # encode and apply metadata
        cmd = [FFMPEG, "-v", "warning", "-i", avs, "-i", tmp_raw,
               "-map", "0:v", "-map", "1:a", "-map_metadata", "-1",
               "-metadata", f"title={title}",
               "-metadata", f"creation_time={ctime}",
               "-metadata", f"description=Restored from VHS-C tape: {src.name}"]
        if location:
            cmd += ["-metadata", f"com.apple.quicktime.location.ISO6709={location}",
                    "-metadata", f"location={location}"]

        if USE_HEVC_AMF_ACCEL:
            cmd += ["-c:v", "hevc_amf", "-usage", "transcoding", "-quality", "quality",
                    "-rc", "vbr_latency", "-qvbr_quality_level", "14", "-g", "600", "-bf", "3",
                    "-profile:v", "main10", "-pix_fmt", "yuv420"]
        else:
            cmd += ["-c:v", "libx265", "-preset", "slow", "-crf", "18",
                    "-profile:v", "main10", "-pix_fmt", "yuv420p10le"]

        cmd += ["-tag:v", "hvc1", "-movflags", "+faststart+write_colr", "-brand", "mp42"]
        cmd += ["-c:a", "aac", "-b:a", "48k", "-ac", "1",
                "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-28,dynaudnorm=g=15",
                "-y", final]
        run(cmd, cwd=out_dir)

        # cleanup temp files
        for p in (tmp_raw, tmp_raw.with_suffix(".mkv.ffindex"), avs):
            p.unlink(missing_ok=True)

    return f"Job {job_id} done"

def build_jobs(files_glob):
    files = list(ARCHIVE.glob(files_glob))
    jobs=[]; jid=0
    for src in files:
        name=src.stem; prefix="_".join(name.rsplit("_",2)[:2])
        chfile = BASE/"media_metadata"/prefix/"chapters.ffmetadata"
        if not chfile.exists(): continue
        for idx, ch in enumerate(parse_chapters(chfile),1):
            jobs.append((jid, str(src), prefix, idx, ch)); jid+=1
    return jobs

def main():
    if not FFMPEG.exists(): sys.exit(f"ffmpeg not found: {FFMPEG}")
    jobs = build_jobs("*.mkv")
    if not jobs: sys.exit("No chapters found")
    print(f"Chapters: {len(jobs)} — running {MAX_PARALLEL} parallel")
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = [ex.submit(encode_and_remux, job) for job in jobs]
        for f in as_completed(futures):
            try:
                print(f.result())
            except Exception as e:
                print("ERROR:", e)
    print("All done.")

if __name__ == "__main__":
    main()


