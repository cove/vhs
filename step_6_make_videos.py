#!/usr/bin/env python3.11
#
# Processes archival MKV files by extracting chapters, deinterlacing/applying filters,
# transcribing audio to SRT/VTT, converting SRT to ASS subtitles, and encoding final MP4s
# with embedded metadata and subtitles for access/delivery copies.
#
import shutil, time, re, whisper
from whisper.utils import get_writer
from common import *

def chapter_done(final_file):
    return final_file.exists() and final_file.stat().st_size > 100_000

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
Style: Default,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,1,0,2,10,10,0,1

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

def make_create_avs(temp_extracted: str, avs_filter_path: Path):
    return f'''
LoadPlugin("{QTGMC_DIR}/ffms2.dll") 
LoadPlugin("{QTGMC_DIR}/masktools2.dll") 
LoadPlugin("{QTGMC_DIR}/Rgtools.dll") 
LoadPlugin("{QTGMC_DIR}/mvtools2.dll") 
LoadPlugin("{QTGMC_DIR}/nnedi3.dll") 
LoadPlugin("{QTGMC_DIR}/yadifmod2.dll") 
LoadPlugin("{QTGMC_DIR}/fft3dfilter.dll") 
LoadPlugin("{QTGMC_DIR}/LoadDLL64.dll")
LoadPlugin("{QTGMC_DIR}/SmoothAdjust.dll")
LoadDLL("{QTGMC_DIR}/libfftw3f-3.dll") 
Import("{QTGMC_DIR}/Zs_RF_Shared.avsi") 
Import("{QTGMC_DIR}/QTGMC.avsi") 
FFmpegSource2("{temp_extracted}", atrack=-1) 
{open(avs_filter_path).read()}
'''

def make_extract_audio(temp_extracted, temp_transcript):
    return [FFMPEG_BIN,
        "-nostdin", "-v", "error",
        "-i", str(temp_extracted),
        "-vn",
        "-af", "highpass=f=120,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=13,aresample=16000,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-y", str(temp_transcript)]

def make_extract_chapter(src, start, end, dest):
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(src),
        "-ss", f"{start}", "-to", f"{end}",
        "-force_key_frames", "0",
        "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-y", str(dest)]

def make_encode_final_x265(temp_qtgmc, final_ass, final_file, author, title, archive_tape_title, start_hms, end_hms, creation_time, location):
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_qtgmc),
        "-i", str(final_ass),
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx265", "-crf", "20", "-preset", "slow",
        "-profile:v", "main", "-level", "4.0",
        "-x265-params", "log-level=0",
        "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "1",
        "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-tag:v", "hvc1", "-brand", "mp42",
        "-map", "0:v:0", "-map", "0:a:0", "-map", "1:s:0",
        "-c:s", "mov_text",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "forced",
        "-metadata", f"title={title}",
        "-metadata", f"comment=Filmed by {author} on {creation_time} at {location}, original tape {archive_tape_title} @ {start_hms}-{end_hms} ",
        "-metadata", f"creation_time={creation_time}",
        "-metadata", f"location={location}",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart+use_metadata_tags", "-y", str(final_file)]

def make_encode_final_x264(temp_qtgmc, final_ass, final_file, author, title, archive_tape_title, start_hms, end_hms, creation_time, location):
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_qtgmc),
        "-i", str(final_ass),
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-level", "4.0", "-tune", "grain",
        "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "1",
        "-af", "highpass=f=80,lowpass=f=14000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-map", "0:v:0", "-map", "0:a:0", "-map", "1:s:0",
        "-c:s", "mov_text",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:s:0", "language=eng",
        "-disposition:s:0", "forced",
        "-metadata", f"title={title}",
        "-metadata", f"comment=Filmed by {author} on {creation_time} at {location}, original tape {archive_tape_title} @ {start_hms}-{end_hms} ",
        "-metadata", f"creation_time={creation_time}",
        "-metadata", f"location={location}",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart+use_metadata_tags",
        "-y", str(final_file)]

def make_deinterlace(temp_avs, temp_extracted, temp_qtgmc):
    return [FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(temp_avs),
        "-i", str(temp_extracted),
        "-pix_fmt", "yuv422p",
        "-map", "0:v:0", "-c:v", "ffv1",
        "-level", "3", "-coder", "1", "-context", "1",
        "-map", "0:a", "-c:a", "copy",
        "-fflags", "+genpts", "-start_at_zero", "-avoid_negative_ts", "make_zero",
        "-y", str(temp_qtgmc)]

def transcribe_audio(model, temp_transcript, final_srt, final_vtt, final_dir):
    prompt_text = (
        "Glenda and Terry were talking with Uncle Al and Tara. "
        "Buddy and Morgan were playing in the pool at the Swim and Tennis Club. "
        "Asia and Hazel live near Poppyfields Drive in Altadena. "
        "Uncle Al, Jacky, Dory, Ralph, Monica, Gene, and Michael were singing "
        "Christmas carols like Jingle Bells and Rudolph the Red-Nosed Reindeer. "
        "Lance, Jan, Rhett, Butler, and talked about birthdays, school plays, "
        "weddings, and Christmas Eve. Ralph knows that Beau Brummell was a fashion icon."
    )
    srt_writer = get_writer("srt", final_dir)
    vtt_writer = get_writer("vtt", final_dir)
    result = model.transcribe(str(temp_transcript), word_timestamps=True, language="en", fp16=False, prompt=prompt_text)
    srt_writer(result, str(final_srt))
    vtt_writer(result, str(final_vtt))

def main():
    model = whisper.load_model("turbo", download_root=WHISPER_MODEL_DIR)

    for src in ARCHIVE_DIR.glob("*.mkv"):
        archive_name = src.stem
        chapters_file = METADATA_DIR / archive_name / "chapters.ffmetadata"
        if not chapters_file.exists():
            print(f"Skipping {src.name}: no metadata found {chapters_file}")
            continue

        ffm, chapters = parse_chapters(chapters_file)
        if not chapters:
            print(f"No chapters for {src.name}")
            continue

        for ch in chapters:
            ch["duration"] = float(ch.get("end", 0)) - float(ch.get("start", 0))
        chapters.sort(key=lambda x: x["duration"])

        cur_count = 1
        total_chapters = len(chapters)

        for i, ch in enumerate(chapters):
            title = ch.get("title")
            start_sec = ch["start"]
            end_sec = ch["end"]

            final_dir = VIDEOS_DIR if ch["duration"] >= 200 else CLIPS_DIR
            final_file = final_dir / f"{safe(title)}.mp4"
            final_srt = final_dir / f"{safe(title)}.srt"
            final_vtt = final_dir / f"{safe(title)}.vtt"
            final_ass = final_dir / f"{safe(title)}.ass"

            if chapter_done(final_file):
                print(f"Skipping existing chapter: {title}")
                cur_count += 1
                continue

            # inline temp path creation
            temp_dir = final_dir / f"{safe(title)}_temp"
            temp_dir.mkdir(exist_ok=True)
            extracted = temp_dir / "extracted.mkv"
            qtgmc = temp_dir / "qtgmc.mkv"
            audio = temp_dir / "audio.wav"
            avs = temp_dir / "script.avs"
            filter_script = METADATA_DIR / archive_name / "filter.avs"
            chapter_filter_script = METADATA_DIR / archive_name / f"{title}.avs"
            if chapter_filter_script.exists():
                filter_script = chapter_filter_script

            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            date = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            progress = f"({cur_count} of {total_chapters} chapters) [{date}]"
            print(f"Processing: {src.name} {progress}")

            try:
                print(f"Extracting chapter...")
                run(make_extract_chapter(src, start_sec, end_sec, extracted))

                print(f"Applying video filters...")
                if sys.platform == "win32":
                    if filter_script.exists():
                        script = make_create_avs(extracted, filter_script)
                        avs.write_text(script, encoding="ascii")
                        run(make_deinterlace(avs, extracted, qtgmc))
                    else:
                        print("Skipping since there's no filter script for this archive...")
                        shutil.copy(extracted, qtgmc)
                elif os.environ.get("TEST_ENV"):
                    print("Skipping deinterlacing for test run...")
                    shutil.copy(extracted, qtgmc)
                else:
                    raise RuntimeError("Unsupported platform for QTGMC: " + sys.platform)

                print(f"Transcribing audio...")
                run(make_extract_audio(extracted, audio))
                transcribe_audio(model, audio, final_srt, final_vtt, final_dir)
                srt_to_ass(final_srt, final_ass)

                print(f"Final encoding...")
                author = ch.get("author", ffm.get("author"))
                archive_tape_title = ffm.get("title")
                start_hms = format_hms(start_sec)
                end_hms = format_hms(end_sec)
                ctime = ch.get("creation_time")
                location = ch.get("location")

                cmd = make_encode_final_x264(qtgmc, final_ass, final_file, author, title, archive_tape_title, start_hms, end_hms, ctime, location)
                run(cmd)

            finally:
                os.chdir(original_cwd)
                shutil.rmtree(temp_dir, ignore_errors=True)

            cur_count += 1

        print(f"All done")

if __name__ == "__main__":
    main()

