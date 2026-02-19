import shutil
import os
import sys
import subprocess
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TEST_ENV"] = "1"
from common import *
TESTDATA_DIR = BASE / "test" / "test_data"
os.environ["PYTHONPATH"] = str(BASE)

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

def _framemd5_hashes(path: Path):
    hashes = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            hashes.append(parts[5])
    return hashes

def _encode_frame_id_token(frame_id: int):
    fid = int(frame_id) & 0xFFFF
    chk = (fid ^ (fid >> 8) ^ 0x5A) & 0xFF
    return ((fid << 8) | chk) & 0xFFFFFF

def _draw_frame_id_overlay(frame, frame_id, x, y, bits=24, cell_w=20, cell_h=28):
    import cv2
    token = _encode_frame_id_token(frame_id)
    box_w = bits * cell_w
    box_h = cell_h + 64
    cv2.rectangle(frame, (x - 10, y - 52), (x + box_w + 10, y + box_h), (0, 0, 0), -1)
    text = str(int(frame_id))
    cv2.putText(frame, text, (x, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (32, 32, 32), 2, cv2.LINE_AA)
    for i in range(bits):
        bit_index = bits - 1 - i
        bit = (token >> bit_index) & 1
        px = x + (i * cell_w)
        color = (255, 255, 255) if bit else (0, 0, 0)
        cv2.rectangle(frame, (px, y), (px + cell_w - 2, y + cell_h - 2), color, -1)
        cv2.rectangle(frame, (px, y), (px + cell_w - 2, y + cell_h - 2), (96, 96, 96), 1)

def _decode_frame_id_overlay(frame, x, y, bits=24, cell_w=20, cell_h=28):
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    token = 0
    for i in range(bits):
        cx = int(round(x + (i * cell_w) + (cell_w * 0.5)))
        cy = int(round(y + (cell_h * 0.5)))
        x0 = max(0, cx - 2)
        x1 = min(gray.shape[1], cx + 3)
        y0 = max(0, cy - 2)
        y1 = min(gray.shape[0], cy + 3)
        sample = gray[y0:y1, x0:x1]
        mean_v = float(sample.mean()) if sample.size else 0.0
        token = (token << 1) | (1 if mean_v >= 128.0 else 0)

    frame_id = (token >> 8) & 0xFFFF
    chk = token & 0xFF
    expected_chk = (frame_id ^ (frame_id >> 8) ^ 0x5A) & 0xFF
    ok = chk == expected_chk
    return frame_id, ok

def _map_overlay_geometry_callahan01_to_filtered(x, y, cell_w, cell_h):
    # callahan_01 filter: Crop(10,2,-8,-10) then LanczosResize(640,480)
    src_w, src_h = 720, 480
    crop_l, crop_t, crop_r, crop_b = 10, 2, 8, 10
    cropped_w = src_w - crop_l - crop_r
    cropped_h = src_h - crop_t - crop_b
    dst_w, dst_h = 640, 480
    sx = dst_w / float(cropped_w)
    sy = dst_h / float(cropped_h)
    fx = int(round((x - crop_l) * sx))
    fy = int(round((y - crop_t) * sy))
    fw = max(1, int(round(cell_w * sx)))
    fh = max(1, int(round(cell_h * sy)))
    return fx, fy, fw, fh

def import_step_6_module():
    def _install_whisper_stub():
        whisper_stub = types.ModuleType("whisper")
        whisper_utils_stub = types.ModuleType("whisper.utils")

        class _DummyWhisperModel:
            def transcribe(self, *_args, **_kwargs):
                return {"text": "", "segments": []}

        def _load_model(*_args, **_kwargs):
            return _DummyWhisperModel()

        def _get_writer(_fmt, _out_dir):
            def _writer(_result, out_path):
                Path(out_path).write_text("", encoding="utf-8")
            return _writer

        whisper_stub.load_model = _load_model
        whisper_utils_stub.get_writer = _get_writer
        whisper_stub.utils = whisper_utils_stub
        sys.modules["whisper"] = whisper_stub
        sys.modules["whisper.utils"] = whisper_utils_stub
        return whisper_stub, whisper_utils_stub

    try:
        import step_6_make_videos
        if getattr(step_6_make_videos, "whisper", None) is None:
            whisper_stub, whisper_utils_stub = _install_whisper_stub()
            step_6_make_videos.whisper = whisper_stub
            step_6_make_videos.get_writer = whisper_utils_stub.get_writer
        return step_6_make_videos
    except ModuleNotFoundError as exc:
        if exc.name != "whisper":
            raise

    _install_whisper_stub()

    import step_6_make_videos
    return step_6_make_videos

def test_step_4_generate_archive_metadata():
    print("Testing step_4_generate_archive_metadata.py...")
    shutil.copy(TESTDATA_DIR / "test_01_archive.mkv", ARCHIVE_DIR / "test_01_archive.mkv")
    import step_3_generate_archive_metadata
    assert step_3_generate_archive_metadata.main() is None
    assert step_3_generate_archive_metadata.ARCHIVE_CHECKSUM_FILE.stat().st_size > 50
    assert (ARCHIVE_DIR / "test_01_archive_mediainfo.txt").stat().st_size > 50
    print("Test step_4_generate_archive_metadata.py: PASSED.")
    step_3_generate_archive_metadata.ARCHIVE_CHECKSUM_FILE.unlink()
    shutil.rmtree(ARCHIVE_DIR / "test_01_archive_metadata")
    (ARCHIVE_DIR / "test_01_archive.mkv").unlink()
    (ARCHIVE_DIR / "test_01_archive_mediainfo.txt").unlink()
    (ARCHIVE_DIR / "test_01_archive_mediainfo.xml").unlink()
    (METADATA_DIR / "test_01_archive" / "markers.tsv").unlink(missing_ok=True)
    (METADATA_DIR / "test_01_archive" / "markers.mkvchapters.xml").unlink(missing_ok=True)
    del sys.modules['step_3_generate_archive_metadata']

def test_step_6_make_videos():
    print("Testing step_6_make_videos.py...")
    shutil.copy(TESTDATA_DIR / "test_01_archive.mkv", ARCHIVE_DIR / "test_01_archive.mkv")
    step_6_make_videos = import_step_6_module()
    assert step_6_make_videos.main() is None
    assert (CLIPS_DIR / "Test Video 01.mp4").stat().st_size > 100
    print("Test step_6_make_videos.py: PASSED.")
    (CLIPS_DIR / "Test Video 01.mp4").unlink()
    (CLIPS_DIR / "Test Video 01.srt").unlink(missing_ok=True)
    (CLIPS_DIR / "Test Video 01.vtt").unlink(missing_ok=True)
    (CLIPS_DIR / "Test Video 01.ass").unlink(missing_ok=True)
    (ARCHIVE_DIR / "test_01_archive.mkv").unlink(missing_ok=True)
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_title_filter_and_rebuild():
    print("Testing step_6_make_videos title filter and rebuild...")
    shutil.copy(TESTDATA_DIR / "test_01_archive.mkv", ARCHIVE_DIR / "test_01_archive.mkv")
    step_6_make_videos = import_step_6_module()

    out_mp4 = CLIPS_DIR / "Test Video 01.mp4"

    assert step_6_make_videos.main(["--title", "does-not-match"]) is None
    assert not out_mp4.exists()

    assert step_6_make_videos.main(["--title", "Video 01"]) is None
    assert out_mp4.exists()
    first_mtime = out_mp4.stat().st_mtime

    assert step_6_make_videos.main(["--title", "Video 01"]) is None
    second_mtime = out_mp4.stat().st_mtime
    assert second_mtime >= first_mtime

    print("Test step_6_make_videos title filter and rebuild: PASSED.")
    out_mp4.unlink(missing_ok=True)
    (CLIPS_DIR / "Test Video 01.srt").unlink(missing_ok=True)
    (CLIPS_DIR / "Test Video 01.vtt").unlink(missing_ok=True)
    (CLIPS_DIR / "Test Video 01.ass").unlink(missing_ok=True)
    (ARCHIVE_DIR / "test_01_archive.mkv").unlink(missing_ok=True)
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_badframe_sidecar_mapping():
    print("Testing step_6_make_videos badframe sidecar mapping...")
    step_6_make_videos = import_step_6_module()

    chapter = {
        "start": 1000 * 1001.0 / 30000.0,
        "end": 1010 * 1001.0 / 30000.0,
    }
    assert step_6_make_videos.chapter_global_frame_bounds(chapter) == (1000, 1010)
    rounded_chapter = {"start": 19.319, "end": 206.506}
    assert step_6_make_videos.chapter_global_frame_bounds(rounded_chapter) == (579, 6189)
    exact_start, exact_end = step_6_make_videos.chapter_exact_time_bounds(rounded_chapter)
    assert abs(exact_start - (579 * 1001.0 / 30000.0)) < 1e-9
    assert abs(exact_end - (6189 * 1001.0 / 30000.0)) < 1e-9
    local = step_6_make_videos.map_bad_ranges_to_chapter_local_frames([(999, 1002), (1010, 1015)], chapter)
    assert local == [0, 1, 2]
    assert local == [0, 1, 2]

    tmp_tsv = METADATA_DIR / "test_01_archive" / "_badframes_test.tsv"
    tmp_tsv.write_text(
        "start_frame\tend_frame\tnote\n"
        "100\t102\tshort,no_pad\n"
        "200\t200\t\n"
        "500\t502\t\n"
        "300\t1700\ttoo long\n"
        "300\t1700\tallow_long\n",
        encoding="utf-8",
    )
    try:
        ranges = step_6_make_videos.load_badframe_ranges(tmp_tsv)
        assert (100, 102) in ranges
        assert (200, 200) in ranges
        assert any(a <= 499 and b >= 502 for (a, b) in ranges)
        assert (298, 1700) in ranges
    finally:
        tmp_tsv.unlink(missing_ok=True)

    tmp_source_tsv = METADATA_DIR / "test_01_archive" / "_badframes_source_test.tsv"
    tmp_source_tsv.write_text(
        "start_frame\tend_frame\tsource_frame\tnote\n"
        "100\t102\t98\tno_pad\n"
        "200\t200\t\tauto\n"
        "300\t301\tno_pad\n",
        encoding="utf-8",
    )
    try:
        repairs = step_6_make_videos.load_badframe_repairs(tmp_source_tsv)
        assert (100, 102, 98) in repairs
        assert (200, 200, None) in repairs
        assert (300, 301, None) in repairs

        chapter_local = step_6_make_videos.map_bad_repairs_to_chapter_local_ranges(
            [(1000, 1002, 1005)],
            chapter,
        )
        assert chapter_local == [(0, 2, 5)]
    finally:
        tmp_source_tsv.unlink(missing_ok=True)

    tmp_bool_tsv = METADATA_DIR / "test_01_archive" / "_badframes_no_pad_bool.tsv"
    tmp_bool_tsv.write_text(
        "start_frame\tend_frame\tsource_frame\tno_pad\tnote\n"
        "100\t101\t99\ttrue\t\n"
        "200\t201\t199\tfalse\t\n"
        "300\t301\t299\t\tyes-no-note\n"
        "400\t401\t399\tinvalid\t\n",
        encoding="utf-8",
    )
    try:
        repairs = step_6_make_videos.load_badframe_repairs(tmp_bool_tsv)
        # no_pad=true keeps exact range.
        assert (100, 101, 99) in repairs
        # no_pad=false keeps adaptive pre-pad for 2-frame burst.
        assert (199, 201, 199) in repairs
        # blank no_pad falls back to note parsing (no no_pad token here -> adaptive pre-pad).
        assert (299, 301, 299) in repairs
        # invalid no_pad token is ignored (falls back to note/default behavior).
        assert (399, 401, 399) in repairs
    finally:
        tmp_bool_tsv.unlink(missing_ok=True)

    print("Test step_6_make_videos badframe sidecar mapping: PASSED.")
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_badframe_repair_injection_and_comment():
    print("Testing step_6_make_videos badframe repair injection and filmed comment...")
    step_6_make_videos = import_step_6_module()

    out = step_6_make_videos.build_badframe_prefilter_lines([6, 7, 8, 20])
    assert out.count("FreezeFrame(") == 2
    assert "FreezeFrame(20,20,21)" in out
    assert "FreezeFrame(6,8,9)" in out
    assert out.find("FreezeFrame(20,20,21)") < out.find("FreezeFrame(6,8,9)")

    out_override = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(10, 12, 20), (30, 30, None)]
    )
    assert "FreezeFrame(30,30,31)" in out_override
    assert "FreezeFrame(10,12,20)" in out_override

    out_invalid_override = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(6, 8, 7)]
    )
    assert "FreezeFrame(6,8,9)" in out_invalid_override

    out_forward_only = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(0, 0, None), (10, 10, None)]
    )
    # Auto-picked ranges should always use future source frames.
    assert "FreezeFrame(10,10,11)" in out_forward_only
    assert "FreezeFrame(10,10,9)" not in out_forward_only

    out_forward_only_adjacent = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(1, 1, None), (2, 2, None)]
    )
    assert "FreezeFrame(1,1,3)" in out_forward_only_adjacent
    assert "FreezeFrame(2,2,3)" in out_forward_only_adjacent
    assert "FreezeFrame(2,2,1)" not in out_forward_only_adjacent

    out_monotonic = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(0, 0, None), (100, 100, None)]
    )
    # Source-frame selection should remain forward and monotonic.
    assert "FreezeFrame(100,100,101)" in out_monotonic
    assert "FreezeFrame(100,100,99)" not in out_monotonic

    out_post = step_6_make_videos.build_badframe_postfilter_lines([6, 7, 8, 20])
    # Post-QTGMC stabilization: map source-frame repairs to doubled-rate output.
    assert "FreezeFrame(40,41,42)" in out_post
    assert "FreezeFrame(12,17,18)" in out_post

    c_none = step_6_make_videos.build_filmed_comment(
        None, "1995-03-18T19:25:00-08:00", "Altadena", "Tape 01", "00:01:00", "00:02:00"
    )
    assert c_none.startswith("Filmed on ")
    assert "Filmed by" not in c_none

    c_name = step_6_make_videos.build_filmed_comment(
        "Jim", "1995-03-18T19:25:00-08:00", "Altadena", "Tape 01", "00:01:00", "00:02:00"
    )
    assert c_name.startswith("Filmed by Jim on ")

    print("Test step_6_make_videos badframe repair injection and filmed comment: PASSED.")
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_make_create_avs_includes_chapter_bounds():
    print("Testing step_6_make_videos AVS generation with chapter bounds...")
    step_6_make_videos = import_step_6_module()
    tmp_filter = METADATA_DIR / "test_01_archive" / "_tmp_filter.avs"
    tmp_filter.write_text("c = last\nreturn c\n", encoding="utf-8")
    try:
        script = step_6_make_videos.make_create_avs(
            "C:/tmp/extracted.mkv",
            tmp_filter,
            bad_source_frames=[4, 5],
            chapter_start_frame=100,
            chapter_end_frame=200,
            no_bob=True,
        )
        assert "chapter_start_frame = 100" in script
        assert "chapter_end_frame = 200" in script
        assert "FreezeFrame(4,5,6)" in script
        assert "FreezeFrame(8,11,12)" in script
        assert "_tmp_filter.avs" in script
        assert "SelectEven()" in script
    finally:
        tmp_filter.unlink(missing_ok=True)

    print("Test step_6_make_videos AVS generation with chapter bounds: PASSED.")
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_real_badframes_do_not_pick_bad_sources():
    print("Testing step_6_make_videos against real badframes.tsv source picking...")
    step_6_make_videos = import_step_6_module()

    real_meta = ROOT / "metadata" / "callahan_01_archive"
    badframes_tsv = real_meta / "badframes.tsv"
    chapters_file = real_meta / "chapters.ffmetadata"
    if not badframes_tsv.exists() or not chapters_file.exists():
        print("Skipping real badframes source-picking test: callahan_01 metadata not present.")
        del sys.modules['step_6_make_videos']
        sys.modules.pop("whisper", None)
        sys.modules.pop("whisper.utils", None)
        return

    repairs = step_6_make_videos.load_badframe_repairs(badframes_tsv)
    raw_ranges = [(a, b) for (a, b, _src) in repairs]
    _ffm, chapters = parse_chapters(chapters_file)

    violations = []
    for ch in chapters:
        start, end = step_6_make_videos.chapter_global_frame_bounds(ch)
        max_local = (end - start) - 1
        if max_local < 0:
            continue

        local_repairs = step_6_make_videos.map_bad_repairs_to_chapter_local_ranges(repairs, ch)
        if not local_repairs:
            continue

        local_bad = set(
            step_6_make_videos.map_bad_ranges_to_chapter_local_frames(raw_ranges, ch)
        )
        resolved = step_6_make_videos._resolve_badframe_repair_ranges(
            bad_repair_ranges=local_repairs,
            max_source_frame=max_local,
        )

        replacement_by_frame = {}
        for a, b, src in resolved:
            if src < 0 or src > max_local:
                violations.append((ch.get("title", ""), a, b, src, "out_of_bounds"))
                continue
            if src in local_bad:
                violations.append((ch.get("title", ""), a, b, src, "bad_source"))
            for f in range(max(0, a), min(max_local, b) + 1):
                replacement_by_frame[f] = src

        for f in range(max_local + 1):
            shown = replacement_by_frame.get(f, f)
            if shown in local_bad:
                violations.append((ch.get("title", ""), f, f, shown, "shown_bad"))
                if len(violations) >= 20:
                    break
        if len(violations) >= 20:
            break

    assert not violations, "Badframe source-picking violations found: " + repr(violations[:20])
    print("Test step_6_make_videos real badframes source picking: PASSED.")
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_proxy_badframes_overlay_e2e():
    print("Testing step_6_make_videos proxy overlay + OpenCV decode badframe safety...")
    if os.getenv("RUN_PROXY_BADFRAME_E2E", "0").strip() != "1":
        print("Skipping proxy overlay E2E test. Set RUN_PROXY_BADFRAME_E2E=1 to enable.")
        return

    keep_outputs = os.getenv("RUN_PROXY_BADFRAME_E2E_KEEP", "1").strip() not in {"0", "false", "False"}

    try:
        import cv2  # noqa: F401
    except Exception:
        print("Skipping proxy overlay E2E test: OpenCV (cv2) is unavailable in this Python.")
        return

    step_6_make_videos = import_step_6_module()
    try:
        proxy_path = ROOT.parent / "Archive" / "callahan_01_archive_proxy.mp4"
        meta_dir = ROOT / "metadata" / "callahan_01_archive"
        filter_src = meta_dir / "filter.avs"
        badframes_src = meta_dir / "badframes.tsv"
        if not proxy_path.exists() or not filter_src.exists() or not badframes_src.exists():
            print("Skipping proxy overlay E2E test: archive proxy/filter/badframes not found.")
            return

        frame_start = 0
        frame_end = int(os.getenv("RUN_PROXY_BADFRAME_E2E_END", "18025"))
        if frame_end < frame_start:
            raise AssertionError("RUN_PROXY_BADFRAME_E2E_END must be >= 0.")
        frame_count = frame_end - frame_start + 1

        work_dir = ROOT / "test" / "_proxy_badframe_e2e"
        work_dir.mkdir(parents=True, exist_ok=True)
        stem = f"proxy_01_{frame_start}_{frame_end}"
        clip_path = work_dir / f"{stem}_clip.mp4"
        numbered_video_only_path = work_dir / f"{stem}_numbered_video_only.mp4"
        numbered_path = work_dir / f"{stem}_numbered.mp4"
        filtered_path = work_dir / f"{stem}_filtered.mp4"
        avs_path = work_dir / f"{stem}_script.avs"
        src_md5 = work_dir / f"{stem}_src.md5"
        clip_md5 = work_dir / f"{stem}_clip.md5"
        filter_copy = work_dir / "filter_copy.avs"
        badframes_copy = work_dir / "badframes_copy.tsv"
        shutil.copy(filter_src, filter_copy)
        shutil.copy(badframes_src, badframes_copy)

        vf_select = f"select='between(n\\,{frame_start}\\,{frame_end})',setpts=N/FRAME_RATE/TB"
        subprocess.run(
            [
                str(FFMPEG_BIN), "-nostdin", "-v", "error",
                "-i", str(proxy_path),
                "-vf", vf_select,
                "-map", "0:v:0",
                "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-an",
                "-y", str(clip_path),
            ],
            check=True,
        )

        # Verify extracted frame order/identity exactly matches selected source frames.
        subprocess.run(
            [
                str(FFMPEG_BIN), "-nostdin", "-v", "error",
                "-i", str(proxy_path),
                "-vf", vf_select,
                "-an",
                "-f", "framemd5",
                "-y", str(src_md5),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(FFMPEG_BIN), "-nostdin", "-v", "error",
                "-i", str(clip_path),
                "-an",
                "-f", "framemd5",
                "-y", str(clip_md5),
            ],
            check=True,
        )
        src_hashes = _framemd5_hashes(src_md5)
        clip_hashes = _framemd5_hashes(clip_md5)
        assert len(src_hashes) == frame_count
        assert len(clip_hashes) == frame_count
        assert src_hashes == clip_hashes, "Extracted clip frame order/content mismatch."

        # Draw frame IDs on every frame so downstream filter output can be decoded.
        import cv2
        cap = cv2.VideoCapture(str(clip_path))
        assert cap.isOpened(), f"Unable to open extracted clip: {clip_path}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30000.0 / 1001.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert width == 720 and height == 480, f"Unexpected proxy frame size: {width}x{height}"

        bits = 24
        cell_w = 20
        cell_h = 28
        draw_x = 180
        draw_y = 330

        writer = cv2.VideoWriter(
            str(numbered_video_only_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        assert writer.isOpened(), f"Unable to open numbered writer: {numbered_path}"

        for idx in range(frame_count):
            ok, frame = cap.read()
            assert ok, f"Extracted clip ended early at frame {idx}."
            frame_id = frame_start + idx
            _draw_frame_id_overlay(frame, frame_id, draw_x, draw_y, bits=bits, cell_w=cell_w, cell_h=cell_h)
            writer.write(frame)
        extra_ok, _extra = cap.read()
        cap.release()
        writer.release()
        assert not extra_ok, "Extracted clip had more frames than expected selection."

        # Add a short silent audio track so FFmpegSource2/AVS can open this clip reliably.
        subprocess.run(
            [
                str(FFMPEG_BIN), "-nostdin", "-v", "error",
                "-i", str(numbered_video_only_path),
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                "-shortest",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
                "-y", str(numbered_path),
            ],
            check=True,
        )

        # Sanity-check OpenCV decoding on the numbered clip itself.
        cap_num = cv2.VideoCapture(str(numbered_path))
        assert cap_num.isOpened(), f"Unable to open numbered clip: {numbered_path}"
        for idx in range(frame_count):
            ok, frame = cap_num.read()
            assert ok, f"Numbered clip ended early at frame {idx}."
            decoded_id, valid = _decode_frame_id_overlay(
                frame, draw_x, draw_y, bits=bits, cell_w=cell_w, cell_h=cell_h
            )
            assert valid, f"Overlay checksum invalid in numbered clip frame {idx}."
            assert decoded_id == frame_start + idx, (
                f"Overlay decode mismatch in numbered clip frame {idx}: "
                f"got {decoded_id}, expected {frame_start + idx}."
            )
        cap_num.release()

        repairs = step_6_make_videos.load_badframe_repairs(badframes_copy)
        fake_chapter = {
            "start": 0.0,
            "end": (frame_end + 1) * 1001.0 / 30000.0,
        }
        local_repairs = step_6_make_videos.map_bad_repairs_to_chapter_local_ranges(repairs, fake_chapter)
        script_text = step_6_make_videos.make_create_avs(
            str(numbered_path),
            filter_copy,
            bad_repair_ranges=local_repairs,
            chapter_start_frame=0,
            chapter_end_frame=frame_count,
            no_bob=True,
        )
        avs_path.write_text(script_text, encoding="ascii")

        subprocess.run(
            [
                str(FFMPEG_BIN), "-nostdin", "-v", "error",
                "-i", str(avs_path),
                "-an",
                "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-y", str(filtered_path),
            ],
            check=True,
        )

        bad_set = set()
        for a, b in step_6_make_videos.load_badframe_ranges(badframes_copy):
            lo = max(frame_start, int(a))
            hi = min(frame_end, int(b))
            if hi < lo:
                continue
            for f in range(lo, hi + 1):
                bad_set.add(f)

        rx, ry, rw, rh = _map_overlay_geometry_callahan01_to_filtered(draw_x, draw_y, cell_w, cell_h)
        cap_out = cv2.VideoCapture(str(filtered_path))
        assert cap_out.isOpened(), f"Unable to open filtered clip: {filtered_path}"
        violations = []
        decode_failures = []
        for idx in range(frame_count):
            ok, frame = cap_out.read()
            if not ok:
                violations.append((idx, "missing_frame"))
                break
            shown_id, valid = _decode_frame_id_overlay(frame, rx, ry, bits=bits, cell_w=rw, cell_h=rh)
            if not valid:
                decode_failures.append((idx, shown_id))
                if len(decode_failures) >= 20:
                    break
                continue
            if shown_id in bad_set:
                violations.append((idx, shown_id))
                if len(violations) >= 20:
                    break
        cap_out.release()

        assert not decode_failures, (
            "Failed to decode frame-id overlay in filtered clip: "
            + repr(decode_failures[:20])
        )
        assert not violations, (
            "Filtered output displayed bad source frame IDs: "
            + repr(violations[:20])
        )
        print("Test step_6_make_videos proxy overlay + OpenCV decode badframe safety: PASSED.")

        if not keep_outputs:
            shutil.rmtree(work_dir, ignore_errors=True)
    finally:
        del sys.modules['step_6_make_videos']
        sys.modules.pop("whisper", None)
        sys.modules.pop("whisper.utils", None)

def test_step_drive_checksums():
    print("Testing step_7_generate_drive_checksum.py...")
    import step_7_generate_drive_checksum
    assert step_7_generate_drive_checksum.main() is None
    import step_8_verify_drive_checksum
    assert step_8_verify_drive_checksum.main() is None
    print("Test step_drive_checksums: PASSED.")
    DRIVE_CHECKSUM_FILE.unlink()
    del sys.modules['step_7_generate_drive_checksum']
    del sys.modules['step_8_verify_drive_checksum']

def test_sha3_generate_and_verify():
    print("Testing SHA3-256 generate + verify...")
    test_root = ARCHIVE_DIR / "_sha3_test"
    test_root.mkdir(parents=True, exist_ok=True)
    test_file = test_root / "hello.txt"
    test_file.write_text("hello sha3", encoding="utf-8")

    manifest = test_root / "sha3-manifest.txt"
    write_sha3_manifest(test_root, manifest, relative_base=test_root)
    rc = verify_manifest(test_root, manifest, algo="sha3")
    assert rc == 0

    manifest.unlink()
    test_file.unlink()
    test_root.rmdir()
    print("Test SHA3-256 generate + verify: PASSED.")

def test_blake3_verify_only():
    print("Testing BLAKE3 verify (legacy)...")
    test_root = ARCHIVE_DIR / "_blake3_test"
    test_root.mkdir(parents=True, exist_ok=True)
    test_file = test_root / "hello.txt"
    test_file.write_text("hello blake3", encoding="utf-8")

    manifest = test_root / "blake3-manifest.txt"
    old_cwd = os.getcwd()
    os.chdir(test_root)
    try:
        r = subprocess.run([str(B3SUM_BIN), test_file.name], capture_output=True, text=True)
        assert r.returncode == 0
        manifest.write_text(r.stdout, encoding="utf-8")
    finally:
        os.chdir(old_cwd)

    rc = verify_manifest(test_root, manifest, algo="blake3")
    assert rc == 0

    manifest.unlink()
    test_file.unlink()
    test_root.rmdir()
    print("Test BLAKE3 verify (legacy): PASSED.")

def main():
    print("Running tests...")
    test_step_4_generate_archive_metadata()
    test_step_6_make_videos()
    test_step_6_title_filter_and_rebuild()
    test_step_6_badframe_sidecar_mapping()
    test_step_6_badframe_repair_injection_and_comment()
    test_step_6_make_create_avs_includes_chapter_bounds()
    test_step_6_real_badframes_do_not_pick_bad_sources()
    test_step_6_proxy_badframes_overlay_e2e()
    test_step_drive_checksums()
    test_sha3_generate_and_verify()
    test_blake3_verify_only()

if __name__ == "__main__":
    main()
