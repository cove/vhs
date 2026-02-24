import shutil
import os
import sys
import subprocess
import types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TEST_ENV"] = "1"
from common import *
TESTDATA_DIR = BASE / "test" / "test_data"
TEST_ARCHIVE_FIXTURE = TESTDATA_DIR / "test_01_archive.mkv"
os.environ["PYTHONPATH"] = str(BASE)

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

def _require_test_archive_fixture(test_name: str) -> bool:
    if TEST_ARCHIVE_FIXTURE.exists():
        return True
    print(f"Skipping {test_name}: missing fixture {TEST_ARCHIVE_FIXTURE}")
    return False

def _ensure_test_archive_metadata_dir() -> Path:
    p = METADATA_DIR / "test_01_archive"
    p.mkdir(parents=True, exist_ok=True)
    return p

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
    if not _require_test_archive_fixture("test_step_4_generate_archive_metadata"):
        return
    shutil.copy(TEST_ARCHIVE_FIXTURE, ARCHIVE_DIR / "test_01_archive.mkv")
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
    if not _require_test_archive_fixture("test_step_6_make_videos"):
        return
    shutil.copy(TEST_ARCHIVE_FIXTURE, ARCHIVE_DIR / "test_01_archive.mkv")
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
    if not _require_test_archive_fixture("test_step_6_title_filter_and_rebuild"):
        return
    shutil.copy(TEST_ARCHIVE_FIXTURE, ARCHIVE_DIR / "test_01_archive.mkv")
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
    test_meta_dir = _ensure_test_archive_metadata_dir()

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

    tmp_tsv = test_meta_dir / "_frame_quality_test.tsv"
    tmp_tsv.write_text(
        "frame\tscore\tbad_frame\tmanual_override\n"
        "102\t0.1\t1\t0\n"
        "100\t0.2\t1\t0\n"
        "101\t0.3\t1\t0\n"
        "101\t0.4\t1\t1\n"
        "200\t0.5\t1\t0\n"
        "500\t0.6\t1\t0\n"
        "502\t0.7\t1\t0\n"
        "999\t0.8\t0\t0\n",
        encoding="utf-8",
    )
    try:
        ranges = step_6_make_videos.load_badframe_ranges(tmp_tsv)
        assert (100, 102) in ranges
        assert (200, 200) in ranges
        assert (500, 500) in ranges
        assert (502, 502) in ranges
    finally:
        tmp_tsv.unlink(missing_ok=True)

    # Invalid sidecar schema should fail fast.
    tmp_invalid_tsv = test_meta_dir / "_frame_quality_invalid.tsv"
    tmp_invalid_tsv.write_text("start_frame\tend_frame\n100\t102\n", encoding="utf-8")
    try:
        try:
            step_6_make_videos.load_badframe_repairs(tmp_invalid_tsv)
            raise AssertionError("Expected ValueError for invalid frame_quality sidecar schema.")
        except ValueError:
            pass
    finally:
        tmp_invalid_tsv.unlink(missing_ok=True)

    # Repairs parsed from frame_quality sidecar should map with source=None.
    tmp_repairs_tsv = test_meta_dir / "_frame_quality_repairs.tsv"
    tmp_repairs_tsv.write_text(
        "frame\tscore\tbad_frame\tmanual_override\n"
        "1000\t0.1\t1\t0\n"
        "1001\t0.1\t1\t0\n"
        "1002\t0.1\t1\t0\n"
        "1005\t0.1\t1\t0\n",
        encoding="utf-8",
    )
    try:
        repairs = step_6_make_videos.load_badframe_repairs(tmp_repairs_tsv)
        assert (1000, 1002, None) in repairs
        assert (1005, 1005, None) in repairs
        chapter_local = step_6_make_videos.map_bad_repairs_to_chapter_local_ranges(repairs, chapter)
        assert chapter_local == [(0, 2, None), (5, 5, None)]
    finally:
        tmp_repairs_tsv.unlink(missing_ok=True)

    # Local sidecar schema is supported when chapter context is provided.
    tmp_local_tsv = test_meta_dir / "_frame_quality_local.tsv"
    tmp_local_tsv.write_text(
        "chapter\tlocal_frame\tscore\tbad_frame\tmanual_override\n"
        "Chapter A\t10\t0.1\t1\t1\n"
        "Chapter A\t11\t0.1\t1\t1\n"
        "Chapter B\t8\t0.1\t1\t1\n",
        encoding="utf-8",
    )
    try:
        local_repairs = step_6_make_videos.load_badframe_repairs(
            tmp_local_tsv,
            chapter_title="Chapter A",
            chapter_start_frame=1000,
        )
        assert local_repairs == [(1010, 1011, None)]
        local_ranges = step_6_make_videos.load_badframe_ranges(
            tmp_local_tsv,
            chapter_title="Chapter A",
            chapter_start_frame=1000,
        )
        assert local_ranges == [(1010, 1011)]
        no_match = step_6_make_videos.load_badframe_repairs(
            tmp_local_tsv,
            chapter_title="Chapter C",
            chapter_start_frame=1000,
        )
        assert no_match == []
    finally:
        tmp_local_tsv.unlink(missing_ok=True)

    print("Test step_6_make_videos badframe sidecar mapping: PASSED.")
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_badframe_repair_injection_and_comment():
    print("Testing step_6_make_videos badframe repair injection and filmed comment...")
    step_6_make_videos = import_step_6_module()

    out = step_6_make_videos.build_badframe_prefilter_lines([6, 7, 8, 20])
    assert out.count("FreezeFrame(") == 3
    assert "FreezeFrame(20,20,21)" in out
    assert "FreezeFrame(7,8,9)" in out
    assert "FreezeFrame(6,6,5)" in out
    assert out.find("FreezeFrame(20,20,21)") < out.find("FreezeFrame(7,8,9)")
    assert out.find("FreezeFrame(7,8,9)") < out.find("FreezeFrame(6,6,5)")

    out_override = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(10, 12, 20), (30, 30, None)]
    )
    assert "FreezeFrame(30,30,31)" in out_override
    assert "FreezeFrame(10,12,20)" in out_override

    out_invalid_override = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(6, 8, 7)]
    )
    assert "FreezeFrame(7,8,9)" in out_invalid_override
    assert "FreezeFrame(6,6,5)" in out_invalid_override

    out_forward_only = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(0, 0, None), (10, 10, None)]
    )
    # Auto-picked ranges should always use future source frames.
    assert "FreezeFrame(10,10,11)" in out_forward_only
    assert "FreezeFrame(10,10,9)" not in out_forward_only

    out_forward_only_adjacent = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(1, 1, None), (2, 2, None)]
    )
    assert "FreezeFrame(1,2,3)" in out_forward_only_adjacent
    assert "FreezeFrame(2,2,1)" not in out_forward_only_adjacent

    out_monotonic = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(0, 0, None), (100, 100, None)]
    )
    # Source-frame selection should remain forward and monotonic.
    assert "FreezeFrame(100,100,101)" in out_monotonic
    assert "FreezeFrame(100,100,99)" not in out_monotonic

    out_post = step_6_make_videos.build_badframe_postfilter_lines([6, 7, 8, 20])
    # Post-QTGMC stabilization is single-rate when QTGMC uses FPSDivisor=2.
    assert "FreezeFrame(20,20,21)" in out_post
    assert "FreezeFrame(7,8,9)" in out_post
    assert "FreezeFrame(6,6,5)" in out_post

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


def test_step_6_badframe_split_strategy_logic_paths():
    print("Testing step_6_make_videos badframe split strategy logic paths...")
    step_6_make_videos = import_step_6_module()

    # Both-side neighbor split (even span).
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(10, 13, None)]
    )
    assert r == [(10, 11, 9), (12, 13, 14)]

    # Both-side neighbor split (odd span): later half gets the extra frame.
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(10, 14, None)]
    )
    assert r == [(10, 11, 9), (12, 14, 15)]

    # Chapter start edge: no previous-good source, use next-good for full range.
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(0, 2, None)],
        max_source_frame=10,
    )
    assert r == [(0, 2, 3)]

    # Chapter end edge: no next-good source in bounds, use previous-good for full range.
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(8, 10, None)],
        max_source_frame=10,
    )
    assert r == [(8, 10, 7)]

    # Unrepairable edge case: no previous or next source exists.
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(0, 2, None)],
        max_source_frame=2,
    )
    assert r == []

    # Explicit valid override should be preserved as-is.
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(10, 12, 20)],
        max_source_frame=30,
    )
    assert r == [(10, 12, 20)]

    # Explicit invalid override should fall back to auto split behavior.
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(10, 12, 50)],
        max_source_frame=30,
    )
    assert r == [(10, 10, 9), (11, 12, 13)]

    # Adjacent bad ranges should avoid selecting bad frames as sources.
    r = step_6_make_videos._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(5, 6, None), (7, 8, None)],
        max_source_frame=20,
    )
    assert r == [(5, 5, 4), (6, 6, 9), (7, 7, 4), (8, 8, 9)]

    print("Test step_6_make_videos badframe split strategy logic paths: PASSED.")
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)

def test_step_6_make_create_avs_includes_chapter_bounds():
    print("Testing step_6_make_videos AVS generation with chapter bounds...")
    step_6_make_videos = import_step_6_module()
    tmp_filter = _ensure_test_archive_metadata_dir() / "_tmp_filter.avs"
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
        assert "FreezeFrame(5,5,6)" in script
        assert "FreezeFrame(4,4,3)" in script
        assert script.count("FreezeFrame(5,5,6)") == 2
        assert script.count("FreezeFrame(4,4,3)") == 2
        assert "_tmp_filter.avs" in script
        assert "SelectEven()" not in script
    finally:
        tmp_filter.unlink(missing_ok=True)

    print("Test step_6_make_videos AVS generation with chapter bounds: PASSED.")
    del sys.modules['step_6_make_videos']
    sys.modules.pop("whisper", None)
    sys.modules.pop("whisper.utils", None)


def test_step_6_real_badframes_do_not_pick_bad_sources():
    print("Testing step_6_make_videos against real frame_quality.tsv source picking...")
    step_6_make_videos = import_step_6_module()

    real_meta = ROOT / "metadata" / "callahan_01_archive"
    frame_quality_tsv = real_meta / "frame_quality.tsv"
    chapters_file = real_meta / "chapters.ffmetadata"
    if not frame_quality_tsv.exists() or not chapters_file.exists():
        print("Skipping real frame-quality source-picking test: callahan_01 metadata not present.")
        del sys.modules['step_6_make_videos']
        sys.modules.pop("whisper", None)
        sys.modules.pop("whisper.utils", None)
        return

    repairs = step_6_make_videos.load_badframe_repairs(frame_quality_tsv)
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


def test_step_6_frame_quality_ingest_exact_archive01():
    print("Testing step_6_make_videos exact ingest from archive-01 frame_quality.tsv...")
    step_6_make_videos = import_step_6_module()
    try:
        real_meta = ROOT / "metadata" / "callahan_01_archive"
        frame_quality_tsv = real_meta / "frame_quality.tsv"
        chapters_file = real_meta / "chapters.ffmetadata"
        if not frame_quality_tsv.exists() or not chapters_file.exists():
            print("Skipping exact frame-quality ingest test: callahan_01 metadata not present.")
            return

        # Exact bad-frame set from source TSV.
        bad_exact = set()
        idx_frame = 0
        idx_bad = 2
        for raw in frame_quality_tsv.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split("\t")]
            low = [p.lower() for p in parts]
            if low and low[0] == "frame":
                idx_frame = low.index("frame")
                idx_bad = low.index("bad_frame")
                continue
            try:
                frame = int(parts[idx_frame])
                is_bad = int(parts[idx_bad]) == 1
            except Exception:
                continue
            if is_bad:
                bad_exact.add(frame)

        repairs = step_6_make_videos.load_badframe_repairs(frame_quality_tsv)
        bad_from_repairs = set()
        for a, b, _src in repairs:
            for f in range(int(a), int(b) + 1):
                bad_from_repairs.add(f)

        assert bad_from_repairs == bad_exact, (
            "step_6 frame_quality ingestion mismatch: "
            f"missing={sorted(bad_exact - bad_from_repairs)[:20]} "
            f"extra={sorted(bad_from_repairs - bad_exact)[:20]}"
        )

        # Chapter-local mapping should preserve the exact per-chapter intersections.
        _ffm, chapters = parse_chapters(chapters_file)
        raw_ranges = [(a, b) for (a, b, _src) in repairs]
        for ch in chapters:
            start, end = step_6_make_videos.chapter_global_frame_bounds(ch)
            expect_local = {
                f - start for f in bad_exact
                if start <= f <= max(start, end - 1)
            }
            got_local = set(step_6_make_videos.map_bad_ranges_to_chapter_local_frames(raw_ranges, ch))
            assert got_local == expect_local, (
                f"chapter mapping mismatch for '{ch.get('title', '')}': "
                f"missing={sorted(expect_local - got_local)[:10]} "
                f"extra={sorted(got_local - expect_local)[:10]}"
            )
        print("Test step_6_make_videos exact ingest from archive-01 frame_quality.tsv: PASSED.")
    finally:
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
        frame_quality_src = meta_dir / "frame_quality.tsv"
        if not proxy_path.exists() or not filter_src.exists() or not frame_quality_src.exists():
            print("Skipping proxy overlay E2E test: archive proxy/filter/frame_quality not found.")
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
        frame_quality_copy = work_dir / "frame_quality_copy.tsv"
        shutil.copy(filter_src, filter_copy)
        shutil.copy(frame_quality_src, frame_quality_copy)

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

        repairs = step_6_make_videos.load_badframe_repairs(frame_quality_copy)
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
        for a, b in step_6_make_videos.load_badframe_ranges(frame_quality_copy):
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

def test_step_6_qtgmc_freezeframe_long_e2e():
    print("Testing step_6_make_videos QTGMC + FreezeFrame long-range drift safety...")
    if os.getenv("RUN_QTGMC_FREEZE_E2E", "0").strip() != "1":
        print("Skipping QTGMC FreezeFrame E2E test. Set RUN_QTGMC_FREEZE_E2E=1 to enable.")
        return
    if sys.platform != "win32":
        print("Skipping QTGMC FreezeFrame E2E test: AviSynth/QTGMC path is Windows-only.")
        return

    keep_outputs = os.getenv("RUN_QTGMC_FREEZE_E2E_KEEP", "1").strip() not in {"0", "false", "False"}

    try:
        import cv2  # noqa: F401
    except Exception:
        print("Skipping QTGMC FreezeFrame E2E test: OpenCV (cv2) is unavailable in this Python.")
        return

    step_6_make_videos = import_step_6_module()
    try:
        proxy_path = ROOT.parent / "Archive" / "callahan_01_archive_proxy.mp4"
        if not proxy_path.exists():
            print("Skipping QTGMC FreezeFrame E2E test: callahan_01 proxy not found.")
            return

        frame_start = int(os.getenv("RUN_QTGMC_FREEZE_E2E_START", "12000"))
        frame_end = int(os.getenv("RUN_QTGMC_FREEZE_E2E_END", str(frame_start + 6999)))
        if frame_end < frame_start:
            raise AssertionError("RUN_QTGMC_FREEZE_E2E_END must be >= RUN_QTGMC_FREEZE_E2E_START.")
        frame_count = frame_end - frame_start + 1
        if frame_count < 6000:
            raise AssertionError(
                "QTGMC FreezeFrame E2E requires at least 6000 frames; "
                f"got {frame_count} ({frame_start}-{frame_end})."
            )

        work_dir = ROOT / "test" / "_qtgmc_freeze_e2e"
        work_dir.mkdir(parents=True, exist_ok=True)
        stem = f"qtgmc_freeze_{frame_start}_{frame_end}"
        clip_path = work_dir / f"{stem}_clip.mkv"
        numbered_video_only_path = work_dir / f"{stem}_numbered_video_only.mp4"
        numbered_path = work_dir / f"{stem}_numbered.mp4"
        filtered_path = work_dir / f"{stem}_filtered.mp4"
        avs_path = work_dir / f"{stem}_script.avs"
        filter_path = work_dir / f"{stem}_qtgmc_filter.avs"
        src_md5 = work_dir / f"{stem}_src.md5"
        clip_md5 = work_dir / f"{stem}_clip.md5"

        vf_select = f"select='between(n\\,{frame_start}\\,{frame_end})',setpts=N/FRAME_RATE/TB"
        extract_start_sec = frame_start * 1001.0 / 30000.0
        extract_end_sec = (frame_end + 1) * 1001.0 / 30000.0
        subprocess.run(
            step_6_make_videos.make_extract_chapter(
                proxy_path,
                extract_start_sec,
                extract_end_sec,
                clip_path,
                start_frame=frame_start,
                end_frame=frame_end + 1,
            ),
            check=True,
        )

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
        assert src_hashes == clip_hashes, "Extracted long clip frame order/content mismatch."

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
        cell_h = 30
        draw_x = 170
        draw_y = 320

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

        # Synthetic bad ranges across the long clip to catch drift at boundaries and deep timeline positions.
        bad_ranges_local = [
            (0, 2),
            (47, 55),
            (1024, 1041),
            (3072, 3099),
            (frame_count // 2 - 12, frame_count // 2 + 17),
            (frame_count - 140, frame_count - 121),
            (frame_count - 6, frame_count - 1),
        ]
        bad_ranges_local = [
            (max(0, int(a)), min(frame_count - 1, int(b)))
            for a, b in bad_ranges_local
            if int(a) <= int(b)
        ]
        bad_ranges_local = [r for r in bad_ranges_local if r[0] <= r[1]]
        assert bad_ranges_local, "No valid bad ranges for QTGMC FreezeFrame E2E."

        resolved_local_repairs = step_6_make_videos._resolve_badframe_repair_ranges(
            bad_repair_ranges=[(a, b, None) for a, b in bad_ranges_local],
            max_source_frame=frame_count - 1,
        )
        assert resolved_local_repairs, "No resolved badframe repairs generated for long-range E2E."

        expected_local_shown = list(range(frame_count))
        for a, b, src in resolved_local_repairs:
            assert src is not None
            for fi in range(max(0, int(a)), min(frame_count - 1, int(b)) + 1):
                expected_local_shown[fi] = int(src)

        filter_path.write_text(
            "c = last\n"
            "c = c.AssumeTFF()\n"
            "c = QTGMC(Preset=\"Very Fast\", FPSDivisor=2)\n"
            "c\n",
            encoding="ascii",
        )

        script_text = step_6_make_videos.make_create_avs(
            str(numbered_path),
            filter_path,
            bad_repair_ranges=resolved_local_repairs,
            chapter_start_frame=0,
            chapter_end_frame=frame_count,
            no_bob=True,
        )
        assert "FreezeFrame(" in script_text, "AVS script is missing FreezeFrame repair lines."
        assert filter_path.name in script_text, "AVS script does not import the QTGMC filter script."
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

        cap_out = cv2.VideoCapture(str(filtered_path))
        assert cap_out.isOpened(), f"Unable to open filtered clip: {filtered_path}"
        mismatches = []
        decode_failures = []
        for idx in range(frame_count):
            ok, frame = cap_out.read()
            if not ok:
                mismatches.append((idx, "missing_frame"))
                break
            shown_id, valid = _decode_frame_id_overlay(
                frame, draw_x, draw_y, bits=bits, cell_w=cell_w, cell_h=cell_h
            )
            if not valid:
                decode_failures.append((idx, shown_id))
                if len(decode_failures) >= 20:
                    break
                continue
            expected_global = frame_start + expected_local_shown[idx]
            if int(shown_id) != int(expected_global):
                mismatches.append((idx, int(shown_id), int(expected_global)))
                if len(mismatches) >= 20:
                    break
        cap_out.release()

        assert not decode_failures, (
            "Failed to decode frame-id overlay in QTGMC filtered long clip: "
            + repr(decode_failures[:20])
        )
        assert not mismatches, (
            "QTGMC+FreezeFrame long-range drift/mapping mismatch: "
            + repr(mismatches[:20])
        )
        print("Test step_6_make_videos QTGMC + FreezeFrame long-range drift safety: PASSED.")

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

def test_vhs_tuner_toggle_override_cycle():
    print("Testing vhs_tuner frame-toggle override cycle...")
    import vhs_tuner

    fids = [100, 101, 102]
    sigs = {
        "chroma": np.array([0.0, 10.0, 20.0], dtype=np.float64),
        "noise": np.zeros(3, dtype=np.float64),
        "tear": np.zeros(3, dtype=np.float64),
        "wave": np.zeros(3, dtype=np.float64),
    }

    # 3-state manual override:
    # auto-good -> force bad -> clear
    # auto-bad  -> force good -> clear
    overrides = vhs_tuner.toggle_frame_override(
        fid=100, fids=fids, sigs=sigs, overrides={},
        wc=1.0, wn=0.0, wt=0.0, ww=0.0, tm="value", ik=3.5, tv=0.0, bp=10.0,
    )
    assert overrides.get(100) == "bad"
    overrides = vhs_tuner.toggle_frame_override(
        fid=100, fids=fids, sigs=sigs, overrides=overrides,
        wc=1.0, wn=0.0, wt=0.0, ww=0.0, tm="value", ik=3.5, tv=0.0, bp=10.0,
    )
    assert 100 not in overrides

    overrides = vhs_tuner.toggle_frame_override(
        fid=102, fids=fids, sigs=sigs, overrides={},
        wc=1.0, wn=0.0, wt=0.0, ww=0.0, tm="value", ik=3.5, tv=0.0, bp=10.0,
    )
    assert overrides.get(102) == "good"
    overrides = vhs_tuner.toggle_frame_override(
        fid=102, fids=fids, sigs=sigs, overrides=overrides,
        wc=1.0, wn=0.0, wt=0.0, ww=0.0, tm="value", ik=3.5, tv=0.0, bp=10.0,
    )
    assert 102 not in overrides

    # Regression check: signal sparkline HTML should include the red cut line.
    scores = vhs_tuner.combined_score(sigs, 1.0, 0.0, 0.0, 0.0)
    thr = vhs_tuner.compute_threshold(scores, "value", 3.5, 0.0, 10.0)
    sc_ch, sc_no, sc_te, sc_wa, _ = vhs_tuner.build_sparklines_html(
        sigs=sigs, scores=scores, threshold=thr, wc=0.2, wn=0.3, wt=0.4, ww=0.5
    )
    for svg in (sc_ch, sc_no, sc_te, sc_wa):
        assert 'stroke="#e03030"' in svg

    print("Test vhs_tuner frame-toggle override cycle: PASSED.")


def _write_unit_chapters_ffmetadata(meta_root: Path, bad_csv: str = "") -> Path:
    cf = meta_root / "unit_archive" / "chapters.ffmetadata"
    cf.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ";FFMETADATA1",
        "[CHAPTER]",
        "TIMEBASE=1001/30000",
        "START=1000",
        "END=1100",
        "TITLE=Unit Chapter",
        f"BAD_FRAMES={bad_csv}",
        "",
    ]
    cf.write_text("\n".join(lines), encoding="utf-8")
    return cf


def test_vhs_tuner_manual_click_persists_bad_frames():
    print("Testing vhs_tuner manual click persistence to chapters BAD_FRAMES...")
    import tempfile
    import time
    import vhs_tuner

    fids = [1000]
    sigs = {
        "chroma": np.array([0.0], dtype=np.float64),
        "noise": np.array([0.0], dtype=np.float64),
        "tear": np.array([0.0], dtype=np.float64),
        "wave": np.array([0.0], dtype=np.float64),
    }

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="")
            overrides, last_click, dbg = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1000",
                fids=fids,
                sigs=sigs,
                overrides={},
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
                last_click_event={"fid": -1, "ts": -1},
            )
            assert overrides.get(1000) == "bad", dbg
            chapters = vhs_tuner.parse_ffmetadata_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == []

            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=overrides,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
            )
            chapters = vhs_tuner.parse_ffmetadata_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == [1000]

            time.sleep(0.30)
            overrides2, _last2, dbg2 = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1400",
                fids=fids,
                sigs=sigs,
                overrides=overrides,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
                last_click_event=last_click,
            )
            assert 1000 not in overrides2, dbg2
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=overrides2,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
            )
            chapters = vhs_tuner.parse_ffmetadata_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == []
        finally:
            vhs_tuner.METADATA_DIR = old_meta
    print("Test vhs_tuner manual click persistence to chapters BAD_FRAMES: PASSED.")


def test_vhs_tuner_click_dedupe_prevents_double_toggle():
    print("Testing vhs_tuner click dedupe for duplicate events...")
    import tempfile
    import vhs_tuner

    fids = [1000]
    sigs = {
        "chroma": np.array([0.0], dtype=np.float64),
        "noise": np.array([0.0], dtype=np.float64),
        "tear": np.array([0.0], dtype=np.float64),
        "wave": np.array([0.0], dtype=np.float64),
    }

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        vhs_tuner.METADATA_DIR = Path(td)
        try:
            overrides, last_click, dbg = vhs_tuner.apply_manual_click_override(
                raw_click="1000:2000",
                fids=fids,
                sigs=sigs,
                overrides={},
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
                last_click_event={"fid": -1, "ts": -1},
            )
            assert overrides.get(1000) == "bad", dbg

            overrides2, last2, dbg2 = vhs_tuner.apply_manual_click_override(
                raw_click="1000:2050",
                fids=fids,
                sigs=sigs,
                overrides=overrides,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
                last_click_event=last_click,
            )
            assert overrides2 == overrides
            assert last2 == last_click
            assert "ignored: duplicate click" in dbg2
        finally:
            vhs_tuner.METADATA_DIR = old_meta
    print("Test vhs_tuner click dedupe for duplicate events: PASSED.")


def test_vhs_tuner_manual_click_modes_bad_and_good():
    print("Testing vhs_tuner manual click mark modes (bad/good/clear)...")
    import tempfile
    import time
    import vhs_tuner

    fids = [1000]
    sigs = {
        "chroma": np.array([0.0], dtype=np.float64),
        "noise": np.array([0.0], dtype=np.float64),
        "tear": np.array([0.0], dtype=np.float64),
        "wave": np.array([0.0], dtype=np.float64),
    }

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="")
            ov_bad, last_bad, dbg_bad = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1000",
                fids=fids,
                sigs=sigs,
                overrides={},
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
                mark_mode="bad",
                last_click_event={"fid": -1, "ts": -1},
            )
            assert ov_bad.get(1000) == "bad", dbg_bad
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=ov_bad,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
            )
            ch = vhs_tuner._find_chapter(vhs_tuner.parse_ffmetadata_chapters(cf), "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == [1000]
            text = cf.read_text(encoding="utf-8")
            assert "BAD_FRAME_OVERRIDE=" not in text
            assert "GOOD_FRAME_OVERRIDE=" not in text

            time.sleep(0.30)
            ov_good, last_good, dbg_good = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1300",
                fids=fids,
                sigs=sigs,
                overrides=ov_bad,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
                mark_mode="good",
                last_click_event=last_bad,
            )
            assert ov_good.get(1000) == "good", dbg_good
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=ov_good,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
            )
            ch = vhs_tuner._find_chapter(vhs_tuner.parse_ffmetadata_chapters(cf), "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == []
            text = cf.read_text(encoding="utf-8")
            assert "BAD_FRAME_OVERRIDE=" not in text
            assert "GOOD_FRAME_OVERRIDE=" not in text

            time.sleep(0.30)
            ov_clear, _last_clear, dbg_clear = vhs_tuner.apply_manual_click_override(
                raw_click="1000:1600",
                fids=fids,
                sigs=sigs,
                overrides=ov_good,
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
                mark_mode="clear",
                last_click_event=last_good,
            )
            assert 1000 not in ov_clear, dbg_clear
            _p, _n = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides=ov_clear,
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=1.0, bp=10.0,
            )
            text = cf.read_text(encoding="utf-8")
            assert "BAD_FRAME_OVERRIDE=" not in text
            assert "GOOD_FRAME_OVERRIDE=" not in text
        finally:
            vhs_tuner.METADATA_DIR = old_meta
    print("Test vhs_tuner manual click mark modes (bad/good/clear): PASSED.")


def test_vhs_tuner_auto_and_manual_persist_to_bad_frames():
    print("Testing vhs_tuner auto + manual persistence to chapters BAD_FRAMES...")
    import tempfile
    import vhs_tuner

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="1005,1007")
            fids = [1000, 1001, 1002]
            sigs = {
                "chroma": np.array([0.0, 10.0, 0.0], dtype=np.float64),
                "noise": np.zeros(3, dtype=np.float64),
                "tear": np.zeros(3, dtype=np.float64),
                "wave": np.zeros(3, dtype=np.float64),
            }
            # Manual bad at frame 1000; auto should mark frame 1001 bad at threshold 0.
            _path, _count = vhs_tuner._persist_visible_bad_frames(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=fids,
                sigs=sigs,
                overrides={1000: "bad"},
                wc=1.0, wn=0.0, wt=0.0, ww=0.0,
                tm="value", ik=3.5, tv=0.0, bp=10.0,
            )
            chapters = vhs_tuner.parse_ffmetadata_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            out = set(int(x) for x in ch.get("bad_frames", []))
            # Existing unsampled values preserved + manual + auto (global IDs).
            assert {1000, 1001, 1005, 1007}.issubset(out), (
                f"persisted BAD_FRAMES missing expected values: {sorted(out)}"
            )
        finally:
            vhs_tuner.METADATA_DIR = old_meta

    print("Test vhs_tuner auto + manual persistence to chapters BAD_FRAMES: PASSED.")


def test_vhs_tuner_persist_loaded_frame_set_mode():
    print("Testing vhs_tuner BAD_FRAMES persistence from loaded frame set only...")
    import tempfile
    import vhs_tuner

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = _write_unit_chapters_ffmetadata(root, bad_csv="")

            path, count, analyzed, err = vhs_tuner.persist_bad_frames_for_chapter(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
                fids=[1000, 1001],
                sigs={
                    "chroma": np.array([0.0, 10.0], dtype=np.float64),
                    "noise": np.array([0.0, 0.0], dtype=np.float64),
                    "tear": np.array([0.0, 0.0], dtype=np.float64),
                    "wave": np.array([0.0, 0.0], dtype=np.float64),
                },
                overrides={1000: "bad"},
                wc=1.0,
                wn=0.0,
                wt=0.0,
                ww=0.0,
                tm="value",
                ik=3.5,
                tv=0.0,
                bp=10.0,
                progress=None,
            )
            assert not err, err
            assert path == cf
            assert analyzed == 2
            assert count == 2

            chapters = vhs_tuner.parse_ffmetadata_chapters(cf)
            ch = vhs_tuner._find_chapter(chapters, "Unit Chapter")
            assert ch is not None
            assert ch.get("bad_frames", []) == [1000, 1001]
        finally:
            vhs_tuner.METADATA_DIR = old_meta

    print("Test vhs_tuner BAD_FRAMES persistence from loaded frame set only: PASSED.")


def test_vhs_tuner_chapter_bad_overrides_half_open_range():
    print("Testing vhs_tuner ignores persisted override metadata lines...")
    import tempfile
    import vhs_tuner

    old_meta = vhs_tuner.METADATA_DIR
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vhs_tuner.METADATA_DIR = root
        try:
            cf = root / "unit_archive" / "chapters.ffmetadata"
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(
                ";FFMETADATA1\n"
                "[CHAPTER]\n"
                "TIMEBASE=1001/30000\n"
                "START=1000\n"
                "END=1100\n"
                "TITLE=Unit Chapter\n"
                "BAD_FRAME_OVERRIDE=1099,1100\n",
                encoding="utf-8",
            )
            out = vhs_tuner._chapter_bad_overrides(
                archive="unit_archive",
                chapter_title="Unit Chapter",
                ch_start=1000,
                ch_end=1100,
            )
            assert out == {}
        finally:
            vhs_tuner.METADATA_DIR = old_meta

    print("Test vhs_tuner ignores persisted override metadata lines: PASSED.")


def test_update_chapter_bad_frames_preserves_untouched_chapters():
    print("Testing BAD_FRAMES updates preserve untouched chapter blocks...")
    import tempfile
    from common import update_chapter_bad_frames_in_ffmetadata, parse_chapters

    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "chapters.ffmetadata"
        cf.write_text(
            ";FFMETADATA1\n"
            "[CHAPTER]\n"
            "TIMEBASE=1001/30000\n"
            "START=0\n"
            "END=100\n"
            "TITLE=Chap A\n"
            "BAD_FRAMES=1,2\n"
            "[CHAPTER]\n"
            "TIMEBASE=1001/30000\n"
            "START=100\n"
            "END=200\n"
            "TITLE=Chap B\n"
            "BAD_FRAMES=3,4\n",
            encoding="utf-8",
        )
        touched = update_chapter_bad_frames_in_ffmetadata(cf, {"Chap A": [9, 10]})
        assert touched == 1

        _ffm, chapters = parse_chapters(cf)
        by_title = {str(ch.get("title", "")).strip(): str(ch.get("bad_frames", "")).strip() for ch in chapters}
        assert by_title.get("Chap A") == "9,10"
        assert by_title.get("Chap B") == "3,4"

    print("Test BAD_FRAMES update preserves untouched chapter blocks: PASSED.")


def test_update_chapter_bad_frames_omits_empty_line():
    print("Testing BAD_FRAMES empty updates remove BAD_FRAMES line...")
    import tempfile
    from common import update_chapter_bad_frames_in_ffmetadata

    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "chapters.ffmetadata"
        cf.write_text(
            ";FFMETADATA1\n"
            "[CHAPTER]\n"
            "TIMEBASE=1001/30000\n"
            "START=0\n"
            "END=100\n"
            "TITLE=Chap A\n"
            "BAD_FRAMES=1,2\n",
            encoding="utf-8",
        )
        touched = update_chapter_bad_frames_in_ffmetadata(cf, {"Chap A": []})
        assert touched == 1
        text = cf.read_text(encoding="utf-8")
        assert "BAD_FRAMES=" not in text

    print("Test BAD_FRAMES empty updates remove BAD_FRAMES line: PASSED.")


def test_vhs_tuner_ui_defaults_and_controls():
    print("Testing vhs_tuner UI defaults and control layout...")
    src = (ROOT / "vhs_tuner.py").read_text(encoding="utf-8", errors="ignore")

    assert 'n_sl = gr.Slider(20, 10000, value=400, step=10, label="n")' in src
    assert 'context_sl = gr.Slider(0, 200, value=10, step=1, label="Frames Around Bad")' in src
    assert 'strict_sampling_cb = gr.Checkbox(label="Strict Sampling", value=True)' in src
    assert 'with gr.Tab("Frames", id="frames-tab"):' in src
    assert 'apply_btn = gr.Button("Apply", variant="primary")' in src
    assert "apply_btn.click(on_save_bad_frames, _SAVE_INS, [status_md])" in src
    assert 'choices=["toggle", "bad", "good", "clear"]' in src
    assert "if not bool(strict_sampling):" in src
    assert 'with gr.Accordion("Range & Sample", open=False):' in src
    assert 'with gr.Accordion("Manual Marking", open=False):' in src
    assert 'with gr.Accordion("Signal Weights", open=False):' in src
    assert 'with gr.Accordion("Threshold", open=False):' in src
    assert 'with gr.Accordion("Grid", open=False):' in src

    assert "Apply & Regenerate" not in src
    assert "fstep_sl  =" not in src

    print("Test vhs_tuner UI defaults and control layout: PASSED.")

def main():
    print("Running tests...")
    test_step_4_generate_archive_metadata()
    test_step_6_make_videos()
    test_step_6_title_filter_and_rebuild()
    test_step_6_badframe_sidecar_mapping()
    test_step_6_badframe_repair_injection_and_comment()
    test_step_6_badframe_split_strategy_logic_paths()
    test_step_6_make_create_avs_includes_chapter_bounds()
    test_step_6_real_badframes_do_not_pick_bad_sources()
    test_step_6_frame_quality_ingest_exact_archive01()
    test_step_6_proxy_badframes_overlay_e2e()
    test_step_6_qtgmc_freezeframe_long_e2e()
    test_step_drive_checksums()
    test_sha3_generate_and_verify()
    test_blake3_verify_only()
    test_vhs_tuner_toggle_override_cycle()
    test_vhs_tuner_manual_click_persists_bad_frames()
    test_vhs_tuner_click_dedupe_prevents_double_toggle()
    test_vhs_tuner_manual_click_modes_bad_and_good()
    test_vhs_tuner_auto_and_manual_persist_to_bad_frames()
    test_vhs_tuner_persist_loaded_frame_set_mode()
    test_vhs_tuner_chapter_bad_overrides_half_open_range()
    test_update_chapter_bad_frames_preserves_untouched_chapters()
    test_update_chapter_bad_frames_omits_empty_line()
    test_vhs_tuner_ui_defaults_and_controls()

if __name__ == "__main__":
    main()
