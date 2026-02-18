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
    assert "FreezeFrame(20,20,19)" in out
    assert "FreezeFrame(6,8,5)" in out
    assert out.find("FreezeFrame(20,20,19)") < out.find("FreezeFrame(6,8,5)")

    out_override = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(10, 12, 20), (30, 30, None)]
    )
    assert "FreezeFrame(30,30,29)" in out_override
    assert "FreezeFrame(10,12,20)" in out_override

    out_invalid_override = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(6, 8, 7)]
    )
    assert "FreezeFrame(6,8,5)" in out_invalid_override

    out_nearby_auto = step_6_make_videos.build_badframe_prefilter_lines(
        bad_repair_ranges=[(0, 0, None), (10, 10, None)]
    )
    # Nearby auto-picked ranges should not switch from future-looking to past-looking.
    assert "FreezeFrame(10,10,11)" in out_nearby_auto
    assert "FreezeFrame(10,10,9)" not in out_nearby_auto

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
        )
        assert "chapter_start_frame = 100" in script
        assert "chapter_end_frame = 200" in script
        assert "FreezeFrame(4,5,3)" in script
    finally:
        tmp_filter.unlink(missing_ok=True)

    print("Test step_6_make_videos AVS generation with chapter bounds: PASSED.")
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
    test_step_drive_checksums()
    test_sha3_generate_and_verify()
    test_blake3_verify_only()

if __name__ == "__main__":
    main()
