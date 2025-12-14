import shutil
import os
os.environ["TEST_ENV"] = "1"
from common import *
TESTDATA_DIR = BASE / "test" / "test_data"
os.environ["PYTHONPATH"] = str(BASE)

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
    del sys.modules['step_3_generate_archive_metadata']

def test_step_6_make_videos():
    print("Testing step_6_make_videos.py...")
    shutil.copy(TESTDATA_DIR / "test_01_archive.mkv", ARCHIVE_DIR / "test_01_archive.mkv")
    import step_6_make_videos
    assert step_6_make_videos.main() is None
    assert (CLIPS_DIR / "Test Video 01.mp4").stat().st_size > 100
    print("Test step_6_make_videos.py: PASSED.")
    (CLIPS_DIR / "Test Video 01.mp4").unlink()
    del sys.modules['step_6_make_videos']

def test_step_drive_checksums():
    print("Testing step_7_generate_drive_checksum.py...")
    import step_7_generate_drive_checksum
    assert step_7_generate_drive_checksum.main() is None
    import step_8_verify_drive_checksum
    assert step_10_verify_drive_checksum.main() is None
    print("Test step_drive_checksums: PASSED.")
    DRIVE_CHECKSUM_FILE.unlink()
    del sys.modules['step_9_generate_drive_checksum']
    del sys.modules['step_10_verify_drive_checksum']

def main():
    print("Running tests...")
    test_step_4_generate_archive_metadata()
    test_step_6_make_videos()
    test_step_drive_checksums()

if __name__ == "__main__":
    main()
