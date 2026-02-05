#
# Computes SHA3-256 checksums for all files on a drive, ignoring common OS-generated files and directories,
# and writes the results to a manifest for data integrity verification.
#
import hashlib
from common import *
from pathlib import Path

SHA3_DRIVE_CHECKSUM_FILE = ARCHIVE_DIR / "00-drive-manifest-sha3-256sums.txt"

def should_ignore(path: Path) -> bool:
    """
    Return True if the file or directory represented by `path`
    should be ignored based on common OS-generated names.
    """
    name = path.name

    exact = {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        ".Spotlight-V100",
        ".Trashes",
        ".fseventsd",
        ".TemporaryItems",
        ".VolumeIcon.icns",
        ".AppleDouble",
        ".AppleDesktop",
        ".android_secure",
        "LOST.DIR",
        ARCHIVE_CHECKSUM_FILE.name,
        DRIVE_CHECKSUM_FILE.name,
        SHA3_DRIVE_CHECKSUM_FILE.name,
        "venv-mac",
        "venv-win",
        ".git",
        ".gitignore",
        "__pycache__",
    }

    dirs = {
        "$RECYCLE.BIN",
        "System Volume Information",
        "Android",
        ".thumbnails",
    }

    prefix = (
        "._",       # AppleDouble
        ".Trash",   # .Trash, .Trash-1000, etc.
    )

    if name in exact:
        return True

    if name in dirs:
        return True

    if any(name.startswith(p) for p in prefix):
        return True

    return False

def sha3sum_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha3_256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_checksums(root_dir, manifest_path):
    root_dir = Path(root_dir)
    manifest_path = Path(manifest_path)
    manifest_path.unlink(missing_ok=True)

    old_cwd = os.getcwd()
    os.chdir(root_dir)
    try:
        for file_path in root_dir.rglob("*"):
            if should_ignore(file_path):
                continue

            if file_path.is_file():
                digest = sha3sum_file(file_path)
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(f"{digest}  {file_path}\n")

        print("Checksums written to:", manifest_path)
    finally:
        os.chdir(old_cwd)

def main():
    compute_checksums(DRIVE_DIR, SHA3_DRIVE_CHECKSUM_FILE)

    print("Checksum manifest: ", SHA3_DRIVE_CHECKSUM_FILE)
    print("All done.")

if __name__ == "__main__":
    main()
