#
# Generates SHA3-256 checksums for all files on a drive, ignoring common OS-generated
# files and directories, and writes the results to a manifest for verification.
#
from common import *
from pathlib import Path

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
        LEGACY_ARCHIVE_CHECKSUM_FILE.name,
        LEGACY_DRIVE_CHECKSUM_FILE.name,
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

def compute_checksums(root_dir, manifest_path):
    write_sha3_manifest(root_dir, manifest_path, relative_base=root_dir, ignore_fn=should_ignore)

def main():
    compute_checksums(DRIVE_DIR, DRIVE_CHECKSUM_FILE)
    print("Checksum manifest: ", DRIVE_CHECKSUM_FILE)
    print("All done.")

if __name__ == "__main__":
    main()
