#
# Computes BLAKE3 checksums for all files on a drive, ignoring common OS-generated files and directories,
# and writes the results to a manifest for data integrity verification.
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
        "venv-mac",
        "venv-win",
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
    root_dir = Path(root_dir)
    manifest_path = Path(manifest_path)
    manifest_path.unlink(missing_ok=True)

    for file_path in root_dir.rglob("*"):
        if should_ignore(file_path):
            continue

        if file_path.is_file():
            r = subprocess.run([str(B3SUM_BIN), str(file_path)], cwd=root_dir, capture_output=True, text=True)
            if r.returncode:
                print(f"  ERROR: b3sum failed for {file_path}: {r.stderr.strip()}")
                sys.exit(r.returncode)

            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(r.stdout)

    print("Checksums written to:", manifest_path)

def main():
    compute_checksums(DRIVE_DIR, DRIVE_CHECKSUM_FILE)

    print("Checksum manifest: ", DRIVE_CHECKSUM_FILE)
    print("All done.")

if __name__ == "__main__":
    main()
