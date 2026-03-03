#!/usr/bin/env python3.11
#
# Sets up a Python virtual environment for the project based on the host OS
# (Windows, macOS, or Linux), and installs all dependencies from requirements.txt.
# Automatically chooses the correct venv directory and Python binary per platform.
#
import subprocess
import sys
import venv
import os
import shutil
import tarfile
from pathlib import Path

BASE = Path(__file__).parent.resolve()

if sys.platform == "win32":
    VENV_DIR = BASE / ".venv"
elif sys.platform == "darwin":
    VENV_DIR = BASE / "venv-mac"
elif sys.platform.startswith("linux"):
    VENV_DIR = BASE / "venv-linux"
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

REQ_FILE = BASE / "requirements.txt"
BIN_DIR = BASE / "bin"

LINUX_FFMPEG_ARCHIVE = BIN_DIR / "ffmpeg-release-amd64-static.tar.xz"

def create_venv():
    if VENV_DIR.exists():
        print("Virtual environment already exists:", VENV_DIR)
        return

    print("Creating virtual environment:", VENV_DIR)
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)
    print("Done.")

def get_python_bin():
    """Return path to python inside venv."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def install_requirements():
    if not REQ_FILE.exists():
        print("No requirements.txt found - skipping install.")
        return

    python_bin = get_python_bin()
    print("Installing requirements from requirements.txt ...")
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", str(REQ_FILE)])
    print("Python requirements installed.")
    print("All requirements installed.")

def _needs_refresh(target_path, source_paths):
    target = Path(target_path)
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    for src in source_paths:
        src_path = Path(src)
        if src_path.exists() and src_path.stat().st_mtime > target_mtime:
            return True
    return False

def _safe_chmod(path, mode):
    try:
        os.chmod(path, mode)
    except PermissionError:
        # Some shared/mounted filesystems do not support chmod; continue anyway.
        pass

def _extract_tar_member(archive_path, member_suffix, output_path, executable=False):
    archive = Path(archive_path)
    out = Path(output_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    with tarfile.open(archive, mode="r:*") as tf:
        member = None
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(member_suffix):
                member = m
                break
        if member is None:
            raise RuntimeError(f"Member '{member_suffix}' not found in archive: {archive.name}")

        src = tf.extractfile(member)
        if src is None:
            raise RuntimeError(f"Unable to extract member '{member.name}' from: {archive.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if executable:
        _safe_chmod(out, 0o755)
    else:
        _safe_chmod(out, 0o644)

def install_linux_binary_archives():
    if not sys.platform.startswith("linux"):
        return

    if not LINUX_FFMPEG_ARCHIVE.exists():
        print(f"Linux ffmpeg archive missing from bin/: {LINUX_FFMPEG_ARCHIVE.name}")
        print("Skipping binary extraction.")
        return

    ffmpeg_bin = BIN_DIR / "ffmpeg"
    ffprobe_bin = BIN_DIR / "ffprobe"

    if _needs_refresh(ffmpeg_bin, [LINUX_FFMPEG_ARCHIVE]):
        print(f"Extracting ffmpeg from {LINUX_FFMPEG_ARCHIVE.name} ...")
        _extract_tar_member(LINUX_FFMPEG_ARCHIVE, "/ffmpeg", ffmpeg_bin, executable=True)

    if _needs_refresh(ffprobe_bin, [LINUX_FFMPEG_ARCHIVE]):
        print(f"Extracting ffprobe from {LINUX_FFMPEG_ARCHIVE.name} ...")
        _extract_tar_member(LINUX_FFMPEG_ARCHIVE, "/ffprobe", ffprobe_bin, executable=True)

    print("Linux ffmpeg binaries extracted to bin/.")
    if shutil.which("mediainfo") is None:
        print(
            "WARNING: 'mediainfo' not found on PATH. Install MediaInfo CLI "
            "(e.g. sudo apt-get install mediainfo) or set MEDIAINFO_BIN."
        )
    else:
        print("Found system 'mediainfo' on PATH.")

def main():
    create_venv()
    install_linux_binary_archives()
    install_requirements()
    print("Environment setup complete.")

if __name__ == "__main__":
    main()
