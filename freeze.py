#!/usr/bin/env python3.11
#
# Archives all installed Python packages from the current environment:
# 1. Writes a requirements.txt with exact versions.
# 2. Attempts to download prebuilt wheels for macOS and Windows targets.
# 3. Falls back to source distributions if wheels aren't available.
# 4. Saves all files into the package_archive directory.
#

import subprocess
from pathlib import Path
import sys

BASE = Path(__file__).parent.resolve()
ARCHIVE_DIR = BASE / "package_archive"

# Target wheels (feel free to adjust)
TARGETS = [
    ("macosx_13_0_arm64", "311", "cp311"),
    ("win_amd64",          "311", "cp311"),
]

def run(cmd):
    """Run a command and return True if it succeeds, False otherwise."""
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False

def freeze_packages():
    ARCHIVE_DIR.mkdir(exist_ok=True)
    print("Archiving packages into:", ARCHIVE_DIR)

    # Get requirements
    freeze_output = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )

    # Save requirements.txt
    (BASE / "requirements.txt").write_text(freeze_output)
    print("requirements.txt written.")

    packages = [line.strip() for line in freeze_output.splitlines() if line.strip()]

    for pkg in packages:
        name_only = pkg.split("==")[0]
        print(f"\n=== {name_only} ===")

        # For each OS target
        for platform, pyver, abi in TARGETS:
            print(f"  Trying wheel for {platform}...")

            cmd = [
                sys.executable, "-m", "pip", "download",
                "--dest", str(ARCHIVE_DIR),
                "--platform", platform,
                "--python-version", pyver,
                "--implementation", "cp",
                "--abi", abi,
                "--only-binary=:all:",
                name_only
            ]

            if run(cmd):
                print(f"  [ok] wheel downloaded for {platform}")
                continue

            print(f"  [x] no wheel for {platform}, falling back to source...")

        # Download source distribution (universal)
        src_cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(ARCHIVE_DIR),
            name_only
        ]

        if run(src_cmd):
            print("  [ok] source downloaded")
        else:
            print("  [x] could not download ANY distribution for", name_only)

    print("\nAll packages archived.")

if __name__ == "__main__":
    freeze_packages()
