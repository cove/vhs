#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# Try two possible locations for b3sum
B3SUM = BASE_DIR / "b3sum_windows_x64_bin.exe"
if not B3SUM.exists():
    B3SUM = BASE_DIR / "bin" / "b3sum_windows_x64_bin.exe"

MEDIAINFO = BASE_DIR / "bin" / "mediainfo.exe"

# -------------------------------------------------
# Check tools exist
# -------------------------------------------------
if not B3SUM.exists():
    print("ERROR: b3sum not found!")
    print(f"   Looked in:")
    print(f"     {BASE_DIR / 'b3sum_windows_x64_bin.exe'}")
    print(f"     {BASE_DIR / 'bin' / 'b3sum_windows_x64_bin.exe'}")
    input("Press Enter to exit...")
    sys.exit(1)

if not MEDIAINFO.exists():
    print(f"ERROR: mediainfo.exe not found at {MEDIAINFO}")
    sys.exit(1)

# -------------------------------------------------
# Main work
# -------------------------------------------------
output_file = BASE_DIR / "00-manifest-blake3sums.txt"
if output_file.exists():
    output_file.unlink()  # delete old manifest

mkv_files = list(BASE_DIR.glob("*.mkv"))
if not mkv_files:
    print("No .mkv files found in this folder.")
    sys.exit(0)

print(f"Found {len(mkv_files)} .mkv file(s)\n")

for mkv in mkv_files:
    print(f"Processing: {mkv.name}")

    # 1. Generate BLAKE3 hash
    result = subprocess.run([str(B3SUM), str(mkv)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ERROR: b3sum failed on {mkv.name}")
        sys.exit(1)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(result.stdout)

    # 2. Generate MediaInfo text report
    info_file = mkv.with_name(mkv.stem + "_mediainfo.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        subprocess.run([str(MEDIAINFO), "--Output=Text", str(mkv)], stdout=f, check=True)

    print("   Done\n")

print(f"All done!")
print(f"→ BLAKE3 manifest: {output_file.name}")
