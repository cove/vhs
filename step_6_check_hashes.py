#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
ARCHIVE_DIR = BASE_DIR / ".." / "Archive"

B3SUM = r"bin\b3sum_windows_x64_bin.exe"
MANIFEST = ARCHIVE_DIR / "00-manifest-blake3sums.txt"

if not B3SUM:
    print("ERROR: b3sum not found!")
    print("   Put b3sum_windows_x64_bin.exe in this folder or in bin\\")
    sys.exit(1)

if not Path(MANIFEST).exists():
    print(f"ERROR: {MANIFEST} not found in this folder!")
    sys.exit(1)

print(f"Verifying BLAKE3 hashes using: {MANIFEST}")
print("-" * 50)

result = subprocess.run([B3SUM, "-c", MANIFEST],
                        cwd=ARCHIVE_DIR, 
                        capture_output=True, 
                        text=True)

print(result.stdout if result.stdout else result.stderr)

if result.returncode == 0:
    print("ALL FILES VERIFIED — CHECKSUMS MATCH!")
else:
    print("SOME FILES FAILED VERIFICATION!")
    sys.exit(1)
