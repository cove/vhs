#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

# CHANGE ONLY IF YOU MOVED IT
B3SUM = r"b3sum_windows_x64_bin.exe"           # in same folder
B3SUM_ALT = r"bin\b3sum_windows_x64_bin.exe"   # or in bin\

MANIFEST = "00-manifest-blake3sums.txt"

# Find b3sum
b3 = None
if Path(B3SUM).exists():
    b3 = B3SUM
elif Path(B3SUM_ALT).exists():
    b3 = B3SUM_ALT

if not b3:
    print("ERROR: b3sum not found!")
    print("   Put b3sum_windows_x64_bin.exe in this folder or in bin\\")
    input("Press Enter to exit...")
    sys.exit(1)

if not Path(MANIFEST).exists():
    print(f"ERROR: {MANIFEST} not found in this folder!")
    input("Press Enter to exit...")
    sys.exit(1)

print(f"Verifying BLAKE3 hashes using: {MANIFEST}")
print("-" * 50)

# Run b3sum -c
result = subprocess.run([b3, "-c", MANIFEST], capture_output=True, text=True)

# Show output
print(result.stdout if result.stdout else result.stderr)

if result.returncode == 0:
    print("ALL FILES VERIFIED — CHECKSUMS MATCH!")
else:
    print("SOME FILES FAILED VERIFICATION!")
    sys.exit(1)

input("Press Enter to close...")
