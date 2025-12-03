"""
Generate BLAKE3 Checksums and MediaInfo Reports for MKV Files

This script processes all MKV files in the Archive directory by:
- Computing BLAKE3 checksums for integrity verification.
- Writing a manifest file "00-manifest-blake3sums.txt" with all BLAKE3 hashes.
- Generating MediaInfo text reports for each MKV file.

Requirements:
- b3sum executable must be available at bin/b3sum_windows_x64_bin.exe
- MediaInfo executable must be available at bin/mediainfo.exe

Output:
- 00-manifest-blake3sums.txt containing all BLAKE3 hashes.
- <filename>_mediainfo.txt for each MKV file.
"""

import subprocess
import sys
import glob
from pathlib import Path

BASE = Path(__file__).parent.resolve()
ARCHIVE = BASE / ".." / "Archive"
B3SUM = BASE / "bin" / "b3sum_windows_x64_bin.exe"
MEDIAINFO = BASE / "bin" / "mediainfo.exe"

manifest = ARCHIVE / "00-manifest-blake3sums.txt"
manifest.unlink(missing_ok=True)

files = glob.glob("*.mkv", root_dir=ARCHIVE)
if not files:
    print("No .mkv files found.")
    sys.exit(0)

for fn in files:
    print("Processing:", fn)
    path = Path(fn)

    # BLAKE3 checksum
    r = subprocess.run([str(B3SUM), fn], cwd=ARCHIVE, capture_output=True, text=True)
    if r.returncode:
        print("  ERROR: b3sum failed:", fn)
        sys.exit(1)

    # Append checksum to manifest
    if not manifest.exists():
        manifest.write_text(r.stdout, encoding="utf-8")
    else:
        manifest.write_text(manifest.read_text() + r.stdout, encoding="utf-8")

    # MediaInfo report
    info = ARCHIVE / f"{path.stem}_mediainfo.txt"
    with open(info, "w", encoding="utf-8") as out:
        subprocess.run([str(MEDIAINFO), "--Output=Text", fn], cwd=ARCHIVE, stdout=out, check=True)

print("All done!")
print("→ BLAKE3 manifest:", manifest.name)
