import subprocess, sys
from pathlib import Path

BASE = Path(__file__).parent.resolve()
ARCHIVE = BASE / ".." / "Archive"
B3SUM = Path("bin/b3sum_windows_x64_bin.exe")
MANIFEST = ARCHIVE / "00-manifest-blake3sums.txt"

if not B3SUM.exists():
    print("ERROR: b3sum not found in bin/")
    sys.exit(1)

if not MANIFEST.exists():
    print(f"ERROR: manifest not found: {MANIFEST}")
    sys.exit(1)

print(f"Verifying: {MANIFEST}\n")

r = subprocess.run([str(B3SUM), "-c", str(MANIFEST)],
                   cwd=ARCHIVE, capture_output=True, text=True)

print(r.stdout or r.stderr)

if r.returncode == 0:
    print("ALL FILES VERIFIED — CHECKSUMS MATCH!")
else:
    print("SOME FILES FAILED VERIFICATION!")
    sys.exit(1)
