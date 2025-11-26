import subprocess, sys, glob
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

    # BLAKE3
    r = subprocess.run([str(B3SUM), fn], cwd=ARCHIVE, capture_output=True, text=True)
    if r.returncode:
        print("  ERROR: b3sum failed:", fn)
        sys.exit(1)
    manifest.write_text(r.stdout, encoding="utf-8") if not manifest.exists() else manifest.write_text(
        manifest.read_text() + r.stdout, encoding="utf-8"
    )

    # MediaInfo
    info = ARCHIVE / f"{path.stem}_mediainfo.txt"
    with open(info, "w", encoding="utf-8") as out:
        subprocess.run([str(MEDIAINFO), "--Output=Text", fn], cwd=ARCHIVE, stdout=out, check=True)

print("All done!")
print("→ BLAKE3 manifest:", manifest.name)
