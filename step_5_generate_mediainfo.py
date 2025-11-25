import glob
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
ARCHIVE_DIR = BASE_DIR / ".." / "Archive"
B3SUM = BASE_DIR / "bin" / "b3sum_windows_x64_bin.exe"
MEDIAINFO = BASE_DIR / "bin" / "mediainfo.exe"

output_file = ARCHIVE_DIR / "00-manifest-blake3sums.txt"
if output_file.exists():
    output_file.unlink()

mkv_files = list(glob.glob("*.mkv", root_dir=ARCHIVE_DIR))
if not mkv_files:
    print("No .mkv files found.")
    sys.exit(0)

for mkv in mkv_files:
    print(f"Processing: {mkv}")

    # 1. Generate BLAKE3 hash
    result = subprocess.run([str(B3SUM), str(mkv)], 
                            cwd=ARCHIVE_DIR,
                            capture_output=True, 
                            text=True)
    if result.returncode != 0:
        print(f"   ERROR: b3sum failed on {mkv.name}")
        sys.exit(1)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(result.stdout)

    info_file = ARCHIVE_DIR / Path(mkv).with_name(Path(mkv).stem + "_mediainfo.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        subprocess.run([str(MEDIAINFO), "--Output=Text", str(mkv)], 
                       cwd=ARCHIVE_DIR,
                       stdout=f, check=True)

print(f"All done!")
print(f"→ BLAKE3 manifest: {output_file.name}")
