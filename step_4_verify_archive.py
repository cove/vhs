#
# Verifies all files in the archive against the BLAKE3 checksum manifest to ensure data integrity.
#
from common import *

def main():
    print(f"Verifying: {ARCHIVE_CHECKSUM_FILE}\n")
    r = subprocess.run([str(B3SUM_BIN), "-c", str(ARCHIVE_CHECKSUM_FILE)], cwd=ARCHIVE_DIR, capture_output=True, text=True)
    print(r.stdout or r.stderr)

    if r.returncode == 0:
        print("ALL FILES VERIFIED — CHECKSUMS MATCH!")
    else:
        print("SOME FILES FAILED VERIFICATION!")

    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
