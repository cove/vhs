#
# Verifies all files on the drive against the BLAKE3 checksum manifest to ensure integrity.
#

from common import *

def verify_checksums(root_dir, manifest_path):
    print(f"Verifying: {DRIVE_CHECKSUM_FILE}\n")
    r = subprocess.run([str(B3SUM_BIN), "-c", str(DRIVE_CHECKSUM_FILE)], cwd=DRIVE_DIR, capture_output=True, text=True)
    print(r.stdout or r.stderr)

    if r.returncode == 0:
        print("ALL FILES VERIFIED — CHECKSUMS MATCH!")
    else:
        print("SOME FILES FAILED VERIFICATION!")

    sys.exit(r.returncode)

def main():
    verify_checksums(DRIVE_DIR, DRIVE_CHECKSUM_FILE)

    print("Verify manifest: ", DRIVE_CHECKSUM_FILE)
    print("All done.")

if __name__ == "__main__":
    main()
