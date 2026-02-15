#
# Verifies all files on the drive against a checksum manifest (SHA3-256 or legacy BLAKE3).
#
from common import *

def parse_args(argv):
    algo = "auto"
    manifest = None

    for arg in argv:
        if arg in ("--blake3", "--b3"):
            algo = "blake3"
        elif arg in ("--sha3", "--sha3-256"):
            algo = "sha3"
        else:
            manifest = arg

    return manifest, algo

def resolve_manifest(manifest, algo):
    if manifest:
        return Path(manifest), algo

    if DRIVE_CHECKSUM_FILE.exists():
        return DRIVE_CHECKSUM_FILE, algo

    if LEGACY_DRIVE_CHECKSUM_FILE.exists():
        return LEGACY_DRIVE_CHECKSUM_FILE, "blake3" if algo == "auto" else algo

    return DRIVE_CHECKSUM_FILE, algo

def verify_checksums(root_dir, manifest_path, algo):
    print(f"Verifying: {manifest_path}\n")
    return verify_manifest(root_dir, manifest_path, algo=algo)

def main():
    manifest, algo = parse_args(sys.argv[1:])
    manifest, algo = resolve_manifest(manifest, algo)
    rc = verify_checksums(DRIVE_DIR, manifest, algo)

    print("Verify manifest: ", manifest)
    print("All done.")
    sys.exit(rc)

if __name__ == "__main__":
    main()
