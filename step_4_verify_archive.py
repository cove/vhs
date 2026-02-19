#
# Verifies all files in the archive against a checksum manifest (SHA3-256 or legacy BLAKE3).
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

    if ARCHIVE_CHECKSUM_FILE.exists():
        return ARCHIVE_CHECKSUM_FILE, algo

    if LEGACY_ARCHIVE_CHECKSUM_FILE.exists():
        return LEGACY_ARCHIVE_CHECKSUM_FILE, "blake3" if algo == "auto" else algo

    return ARCHIVE_CHECKSUM_FILE, algo

def main():
    manifest, algo = parse_args(sys.argv[1:])
    manifest, algo = resolve_manifest(manifest, algo)

    print(f"Verifying: {manifest}\n")
    rc = verify_manifest(ARCHIVE_DIR, manifest, algo=algo)
    sys.exit(rc)

if __name__ == "__main__":
    main()
