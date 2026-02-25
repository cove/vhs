#
# Legacy entrypoint for archive checksum verification.
# Preferred entrypoint: python vhs.py verify archive
#
import sys

from vhs_pipeline.checksum import verify_archive


def main(argv=None):
    return verify_archive(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

