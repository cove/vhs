#
# Legacy entrypoint for drive checksum verification.
# Preferred entrypoint: python vhs.py verify drive
#
import sys

from vhs_pipeline.checksum import verify_drive


def main(argv=None):
    return verify_drive(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

