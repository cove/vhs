#
# Legacy entrypoint for drive checksum verification.
# Preferred entrypoint: python vhs.py verify drive
#
import sys

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from vhs_pipeline.checksum import verify_drive


def main(argv=None):
    return verify_drive(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
