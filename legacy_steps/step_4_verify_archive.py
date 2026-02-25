#
# Legacy entrypoint for archive checksum verification.
# Preferred entrypoint: python vhs.py verify archive
#
import sys

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from vhs_pipeline.checksum import verify_archive


def main(argv=None):
    return verify_archive(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
