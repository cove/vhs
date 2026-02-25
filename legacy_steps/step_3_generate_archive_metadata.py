#
# Legacy entrypoint for archive metadata generation.
# Preferred entrypoint: python vhs.py metadata build
#
import sys

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from vhs_pipeline.metadata import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
