#!/usr/bin/env python3.11
#
# Legacy entrypoint for chapter comparison generation.
# Preferred entrypoint: python vhs.py compare
#
import sys

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from vhs_pipeline.compare import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
