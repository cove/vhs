#!/usr/bin/env python3.11
#
# Legacy entrypoint for chapter comparison generation.
# Preferred entrypoint: python vhs.py compare
#
import sys

from vhs_pipeline.compare import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

