#
# Legacy entrypoint for archive metadata generation.
# Preferred entrypoint: python vhs.py metadata build
#
import sys

from vhs_pipeline.metadata import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

