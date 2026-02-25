#
# Legacy entrypoint for drive checksum generation.
# Preferred entrypoint: python vhs.py checksum drive
#
import sys

from vhs_pipeline.checksum import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

