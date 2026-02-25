#
# Legacy entrypoint for proxy generation.
# Preferred entrypoint: python vhs.py proxy
#
import sys

from vhs_pipeline.proxy import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

