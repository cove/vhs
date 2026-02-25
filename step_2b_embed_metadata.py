#
# Injects ffmetadata (global tags + chapters) into existing archive MKV files
# without re-encoding. Safe to run multiple times.
#
import sys

from vhs_pipeline.convert import embed_metadata_into_archives


def main(paths):
    embed_metadata_into_archives(paths)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python step_2b_embed_metadata.py archive1.mkv archive2.mkv ...")
        raise SystemExit(1)
    main(args)

