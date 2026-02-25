#
# Converts U-Matic source files (MOV) to archival MKV using FFV1 video and 24-bit PCM audio.
# For Digital Roots, Albany, CA services. Produces lossless preservation masters.
#
import sys

try:
    from ._bootstrap import ensure_project_root_on_path
except ImportError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from vhs_pipeline.convert import convert_umatic_to_archive


def main(argv=None):
    files = list(argv if argv is not None else sys.argv[1:])
    if not files:
        print("Usage: python step_2_convert_umatic_prores_mov_to_archive.py video1.mov video2.mov ...")
        return 1
    convert_umatic_to_archive(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
