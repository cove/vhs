import sys

from vhs_pipeline.convert import convert_avi_to_archive


def main(argv=None):
    files = list(argv if argv is not None else sys.argv[1:])
    if not files:
        print("Usage: python step_2_convert_avi_to_mkv.py video1.avi video2.avi ...")
        return 1
    convert_avi_to_archive(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

