from __future__ import annotations

import sys


def run_render(argv=None):
    import step_6_make_videos as step_6

    return int(step_6.main(list(argv or [])) or 0)


def main(argv=None):
    return run_render(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

