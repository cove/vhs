from __future__ import annotations

import sys


def run_render(argv=None):
    from vhs_pipeline import render_pipeline

    return int(render_pipeline.main(list(argv or [])) or 0)


def main(argv=None):
    return run_render(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
