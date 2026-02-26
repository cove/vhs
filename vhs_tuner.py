#!/usr/bin/env python3.11
"""
Legacy VHS tuner module compatibility shim.

The Gradio UI has been removed. Use:
  python vhs.py tuner

This module re-exports core tuner helpers for existing scripts/tests and
can still launch the plain HTML wizard when run directly.
"""

from __future__ import annotations

from apps.plain_html_wizard.server import run as run_plain_wizard
from libs.vhs_tuner_core import *  # noqa: F401,F403


def main() -> int:
    run_plain_wizard(host="0.0.0.0", port=8092)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
