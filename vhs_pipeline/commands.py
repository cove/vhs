from __future__ import annotations

from vhs_pipeline.checksum import generate_drive_checksum, verify_archive, verify_drive
from vhs_pipeline.compare import run_comparisons
from vhs_pipeline.convert import (
    convert_avi_to_archive,
    convert_umatic_to_archive,
    embed_metadata_into_archives,
)
from vhs_pipeline.metadata import generate_archive_metadata
from vhs_pipeline.proxy import make_proxies
from vhs_pipeline.render import run_render
from apps.plain_html_wizard.server import run as run_tuner_server


def run_convert_avi(paths):
    convert_avi_to_archive(paths)
    return 0


def run_convert_umatic(paths):
    convert_umatic_to_archive(paths)
    return 0


def run_embed_metadata(paths):
    embed_metadata_into_archives(paths)
    return 0


def run_generate_archive_metadata():
    return int(generate_archive_metadata() or 0)


def run_verify_archive(argv):
    return int(verify_archive(argv) or 0)


def run_make_proxies():
    return int(make_proxies() or 0)


def run_make_videos(argv):
    return int(run_render(argv) or 0)


def run_generate_drive_checksum():
    return int(generate_drive_checksum() or 0)


def run_verify_drive(argv):
    return int(verify_drive(argv) or 0)


def run_make_comparisons(argv):
    return int(run_comparisons(argv) or 0)


def run_tuner(host: str = "0.0.0.0", port: int = 8092):
    run_tuner_server(host=host, port=int(port))
    return 0

