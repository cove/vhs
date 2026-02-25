from __future__ import annotations

from common import ARCHIVE_DIR, DRIVE_DIR, verify_manifest
from vhs_pipeline.convert import (
    convert_avi_to_archive,
    convert_umatic_to_archive,
    embed_metadata_into_archives,
)


def _exit_code(value):
    if value is None:
        return 0
    return int(value)


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
    import step_3_generate_archive_metadata as step_3

    return _exit_code(step_3.main())


def run_verify_archive(argv):
    import step_4_verify_archive as step_4

    manifest, algo = step_4.parse_args(list(argv or []))
    manifest, algo = step_4.resolve_manifest(manifest, algo)
    print(f"Verifying: {manifest}\n")
    return verify_manifest(ARCHIVE_DIR, manifest, algo=algo)


def run_make_proxies():
    import step_5_make_proxies as step_5

    return _exit_code(step_5.main())


def run_make_videos(argv):
    import step_6_make_videos as step_6

    return _exit_code(step_6.main(list(argv or [])))


def run_generate_drive_checksum():
    import step_7_generate_drive_checksum as step_7

    return _exit_code(step_7.main())


def run_verify_drive(argv):
    import step_8_verify_drive_checksum as step_8

    manifest, algo = step_8.parse_args(list(argv or []))
    manifest, algo = step_8.resolve_manifest(manifest, algo)
    print(f"Verifying: {manifest}\n")
    return verify_manifest(DRIVE_DIR, manifest, algo=algo)


def run_make_comparisons(argv):
    import step_14_make_original_chapter_comparisons as step_14

    return _exit_code(step_14.main(list(argv or [])))

