from __future__ import annotations

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vhs.py",
        description=(
            "Unified command surface for VHS archive workflows. "
            "Legacy step_*.py scripts remain supported."
        ),
    )
    subparsers = parser.add_subparsers(dest="group", required=True)

    convert_parser = subparsers.add_parser("convert", help="Archive conversion commands")
    convert_sub = convert_parser.add_subparsers(dest="convert_kind", required=True)
    convert_avi = convert_sub.add_parser("avi", help="Convert AVI capture(s) to archive MKV")
    convert_avi.add_argument("files", nargs="+", help="Input AVI file(s)")
    convert_umatic = convert_sub.add_parser(
        "umatic",
        help="Convert U-matic/ProRes source file(s) to archive MKV",
    )
    convert_umatic.add_argument("files", nargs="+", help="Input MOV (or similar) file(s)")

    metadata_parser = subparsers.add_parser("metadata", help="Metadata commands")
    metadata_sub = metadata_parser.add_subparsers(dest="metadata_kind", required=True)
    metadata_sub.add_parser("build", help="Generate archive metadata outputs and checksums")
    metadata_embed = metadata_sub.add_parser(
        "embed",
        help="Embed ffmetadata into existing archive MKV(s) without re-encoding",
    )
    metadata_embed.add_argument("files", nargs="+", help="Archive MKV file(s)")

    subparsers.add_parser("proxy", help="Generate proxy MP4 files")
    subparsers.add_parser("render", help="Run delivery render pipeline (forwards args to step_6)")
    subparsers.add_parser(
        "compare",
        help="Create side-by-side original vs processed chapter comparisons (forwards args to step_14)",
    )

    verify_parser = subparsers.add_parser("verify", help="Checksum verification commands")
    verify_sub = verify_parser.add_subparsers(dest="verify_kind", required=True)
    verify_sub.add_parser("archive", help="Verify archive checksum manifest")
    verify_sub.add_parser("drive", help="Verify drive checksum manifest")

    checksum_parser = subparsers.add_parser("checksum", help="Checksum generation commands")
    checksum_sub = checksum_parser.add_subparsers(dest="checksum_kind", required=True)
    checksum_sub.add_parser("drive", help="Generate drive checksum manifest")
    return parser


def main(argv=None):
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    allowed_extras = {"render", "compare", "verify"}
    if extras and args.group not in allowed_extras:
        parser.error("Unrecognized arguments: " + " ".join(extras))

    if args.group == "convert":
        from vhs_pipeline import commands

        if args.convert_kind == "avi":
            return commands.run_convert_avi(args.files)
        if args.convert_kind == "umatic":
            return commands.run_convert_umatic(args.files)

    if args.group == "metadata":
        from vhs_pipeline import commands

        if args.metadata_kind == "build":
            return commands.run_generate_archive_metadata()
        if args.metadata_kind == "embed":
            return commands.run_embed_metadata(args.files)

    if args.group == "proxy":
        from vhs_pipeline import commands

        return commands.run_make_proxies()

    if args.group == "render":
        from vhs_pipeline import commands

        return commands.run_make_videos(extras)

    if args.group == "compare":
        from vhs_pipeline import commands

        return commands.run_make_comparisons(extras)

    if args.group == "verify":
        from vhs_pipeline import commands

        if args.verify_kind == "archive":
            return commands.run_verify_archive(extras)
        if args.verify_kind == "drive":
            return commands.run_verify_drive(extras)

    if args.group == "checksum":
        from vhs_pipeline import commands

        if args.checksum_kind == "drive":
            return commands.run_generate_drive_checksum()

    parser.error("Unknown command.")
    return 2
