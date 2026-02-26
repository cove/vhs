from __future__ import annotations

import argparse


def _append_flag(argv, enabled, *flags):
    if enabled and flags:
        argv.append(flags[0])


def _append_option(argv, value, *flags):
    if value is None or not flags:
        return
    argv.extend([flags[0], str(value)])


def _append_repeat(argv, values, *flags):
    if not flags:
        return
    for value in list(values or []):
        argv.extend([flags[0], str(value)])


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vhs.py",
        description="Unified command surface for VHS archive workflows.",
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
    tuner_parser = subparsers.add_parser("tuner", help="Launch the plain HTML VHS tuner web UI")
    tuner_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    tuner_parser.add_argument("--port", type=int, default=8092, help="Bind port (default: 8092)")
    render_parser = subparsers.add_parser(
        "render",
        help="Run delivery render pipeline",
    )
    render_parser.add_argument(
        "render_args",
        nargs=argparse.REMAINDER,
        help="Optional args forwarded to the render pipeline.",
    )
    compare_parser = subparsers.add_parser(
        "compare",
        help="Create side-by-side original vs processed chapter comparisons",
    )
    compare_parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Only process archive names containing this substring (case-insensitive). Repeatable.",
    )
    compare_parser.add_argument(
        "--title",
        action="append",
        default=[],
        help="Only process chapter titles containing this substring (case-insensitive). Repeatable.",
    )
    compare_parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Target height for each side before stacking (default: 480).",
    )
    compare_parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max number of comparisons to create (0 = no limit).",
    )
    compare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even if they already exist.",
    )
    compare_parser.add_argument(
        "--output-root",
        default=None,
        help="Output root directory for comparison videos.",
    )

    verify_parser = subparsers.add_parser("verify", help="Checksum verification commands")
    verify_sub = verify_parser.add_subparsers(dest="verify_kind", required=True)
    verify_archive = verify_sub.add_parser("archive", help="Verify archive checksum manifest")
    verify_drive = verify_sub.add_parser("drive", help="Verify drive checksum manifest")
    for verify_cmd in (verify_archive, verify_drive):
        verify_cmd.add_argument("manifest", nargs="?", default=None, help="Path to manifest file.")
        verify_cmd.add_argument("--blake3", "--b3", action="store_true", help="Force BLAKE3 verify mode.")
        verify_cmd.add_argument("--sha3", "--sha3-256", action="store_true", help="Force SHA3 verify mode.")

    checksum_parser = subparsers.add_parser("checksum", help="Checksum generation commands")
    checksum_sub = checksum_parser.add_subparsers(dest="checksum_kind", required=True)
    checksum_sub.add_parser("drive", help="Generate drive checksum manifest")
    return parser


def main(argv=None):
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    allowed_extras = {"render"}
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

    if args.group == "tuner":
        from vhs_pipeline import commands

        return commands.run_tuner(host=args.host, port=args.port)

    if args.group == "render":
        from vhs_pipeline import commands

        render_argv = list(extras or []) + list(args.render_args or [])
        if render_argv and render_argv[0] == "--":
            render_argv = render_argv[1:]
        return commands.run_make_videos(render_argv)

    if args.group == "compare":
        from vhs_pipeline import commands

        compare_argv = []
        _append_repeat(compare_argv, args.archive, "--archive")
        _append_repeat(compare_argv, args.title, "--title")
        _append_option(compare_argv, args.height, "--height")
        _append_option(compare_argv, args.max, "--max")
        _append_flag(compare_argv, args.overwrite, "--overwrite")
        _append_option(compare_argv, args.output_root, "--output-root")
        return commands.run_make_comparisons(compare_argv)

    if args.group == "verify":
        from vhs_pipeline import commands

        verify_argv = []
        if args.manifest:
            verify_argv.append(str(args.manifest))
        if args.blake3:
            verify_argv.append("--blake3")
        elif args.sha3:
            verify_argv.append("--sha3")
        if args.verify_kind == "archive":
            return commands.run_verify_archive(verify_argv)
        if args.verify_kind == "drive":
            return commands.run_verify_drive(verify_argv)

    if args.group == "checksum":
        from vhs_pipeline import commands

        if args.checksum_kind == "drive":
            return commands.run_generate_drive_checksum()

    parser.error("Unknown command.")
    return 2
