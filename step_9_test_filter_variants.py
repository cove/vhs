#
# Generates short test encodes with different QTGMC/brightness/wobble settings
# using the shortest chapter from an archive. Outputs MP4s for quick A/B review.
#
import itertools
from pathlib import Path

from common import *

# Archive to test
TARGET_ARCHIVE = "callahan_01_archive"
FIELD_ORDER = "BFF"  # "BFF" or "TFF"


def fmt_avs_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return f"\"{v}\""


def qtgmc_call(opts):
    parts = []
    for k, v in opts.items():
        parts.append(f"{k}={fmt_avs_value(v)}")
    return "QTGMC(" + ",".join(parts) + ")"


TEST_SEGMENTS = [
    {
        "name": "darkness",
        "start_frame": 47125,
        "end_frame": 47225,
    },
    {
        "name": "wobble",
        "start_frame": 7683,
        "end_frame": 7783,
    },
]


def build_avs(
    src_path,
    start_frame,
    end_frame,
    field_order,
    qtgmc_opts,
    gamma,
    wobble_fix,
    wobble_strength=0.3,
):
    src = str(src_path).replace("\\", "/")

    lines = [
        f'LoadPlugin("{QTGMC_DIR}/ffms2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/masktools2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/Rgtools.dll")',
        f'LoadPlugin("{QTGMC_DIR}/mvtools2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/DePanEstimate.dll")',
        f'LoadPlugin("{QTGMC_DIR}/DePan.dll")',
        f'LoadPlugin("{QTGMC_DIR}/nnedi3.dll")',
        f'LoadPlugin("{QTGMC_DIR}/yadifmod2.dll")',
        f'LoadPlugin("{QTGMC_DIR}/fft3dfilter.dll")',
        f'LoadPlugin("{QTGMC_DIR}/LoadDLL64.dll")',
        f'LoadPlugin("{QTGMC_DIR}/SmoothAdjust.dll")',
        f'LoadDLL("{QTGMC_DIR}/libfftw3f-3.dll")',
        f'Import("{QTGMC_DIR}/Zs_RF_Shared.avsi")',
        f'Import("{QTGMC_DIR}/QTGMC.avsi")',
        f'src = FFmpegSource2("{src}", atrack=-1)',
        f'src = src.Trim({start_frame},{end_frame})',
        'src = src.AssumeFPS(30000,1001)',
    ]

    if field_order == "BFF":
        lines.append('src = src.AssumeBFF()')
    elif field_order == "TFF":
        lines.append('src = src.AssumeTFF()')

    lines += [
        'src = src.ConvertToYV12(matrix="Rec601")',
        f'src = src.{qtgmc_call(qtgmc_opts)}',
        '# stabilization disabled',
        'src = src.Crop(4,2,-8,-10)',
        'src = src.LanczosResize(640,480)',
        'src = src.ConvertToYV12(interlaced=false)',
        'src = src.SmoothLevels(16, 1.0, 255, 16, 235, limiter=1, tvrange=true, dither=0)',
    ]

    if gamma and abs(gamma - 1.0) > 1e-6:
        lines.append(f'src = src.Levels(0, {gamma}, 255, 0, 255)')

    if wobble_fix:
        lines.append(f'src = src.TurnLeft().Blur({wobble_strength}).TurnRight()')


    lines.append('return src')

    return "\n".join(lines) + "\n"


def main():
    ensure_ffmpeg_exists()

    archive_path = None
    archive_stem = None

    if TARGET_ARCHIVE:
        archive_stem = TARGET_ARCHIVE
        archive_path = ARCHIVE_DIR / f"{archive_stem}.mkv"
    else:
        candidates = sorted(
            ARCHIVE_DIR.glob("*.mkv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in candidates:
            if (METADATA_DIR / p.stem / "chapters.ffmetadata").exists():
                archive_path = p
                archive_stem = p.stem
                break

    if not archive_path.exists():
        print(f"Archive MKV not found: {archive_path}")
        sys.exit(1)

    base_dir = CLIPS_DIR / "filter_tests" / archive_stem
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base_dir.glob("run_*"):
        if p.is_dir():
            try:
                existing.append(int(p.name.split("_", 1)[1]))
            except (ValueError, IndexError):
                continue
    next_run = (max(existing) + 1) if existing else 1
    out_dir = base_dir / f"run_{next_run:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    avs_dir = out_dir / "avs"
    avs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using archive: {archive_path}")
    for seg in TEST_SEGMENTS:
        print(f"Segment: {seg['name']} frames {seg['start_frame']} -> {seg['end_frame']}")
    print(f"Output: {out_dir}")

    qtgmc_variants = [
        (
            "qtgmc_base",
            {
                "Preset": "Very Slow",
                "EZKeepGrain": 1.0,
                "Sharpness": 1.2,
                "SourceMatch": 3,
                "Lossless": 2,
                "TR2": 3,
                "MatchEdi": "NNEDI3",
                "MatchEdi2": "NNEDI3",
            },
            0.0,
        ),
        (
            "qtgmc_soft",
            {
                "Preset": "Very Slow",
                "EZKeepGrain": 1.0,
                "Sharpness": 0.8,
                "SourceMatch": 2,
                "Lossless": 0,
                "TR2": 2,
                "MatchEdi": "NNEDI3",
                "MatchEdi2": "NNEDI3",
            },
            0.6,
        ),
        (
            "qtgmc_very_soft",
            {
                "Preset": "Slow",
                "EZKeepGrain": 1.0,
                "Sharpness": 0.6,
                "SourceMatch": 1,
                "Lossless": 0,
                "TR2": 1,
                "MatchEdi": "NNEDI3",
                "MatchEdi2": "NNEDI3",
            },
            1.0,
        ),
    ]

    brightness_variants = [
        ("bright_100", 1.00, 0.0),
        ("bright_110", 1.10, 0.10),
        ("bright_125", 1.25, 0.25),
        ("bright_140", 1.40, 0.40),
        ("bright_160", 1.60, 0.60),
        ("bright_180", 1.80, 0.80),
        ("bright_195", 1.95, 0.95),
    ]

    wobble_variants = [
        ("wobble_off", False, 0.0),
        ("wobble_mild", True, 0.3),
        ("wobble_strong", True, 0.6),
    ]

    field_order = FIELD_ORDER

    summary_lines = []
    for seg in TEST_SEGMENTS:
        idx = 1
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        if end_frame < start_frame:
            print(f"Invalid segment range: {seg}")
            sys.exit(1)

        if seg["name"] == "darkness":
            combos = []
            for q in qtgmc_variants:
                for b in brightness_variants:
                    combos.append((q, b, ("wobble_off", False, 0.0)))
            combos.sort(key=lambda c: (c[0][2] + c[1][2]))
        else:
            combos = []
            for q in qtgmc_variants:
                for w in wobble_variants:
                    combos.append((q, ("bright_100", 1.0, 0.0), w))
            combos.sort(key=lambda c: (c[0][2] + c[2][2]))

        for (qname, qopts, qstrength), (bname, gamma, bstrength), (wname, wobble, wstrength) in combos:
            variant_name = safe(f"{idx:03d}_{seg['name']}_{qname}_{bname}_{wname}")
            avs_path = avs_dir / f"{variant_name}.avs"
            out_path = out_dir / f"{variant_name}.mp4"

            avs_text = build_avs(
                archive_path,
                start_frame,
                end_frame,
                field_order,
                qopts,
                gamma,
                wobble,
                wstrength,
            )
            avs_path.write_text(avs_text, encoding="utf-8")

            cmd = [
                FFMPEG_BIN,
                "-nostdin",
                "-v", "error",
                "-i", str(avs_path),
            ]
            cmd += [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-an",
                "-y", str(out_path),
            ]
            run(cmd)

            summary_lines.append(
                f"{out_path.name}\t{seg['name']}\t{qname}\t{bname}\t{wname}\tframes={start_frame}-{end_frame}\tQTGMC={qopts}\tGamma={gamma}\tStrength={qstrength + bstrength + wstrength:.2f}"
            )
            idx += 1

    (out_dir / "variants.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("Generated tests:", len(summary_lines))


if __name__ == "__main__":
    main()
