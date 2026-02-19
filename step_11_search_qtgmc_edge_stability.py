#!/usr/bin/env python3.11
#
# Searches QTGMC combinations for stable frame edges using cropdetect jitter.
# For each test case:
# - picks a chapter by title match
# - samples a 30-frame midpoint clip
# - sweeps discrete QTGMC combos
# - runs binary interval search over Sharpness
# - ranks candidates by edge stability score (lower is better)
# - writes top-4 preview clips + full-size 2x2 quadrant
#
import json
import re
import statistics
from pathlib import Path

from common import *

SAMPLE_FRAMES = 30
FPS_NUM = 30000
FPS_DEN = 1001

SEARCH_ITERS = 5
SHARPNESS_MIN = 0.20
SHARPNESS_MAX = 1.60

CROP_LIMIT = 24
CROP_ROUND = 2
DETAIL_LOSS_TOLERANCE = 0.01

# Priority: stability > detail > smoothness
WEIGHT_STABILITY = 1.0
WEIGHT_DETAIL = 0.6
WEIGHT_SMOOTHNESS = 0.2

DETAIL_METRIC_VF = "edgedetect=mode=colormix,signalstats,metadata=mode=print"
TEMPORAL_METRIC_VF = "tblend=all_mode=difference,signalstats,metadata=mode=print"
SIGNALSTATS_YAVG_RE = re.compile(r"lavfi\.signalstats\.YAVG=([0-9]+(?:\.[0-9]+)?)")

TEST_CASES = [
    {
        "name": "wedding_vows_mid",
        "archive": "callahan_01_archive",
        "title_contains": "Wedding Vows",
        "offset_sec": 150,
    },
    {
        "name": "ultrasound_mid",
        "archive": "callahan_04_archive",
        "title_contains": "Ultrasound - 01",
    },
]

QTGMC_MACROS = [
    {
        "name": "balanced_ref",
        "sharpness_range": (0.30, 1.45),
        "opts": {
            "Preset": "Very Slow",
            "SourceMatch": 3,
            "Lossless": 2,
            "TR2": 3,
            "EZKeepGrain": 1.0,
            "MatchEdi": "NNEDI3",
            "MatchEdi2": "NNEDI3",
            "NoisePreset": "Fast",
        },
    },
    {
        "name": "balanced_yadif_match",
        "sharpness_range": (0.25, 1.25),
        "opts": {
            "Preset": "Very Slow",
            "SourceMatch": 3,
            "Lossless": 2,
            "TR2": 3,
            "EZKeepGrain": 1.0,
            "MatchEdi": "Yadif",
            "MatchEdi2": "Yadif",
            "NoisePreset": "Fast",
        },
    },
    {
        "name": "sm1_fastmatch_ll2",
        "sharpness_range": (0.20, 0.95),
        "opts": {
            "Preset": "Slow",
            "SourceMatch": 1,
            "Lossless": 2,
            "TR2": 2,
            "MatchPreset": "Very Fast",
            "MatchEdi": "NNEDI3",
            "MatchEdi2": "Bob",
            "EZKeepGrain": 0.5,
            "NoisePreset": "Fast",
        },
    },
    {
        "name": "sm3_matchenhance",
        "sharpness_range": (0.22, 1.10),
        "opts": {
            "Preset": "Slow",
            "SourceMatch": 3,
            "Lossless": 2,
            "TR2": 2,
            "MatchPreset": "Faster",
            "MatchPreset2": "Ultra Fast",
            "MatchEnhance": 0.75,
            "NoiseProcess": 1,
            "Sigma": 1.5,
            "NoiseRestore": 0.7,
            "MatchEdi": "NNEDI3",
            "MatchEdi2": "NNEDI3",
        },
    },
    {
        "name": "motion_stable",
        "sharpness_range": (0.20, 1.05),
        "opts": {
            "Preset": "Placebo",
            "SourceMatch": 3,
            "Lossless": 2,
            "TR0": 2,
            "TR1": 2,
            "TR2": 3,
            "Rep0": 4,
            "Rep2": 4,
            "Search": 5,
            "SearchParam": 2,
            "SubPel": 2,
            "NoiseProcess": 1,
            "Denoiser": "fft3df",
            "NoisePreset": "Slow",
            "Sigma": 1.6,
            "GrainRestore": 0.18,
            "NoiseRestore": 0.06,
            "MatchEdi": "Yadif",
            "MatchEdi2": "Yadif",
        },
    },
    {
        "name": "detail_focus",
        "sharpness_range": (0.70, 1.80),
        "opts": {
            "Preset": "Very Slow",
            "SourceMatch": 3,
            "Lossless": 1,
            "TR0": 2,
            "TR1": 1,
            "TR2": 1,
            "MatchEnhance": 0.72,
            "SMode": 2,
            "SLMode": 0,
            "NoiseProcess": 1,
            "Denoiser": "fft3df",
            "NoisePreset": "Fast",
            "Sigma": 0.9,
            "GrainRestore": 0.08,
            "NoiseRestore": 0.02,
            "MatchEdi": "NNEDI3",
            "MatchEdi2": "NNEDI3",
        },
    },
    {
        "name": "grain_keep",
        "sharpness_range": (0.25, 1.20),
        "opts": {
            "Preset": "Slow",
            "SourceMatch": 2,
            "Lossless": 1,
            "TR0": 2,
            "TR1": 1,
            "TR2": 2,
            "MatchEnhance": 0.45,
            "NoiseProcess": 2,
            "Denoiser": "fft3df",
            "NoisePreset": "Medium",
            "Sigma": 2.0,
            "GrainRestore": 0.42,
            "NoiseRestore": 0.18,
            "StabilizeNoise": True,
            "MatchEdi": "Yadif",
            "MatchEdi2": "NNEDI3",
        },
    },
    {
        "name": "clean_restore",
        "sharpness_range": (0.35, 1.15),
        "opts": {
            "Preset": "Slow",
            "SourceMatch": 2,
            "Lossless": 0,
            "TR0": 1,
            "TR1": 1,
            "TR2": 1,
            "MatchEnhance": 0.55,
            "NoiseProcess": 1,
            "Denoiser": "fft3df",
            "NoisePreset": "Fast",
            "Sigma": 1.35,
            "GrainRestore": 0.12,
            "NoiseRestore": 0.03,
            "MatchEdi": "NNEDI3",
            "MatchEdi2": "Yadif",
        },
    },
]


def fmt_avs_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return f"\"{v}\""


def qtgmc_call(opts):
    parts = []
    for k, v in opts.items():
        if str(k).startswith("_"):
            continue
        parts.append(f"{k}={fmt_avs_value(v)}")
    return "QTGMC(" + ",".join(parts) + ")"


def sec_to_frame(seconds):
    return int(round(float(seconds) * FPS_NUM / FPS_DEN))


def chapter_mid_window(ch, sample_frames):
    start_frame = sec_to_frame(ch["start"])
    end_frame = sec_to_frame(ch["end"])
    if end_frame < start_frame:
        return start_frame, start_frame

    chapter_len = end_frame - start_frame + 1
    n = min(sample_frames, chapter_len)
    mid = (start_frame + end_frame) // 2
    clip_start = mid - (n // 2)
    clip_start = max(start_frame, clip_start)
    clip_end = clip_start + n - 1
    if clip_end > end_frame:
        clip_end = end_frame
        clip_start = max(start_frame, clip_end - n + 1)
    return clip_start, clip_end


def chapter_offset_window(ch, offset_sec, sample_frames):
    ch_start = sec_to_frame(ch["start"])
    ch_end = sec_to_frame(ch["end"])
    start_frame = sec_to_frame(ch["start"] + float(offset_sec))
    end_frame = start_frame + sample_frames - 1
    if end_frame > ch_end:
        end_frame = ch_end
        start_frame = max(ch_start, end_frame - sample_frames + 1)
    if start_frame < ch_start:
        start_frame = ch_start
        end_frame = min(ch_end, start_frame + sample_frames - 1)
    return start_frame, end_frame


def pick_chapter(chapters, title_contains):
    needle = (title_contains or "").strip().lower()
    if not needle:
        return chapters[0] if chapters else None
    for ch in chapters:
        title = str(ch.get("title", "")).lower()
        if needle in title:
            return ch
    return None


def rewrite_filter_qtgmc(filter_text, qtgmc_opts):
    out = []
    replaced = False
    pattern = re.compile(r"^\s*QTGMC\s*\(")
    for line in filter_text.splitlines():
        if (not replaced) and pattern.search(line):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + qtgmc_call(qtgmc_opts))
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise RuntimeError("Could not find QTGMC(...) in filter script.")
    return "\n".join(out) + "\n"


def build_avs(src_path, start_frame, end_frame, filter_text):
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
        f'FFmpegSource2("{src}", atrack=-1)',
        f"Trim({start_frame},{end_frame})",
    ]
    return "\n".join(lines) + "\n" + filter_text


def build_raw_avs(src_path, start_frame, end_frame):
    src = str(src_path).replace("\\", "/")
    lines = [
        f'LoadPlugin("{QTGMC_DIR}/ffms2.dll")',
        f'FFmpegSource2("{src}", atrack=-1)',
        f"Trim({start_frame},{end_frame})",
    ]
    return "\n".join(lines) + "\n"


def parse_crop_text(text):
    pat = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
    rows = []
    for line in str(text).splitlines():
        m = pat.search(line)
        if m:
            rows.append(tuple(int(x) for x in m.groups()))

    if len(rows) < 5:
        return {
            "score": float("inf"),
            "samples": len(rows),
            "x_std": float("inf"),
            "y_std": float("inf"),
            "w_std": float("inf"),
            "h_std": float("inf"),
            "x_span": float("inf"),
            "y_span": float("inf"),
            "w_span": float("inf"),
            "h_span": float("inf"),
        }

    w = [r[0] for r in rows]
    h = [r[1] for r in rows]
    x = [r[2] for r in rows]
    y = [r[3] for r in rows]

    x_std = statistics.pstdev(x)
    y_std = statistics.pstdev(y)
    w_std = statistics.pstdev(w)
    h_std = statistics.pstdev(h)

    score = x_std + y_std + 0.5 * (w_std + h_std)

    return {
        "score": score,
        "samples": len(rows),
        "x_std": x_std,
        "y_std": y_std,
        "w_std": w_std,
        "h_std": h_std,
        "x_span": max(x) - min(x),
        "y_span": max(y) - min(y),
        "w_span": max(w) - min(w),
        "h_span": max(h) - min(h),
    }


def run_signalstats_metric(avs_path, vf, log_path):
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "info",
        "-i", str(avs_path),
        "-vf", vf,
        "-an",
        "-f", "null",
        "-",
    ]
    print("Command: " + " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(c) for c in cmd],
        check=True,
        text=True,
        capture_output=True,
    )
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    log_path.write_text(text, encoding="utf-8")

    vals = [float(m.group(1)) for m in SIGNALSTATS_YAVG_RE.finditer(text)]
    if not vals:
        raise RuntimeError(f"No signalstats YAVG values parsed from {avs_path}")
    return statistics.fmean(vals), len(vals)


def compute_quality_metrics(avs_path, log_dir, tag):
    detail_log = log_dir / f"{safe(tag)}.detail.log"
    temporal_log = log_dir / f"{safe(tag)}.temporal.log"
    detail_mean, detail_samples = run_signalstats_metric(avs_path, DETAIL_METRIC_VF, detail_log)
    temporal_mean, temporal_samples = run_signalstats_metric(avs_path, TEMPORAL_METRIC_VF, temporal_log)
    return {
        "detail_mean": detail_mean,
        "detail_samples": detail_samples,
        "detail_log_path": str(detail_log),
        "temporal_mean": temporal_mean,
        "temporal_samples": temporal_samples,
        "temporal_log_path": str(temporal_log),
    }


def compute_composite_score(stability_score, detail_ratio, temporal_ratio):
    detail_floor = 1.0 - DETAIL_LOSS_TOLERANCE
    detail_penalty = max(0.0, (detail_floor - detail_ratio) / DETAIL_LOSS_TOLERANCE)
    smoothness_penalty = temporal_ratio
    score = (
        WEIGHT_STABILITY * stability_score
        + WEIGHT_DETAIL * detail_penalty
        + WEIGHT_SMOOTHNESS * smoothness_penalty
    )
    return score, detail_penalty, smoothness_penalty


def evaluate_stability(case_ctx, run_ctx, qtgmc_opts, sharpness, eval_index):
    opts = dict(qtgmc_opts)
    opts["Sharpness"] = round(float(sharpness), 4)

    patched_filter = rewrite_filter_qtgmc(case_ctx["filter_text"], opts)
    tag = f"{case_ctx['name']}_{run_ctx['profile_name']}_{eval_index:03d}_s{opts['Sharpness']:.4f}"
    avs_path = run_ctx["avs_dir"] / f"{safe(tag)}.avs"
    log_path = run_ctx["log_dir"] / f"{safe(tag)}.crop.log"
    avs_text = build_avs(
        case_ctx["archive_path"],
        case_ctx["start_frame"],
        case_ctx["end_frame"],
        patched_filter,
    )
    avs_path.write_text(avs_text, encoding="utf-8")

    vf = f"cropdetect=limit={CROP_LIMIT}:round={CROP_ROUND}:reset=1"
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "info",
        "-i", str(avs_path),
        "-vf", vf,
        "-an",
        "-f", "null",
        "-",
    ]
    print("Command: " + " ".join(map(str, cmd)))
    proc = subprocess.run(
        [str(c) for c in cmd],
        check=True,
        text=True,
        capture_output=True,
    )
    crop_text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    log_path.write_text(crop_text, encoding="utf-8")
    crop_metrics = parse_crop_text(crop_text)
    quality = compute_quality_metrics(avs_path, run_ctx["log_dir"], tag)

    raw_detail = float(case_ctx["raw_detail_mean"])
    raw_temporal = float(case_ctx["raw_temporal_mean"])
    detail_ratio = (quality["detail_mean"] / raw_detail) if raw_detail > 0 else 1.0
    temporal_ratio = (quality["temporal_mean"] / raw_temporal) if raw_temporal > 0 else 1.0

    stability_score = float(crop_metrics["score"])
    score, detail_penalty, smoothness_penalty = compute_composite_score(
        stability_score, detail_ratio, temporal_ratio
    )

    metrics = dict(crop_metrics)
    metrics.update(quality)
    metrics.update(
        {
            "score": score,
            "stability_score": stability_score,
            "detail_ratio": detail_ratio,
            "temporal_ratio": temporal_ratio,
            "detail_penalty": detail_penalty,
            "smoothness_penalty": smoothness_penalty,
            "sharpness": opts["Sharpness"],
            "avs_path": str(avs_path),
            "log_path": str(log_path),  # cropdetect log path
        }
    )
    return metrics


def search_sharpness(case_ctx, run_ctx, qtgmc_opts, low, high, iterations):
    cache = {}
    eval_counter = 0

    def eval_at(value):
        nonlocal eval_counter
        key = round(float(value), 4)
        if key not in cache:
            eval_counter += 1
            cache[key] = evaluate_stability(case_ctx, run_ctx, qtgmc_opts, key, eval_counter)
        return cache[key]

    lo = float(low)
    hi = float(high)
    best = None

    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        left = (lo + mid) / 2.0
        right = (mid + hi) / 2.0
        left_res = eval_at(left)
        right_res = eval_at(right)

        if best is None or left_res["score"] < best["score"]:
            best = left_res
        if right_res["score"] < best["score"]:
            best = right_res

        if left_res["score"] <= right_res["score"]:
            hi = mid
        else:
            lo = mid

    center_res = eval_at((lo + hi) / 2.0)
    if best is None or center_res["score"] < best["score"]:
        best = center_res

    history = sorted(cache.values(), key=lambda r: r["sharpness"])
    return best, history


def encode_preview(case_ctx, qtgmc_opts, sharpness, out_path):
    opts = dict(qtgmc_opts)
    opts["Sharpness"] = round(float(sharpness), 4)
    patched_filter = rewrite_filter_qtgmc(case_ctx["filter_text"], opts)
    avs_text = build_avs(
        case_ctx["archive_path"],
        case_ctx["start_frame"],
        case_ctx["end_frame"],
        patched_filter,
    )
    avs_path = out_path.with_suffix(".avs")
    avs_path.write_text(avs_text, encoding="utf-8")

    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(avs_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        "-y", str(out_path),
    ]
    run(cmd)
    return avs_path


def encode_control_preview(case_ctx, out_path):
    avs_text = build_raw_avs(
        case_ctx["archive_path"],
        case_ctx["start_frame"],
        case_ctx["end_frame"],
    )
    avs_path = out_path.with_suffix(".avs")
    avs_path.write_text(avs_text, encoding="utf-8")

    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-v", "error",
        "-i", str(avs_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        "-y", str(out_path),
    ]
    run(cmd)
    return avs_path


def make_quadrant(inputs, out_path):
    if len(inputs) != 4:
        return False
    cmd = [FFMPEG_BIN, "-nostdin", "-v", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    cmd += [
        "-filter_complex",
        "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[v]",
        "-map", "[v]",
        "-an",
        "-shortest",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-y", str(out_path),
    ]
    run(cmd)
    return True


def qtgmc_profile_grid():
    profiles = []
    for macro in QTGMC_MACROS:
        profile = dict(macro["opts"])
        profile["_macro"] = macro["name"]
        low, high = macro["sharpness_range"]
        profile["_sharpness_min"] = float(low)
        profile["_sharpness_max"] = float(high)
        profiles.append(profile)
    return profiles


def profile_sharpness_range(profile):
    low = float(profile.get("_sharpness_min", SHARPNESS_MIN))
    high = float(profile.get("_sharpness_max", SHARPNESS_MAX))
    if high < low:
        low, high = high, low
    return low, high


def profile_name(profile):
    macro = str(profile.get("_macro", "macro")).lower().replace(" ", "_")
    preset = str(profile.get("Preset", "na")).lower().replace(" ", "")
    return (
        f"{macro}_{preset}_sm{profile.get('SourceMatch', 'na')}"
        f"_tr2{profile.get('TR2', 'na')}_ll{profile.get('Lossless', 'na')}"
    )


def write_case_results(case_dir, case_results):
    rows = [
        "rank\tprofile\tmacro\tpreset\tsource_match\ttr2\tlossless\tsharpness_min\tsharpness_max\tbest_sharpness\t"
        "score\tstability_score\tdetail_ratio\tdetail_penalty\ttemporal_ratio\tsmoothness_penalty\t"
        "x_std\ty_std\tw_std\th_std\tsamples\tevals"
    ]
    for i, r in enumerate(case_results, start=1):
        s_low, s_high = profile_sharpness_range(r["profile"])
        rows.append(
            "\t".join(
                [
                    str(i),
                    r["profile_name"],
                    str(r["profile"].get("_macro", "")),
                    str(r["profile"].get("Preset", "")),
                    str(r["profile"].get("SourceMatch", "")),
                    str(r["profile"].get("TR2", "")),
                    str(r["profile"].get("Lossless", "")),
                    f"{s_low:.4f}",
                    f"{s_high:.4f}",
                    f"{r['best']['sharpness']:.4f}",
                    f"{r['best']['score']:.6f}",
                    f"{r['best']['stability_score']:.6f}",
                    f"{r['best']['detail_ratio']:.6f}",
                    f"{r['best']['detail_penalty']:.6f}",
                    f"{r['best']['temporal_ratio']:.6f}",
                    f"{r['best']['smoothness_penalty']:.6f}",
                    f"{r['best']['x_std']:.6f}",
                    f"{r['best']['y_std']:.6f}",
                    f"{r['best']['w_std']:.6f}",
                    f"{r['best']['h_std']:.6f}",
                    str(r["best"]["samples"]),
                    str(len(r["history"])),
                ]
            )
        )
    (case_dir / "edge_stability_rankings.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    history_dump = []
    for r in case_results:
        history_dump.append(
            {
                "profile_name": r["profile_name"],
                "profile": r["profile"],
                "best": r["best"],
                "history": r["history"],
            }
        )
    (case_dir / "edge_stability_history.json").write_text(
        json.dumps(history_dump, indent=2),
        encoding="utf-8",
    )


def main():
    ensure_ffmpeg_exists()

    base_dir = CLIPS_DIR / "filter_tests" / "qtgmc_edge_search"
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
    print(f"Output: {out_dir}")

    profiles = qtgmc_profile_grid()
    print(f"Testing {len(profiles)} QTGMC macro profiles; binary search iters={SEARCH_ITERS}")

    overall_rows = [
        "case\tarchive\tchapter\tbest_profile\tmacro\tbest_sharpness\tscore\tstability_score\t"
        "detail_ratio\ttemporal_ratio\tx_std\ty_std\tsamples"
    ]

    for case in TEST_CASES:
        archive_stem = case["archive"]
        archive_path = ARCHIVE_DIR / f"{archive_stem}.mkv"
        chapters_file = METADATA_DIR / archive_stem / "chapters.ffmetadata"
        filter_path = METADATA_DIR / archive_stem / "filter.avs"
        case_name = case["name"]

        if not archive_path.exists():
            print(f"Skipping {case_name}: archive not found {archive_path}")
            continue
        if not chapters_file.exists():
            print(f"Skipping {case_name}: chapters not found {chapters_file}")
            continue
        if not filter_path.exists():
            print(f"Skipping {case_name}: filter not found {filter_path}")
            continue

        _, chapters = parse_chapters(chapters_file)
        ch = pick_chapter(chapters, case.get("title_contains", ""))
        if not ch:
            print(f"Skipping {case_name}: no chapter match for '{case.get('title_contains', '')}'")
            continue

        offset_sec = case.get("offset_sec")
        if offset_sec is not None:
            start_frame, end_frame = chapter_offset_window(ch, offset_sec, SAMPLE_FRAMES)
        else:
            start_frame, end_frame = chapter_mid_window(ch, SAMPLE_FRAMES)
        case_dir = out_dir / safe(case_name)
        avs_dir = case_dir / "eval_avs"
        log_dir = case_dir / "eval_logs"
        preview_dir = case_dir / "previews"
        avs_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

        case_ctx = {
            "name": case_name,
            "archive_stem": archive_stem,
            "archive_path": archive_path,
            "chapter_title": ch.get("title", ""),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "filter_text": filter_path.read_text(encoding="utf-8"),
        }
        raw_avs = avs_dir / f"{safe(case_name)}_raw_baseline.avs"
        raw_avs.write_text(
            build_raw_avs(case_ctx["archive_path"], case_ctx["start_frame"], case_ctx["end_frame"]),
            encoding="utf-8",
        )
        raw_metrics = compute_quality_metrics(raw_avs, log_dir, f"{case_name}_raw_baseline")
        case_ctx["raw_detail_mean"] = raw_metrics["detail_mean"]
        case_ctx["raw_temporal_mean"] = raw_metrics["temporal_mean"]
        print(
            f"\nCase {case_name}: {archive_stem} | {case_ctx['chapter_title']} | "
            f"frames {start_frame}-{end_frame}"
        )
        print(
            f"  Raw baselines: detail_mean={case_ctx['raw_detail_mean']:.6f} "
            f"temporal_mean={case_ctx['raw_temporal_mean']:.6f}"
        )

        case_results = []
        for idx, profile in enumerate(profiles, start=1):
            pname = profile_name(profile)
            s_low, s_high = profile_sharpness_range(profile)
            print(
                f"  [{idx:02d}/{len(profiles)}] searching {pname} "
                f"(sharpness {s_low:.2f}-{s_high:.2f})"
            )
            run_ctx = {"profile_name": pname, "avs_dir": avs_dir, "log_dir": log_dir}
            best, history = search_sharpness(
                case_ctx,
                run_ctx,
                profile,
                s_low,
                s_high,
                SEARCH_ITERS,
            )
            case_results.append(
                {
                    "profile_name": pname,
                    "profile": profile,
                    "best": best,
                    "history": history,
                }
            )

        case_results.sort(key=lambda r: r["best"]["score"])
        write_case_results(case_dir, case_results)

        control_mp4 = preview_dir / "00_control_raw.mp4"
        encode_control_preview(case_ctx, control_mp4)

        top = case_results[:3]
        preview_inputs = [control_mp4]
        for rank, r in enumerate(top, start=1):
            profile = r["profile"]
            s = r["best"]["sharpness"]
            out_name = safe(
                f"{rank:02d}_{r['profile_name']}_s{s:.4f}_score{r['best']['score']:.4f}"
            )
            preview_mp4 = preview_dir / f"{out_name}.mp4"
            encode_preview(case_ctx, profile, s, preview_mp4)
            preview_inputs.append(preview_mp4)

        if len(preview_inputs) == 4:
            quadrant = case_dir / "quadrant_top4.mp4"
            make_quadrant(preview_inputs, quadrant)
            print(f"  Wrote quadrant: {quadrant}")

        best = case_results[0]
        overall_rows.append(
            "\t".join(
                [
                    case_name,
                    archive_stem,
                    case_ctx["chapter_title"],
                    best["profile_name"],
                    str(best["profile"].get("_macro", "")),
                    f"{best['best']['sharpness']:.4f}",
                    f"{best['best']['score']:.6f}",
                    f"{best['best']['stability_score']:.6f}",
                    f"{best['best']['detail_ratio']:.6f}",
                    f"{best['best']['temporal_ratio']:.6f}",
                    f"{best['best']['x_std']:.6f}",
                    f"{best['best']['y_std']:.6f}",
                    str(best["best"]["samples"]),
                ]
            )
        )
        print(
            f"  Best: {best['profile_name']} sharpness={best['best']['sharpness']:.4f} "
            f"score={best['best']['score']:.6f} "
            f"stability={best['best']['stability_score']:.6f} "
            f"detail_ratio={best['best']['detail_ratio']:.4f} "
            f"temporal_ratio={best['best']['temporal_ratio']:.4f}"
        )

    (out_dir / "overall_best.tsv").write_text("\n".join(overall_rows) + "\n", encoding="utf-8")
    print("\nDone.")


if __name__ == "__main__":
    main()
