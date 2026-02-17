#!/usr/bin/env python3.11
#
# Two-stage search pipeline for QTGMC settings.
# Stage 1:
# - evaluate all profiles with fast metrics over one window
# - search Sharpness by interval narrowing
# Stage 2:
# - rerank top-N profiles over multiple windows
# - optional IQA-PyTorch metric for final tie-breaking/penalty
#
import itertools
import json
import re
import statistics
import subprocess
import urllib.request
from pathlib import Path

from common import *

try:
    import cv2
    import numpy as np
except Exception as exc:
    raise RuntimeError(
        "OpenCV quality metrics require opencv-contrib-python and numpy."
    ) from exc

try:
    import torch
    import pyiqa
except Exception:
    torch = None
    pyiqa = None

SAMPLE_FRAMES = 30
FPS_NUM = 30000
FPS_DEN = 1001

SHARPNESS_MIN = 0.20
SHARPNESS_MAX = 1.60
STAGE1_SEARCH_ITERS = 5
STAGE2_SEARCH_ITERS = 6
STAGE2_TOP_N = 5
STAGE2_OFFSET_SPACING_SEC = 20.0

CROP_LIMIT = 24
CROP_ROUND = 2
DETAIL_LOSS_TOLERANCE = 0.10
EPS = 1e-6

# Priority requested by user: stability > detail > smoothness.
STAGE1_WEIGHTS = {
    "stability": 1.0,
    "detail": 0.8,
    "brisque": 0.6,
    "smoothness": 0.3,
    "iqa": 0.0,
}
STAGE2_WEIGHTS = {
    "stability": 1.0,
    "detail": 0.8,
    "brisque": 0.6,
    "smoothness": 0.3,
    "iqa": 0.4,
}

BRISQUE_FRAME_STRIDE_STAGE1 = 3
BRISQUE_FRAME_STRIDE_STAGE2 = 1

ENABLE_IQA_STAGE2 = True
IQA_METRIC_NAME = "topiq_nr"
IQA_FRAME_STRIDE_STAGE2 = 4

DETAIL_METRIC_VF = "edgedetect=mode=colormix,signalstats,metadata=mode=print"
TEMPORAL_METRIC_VF = "tblend=all_mode=difference,signalstats,metadata=mode=print"
SIGNALSTATS_YAVG_RE = re.compile(r"lavfi\.signalstats\.YAVG=([0-9]+(?:\.[0-9]+)?)")

BRISQUE_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/quality/samples/brisque_model_live.yml"
)
BRISQUE_RANGE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/quality/samples/brisque_range_live.yml"
)

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

PRESETS = ["Slow", "Very Slow", "Placebo"]
SOURCEMATCH_VALUES = [2, 3]
TR2_VALUES = [2, 3]


def safe_div(a, b, default=1.0):
    if abs(float(b)) < EPS:
        return float(default)
    return float(a) / float(b)


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


def sec_to_frame(seconds):
    return int(round(float(seconds) * FPS_NUM / FPS_DEN))


def window_from_center_seconds(ch, center_sec, sample_frames):
    chapter_start = sec_to_frame(ch["start"])
    chapter_end = sec_to_frame(ch["end"])
    if chapter_end < chapter_start:
        return chapter_start, chapter_start

    n = min(sample_frames, chapter_end - chapter_start + 1)
    center_frame = sec_to_frame(center_sec)
    start_frame = center_frame - (n // 2)
    start_frame = max(chapter_start, start_frame)
    end_frame = start_frame + n - 1
    if end_frame > chapter_end:
        end_frame = chapter_end
        start_frame = max(chapter_start, end_frame - n + 1)
    return start_frame, end_frame


def unique_windows(windows):
    seen = set()
    out = []
    for w in windows:
        key = (w["start_frame"], w["end_frame"])
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def build_stage_windows(ch, case):
    chapter_start = float(ch["start"])
    chapter_end = float(ch["end"])
    chapter_duration = max(0.0, chapter_end - chapter_start)

    offset_sec = case.get("offset_sec")
    if offset_sec is not None:
        center_sec = chapter_start + float(offset_sec)
    else:
        center_sec = chapter_start + (chapter_duration / 2.0)

    s1_start, s1_end = window_from_center_seconds(ch, center_sec, SAMPLE_FRAMES)
    stage1_windows = [
        {
            "name": "primary",
            "start_frame": s1_start,
            "end_frame": s1_end,
        }
    ]

    if offset_sec is not None:
        centers = [
            center_sec - STAGE2_OFFSET_SPACING_SEC,
            center_sec,
            center_sec + STAGE2_OFFSET_SPACING_SEC,
        ]
    else:
        centers = [
            chapter_start + chapter_duration * 0.25,
            chapter_start + chapter_duration * 0.50,
            chapter_start + chapter_duration * 0.75,
        ]

    stage2_windows = []
    for idx, csec in enumerate(centers, start=1):
        w_start, w_end = window_from_center_seconds(ch, csec, SAMPLE_FRAMES)
        stage2_windows.append(
            {
                "name": f"w{idx}",
                "start_frame": w_start,
                "end_frame": w_end,
            }
        )

    stage2_windows = unique_windows(stage2_windows)
    if not stage2_windows:
        stage2_windows = stage1_windows

    return stage1_windows, stage2_windows


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
            "stability_score": float("inf"),
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
        "stability_score": score,
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

def run_cropdetect_metrics(avs_path, log_path):
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
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")
    log_path.write_text(text, encoding="utf-8")
    return parse_crop_text(text)


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


def ensure_brisque_models():
    brisque_dir = BASE / "models" / "BRISQUE"
    brisque_dir.mkdir(parents=True, exist_ok=True)
    model_path = brisque_dir / "brisque_model_live.yml"
    range_path = brisque_dir / "brisque_range_live.yml"

    for url, dest in (
        (BRISQUE_MODEL_URL, model_path),
        (BRISQUE_RANGE_URL, range_path),
    ):
        if dest.exists():
            continue
        print(f"Downloading {dest.name}...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            dest.write_bytes(resp.read())
    return {"model_path": model_path, "range_path": range_path}


def init_brisque_context():
    if not hasattr(cv2, "quality") or not hasattr(cv2.quality, "QualityBRISQUE_compute"):
        raise RuntimeError("OpenCV quality module unavailable. Install opencv-contrib-python.")
    return ensure_brisque_models()


def init_iqa_metric():
    if not ENABLE_IQA_STAGE2:
        return None
    if pyiqa is None or torch is None:
        print("IQA rerank disabled: pyiqa/torch not available.")
        return None
    try:
        metric = pyiqa.create_metric(IQA_METRIC_NAME, device="cpu")
        print(f"IQA rerank enabled: {IQA_METRIC_NAME} (CPU)")
        return metric
    except Exception as exc:
        print(f"IQA rerank disabled: failed to init {IQA_METRIC_NAME}: {exc}")
        return None


def probe_dimensions(input_path):
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(input_path),
    ]
    proc = subprocess.run([str(c) for c in cmd], check=True, text=True, capture_output=True)
    line = (proc.stdout or "").strip()
    if not line or "x" not in line:
        raise RuntimeError(f"Could not parse dimensions for {input_path}")
    w_str, h_str = line.split("x", 1)
    return int(w_str), int(h_str)


def extract_bgr_frames(input_path, frame_stride=1):
    width, height = probe_dimensions(input_path)
    cmd = [FFMPEG_BIN, "-nostdin", "-v", "error", "-i", str(input_path)]
    if int(frame_stride) > 1:
        cmd += ["-vf", f"select=not(mod(n\\,{int(frame_stride)}))"]
    cmd += ["-an", "-sn", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    print("Command: " + " ".join(map(str, cmd)))

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_size = width * height * 3
    frames = []
    while True:
        chunk = proc.stdout.read(frame_size)
        if not chunk:
            break
        if len(chunk) < frame_size:
            break
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3)).copy()
        frames.append(frame)

    stderr = proc.stderr.read().decode("utf-8", errors="replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed ({rc}) for {input_path}\n{stderr}")
    return frames


def compute_brisque_mean(frames, brisque_ctx):
    vals = []
    model_path = str(brisque_ctx["model_path"])
    range_path = str(brisque_ctx["range_path"])
    for frame in frames:
        score = cv2.quality.QualityBRISQUE_compute(frame, model_path, range_path)
        vals.append(float(np.asarray(score).reshape(-1)[0]))
    if not vals:
        return float("inf"), 0
    return statistics.fmean(vals), len(vals)


def compute_iqa_mean(frames, iqa_metric, frame_stride):
    if iqa_metric is None:
        return None, 0
    vals = []
    with torch.no_grad():
        for idx in range(0, len(frames), max(1, int(frame_stride))):
            rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            out = iqa_metric(tensor)
            vals.append(float(out.item()))
    if not vals:
        return None, 0
    return statistics.fmean(vals), len(vals)


def compute_avs_metrics(avs_path, log_dir, tag, metric_ctx, stage_cfg, with_iqa=False):
    crop_log = log_dir / f"{safe(tag)}.crop.log"
    detail_log = log_dir / f"{safe(tag)}.detail.log"
    temporal_log = log_dir / f"{safe(tag)}.temporal.log"

    crop = run_cropdetect_metrics(avs_path, crop_log)
    detail_mean, detail_samples = run_signalstats_metric(avs_path, DETAIL_METRIC_VF, detail_log)
    temporal_mean, temporal_samples = run_signalstats_metric(avs_path, TEMPORAL_METRIC_VF, temporal_log)

    frames = extract_bgr_frames(avs_path, frame_stride=stage_cfg["brisque_frame_stride"])
    brisque_mean, brisque_samples = compute_brisque_mean(frames, metric_ctx["brisque"])

    iqa_mean = None
    iqa_samples = 0
    if with_iqa and metric_ctx.get("iqa") is not None:
        iqa_mean, iqa_samples = compute_iqa_mean(
            frames,
            metric_ctx["iqa"],
            stage_cfg["iqa_frame_stride"],
        )

    out = dict(crop)
    out.update(
        {
            "detail_mean": detail_mean,
            "detail_samples": detail_samples,
            "temporal_mean": temporal_mean,
            "temporal_samples": temporal_samples,
            "brisque_mean": brisque_mean,
            "brisque_samples": brisque_samples,
            "iqa_mean": iqa_mean,
            "iqa_samples": iqa_samples,
            "crop_log_path": str(crop_log),
            "detail_log_path": str(detail_log),
            "temporal_log_path": str(temporal_log),
        }
    )
    return out


def score_components(metrics, baseline, weights):
    stability_ratio = safe_div(metrics["stability_score"], baseline["stability_score"], default=float("inf"))
    detail_ratio = safe_div(metrics["detail_mean"], baseline["detail_mean"], default=1.0)
    temporal_ratio = safe_div(metrics["temporal_mean"], baseline["temporal_mean"], default=1.0)
    brisque_ratio = safe_div(metrics["brisque_mean"], baseline["brisque_mean"], default=1.0)

    detail_floor = 1.0 - DETAIL_LOSS_TOLERANCE
    detail_penalty = max(0.0, (detail_floor - detail_ratio) / max(DETAIL_LOSS_TOLERANCE, EPS))
    brisque_penalty = max(0.0, brisque_ratio - 1.0)
    smoothness_penalty = temporal_ratio

    iqa_ratio = None
    iqa_penalty = 0.0
    if (metrics.get("iqa_mean") is not None) and (baseline.get("iqa_mean") is not None):
        iqa_ratio = safe_div(metrics["iqa_mean"], baseline["iqa_mean"], default=1.0)
        iqa_penalty = max(0.0, 1.0 - iqa_ratio)

    score = (
        weights["stability"] * stability_ratio
        + weights["detail"] * detail_penalty
        + weights["brisque"] * brisque_penalty
        + weights["smoothness"] * smoothness_penalty
        + weights["iqa"] * iqa_penalty
    )
    return {
        "score": score,
        "stability_ratio": stability_ratio,
        "detail_ratio": detail_ratio,
        "temporal_ratio": temporal_ratio,
        "brisque_ratio": brisque_ratio,
        "iqa_ratio": iqa_ratio,
        "detail_penalty": detail_penalty,
        "smoothness_penalty": smoothness_penalty,
        "brisque_penalty": brisque_penalty,
        "iqa_penalty": iqa_penalty,
    }


def median_num(values, default=float("inf")):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return default
    return float(statistics.median(vals))


def aggregate_window_results(window_results):
    agg = {
        "score": median_num([w["score"] for w in window_results]),
        "stability_ratio": median_num([w["stability_ratio"] for w in window_results]),
        "detail_ratio": median_num([w["detail_ratio"] for w in window_results]),
        "temporal_ratio": median_num([w["temporal_ratio"] for w in window_results]),
        "brisque_ratio": median_num([w["brisque_ratio"] for w in window_results]),
        "iqa_ratio": median_num([w["iqa_ratio"] for w in window_results], default=None),
        "detail_penalty": median_num([w["detail_penalty"] for w in window_results]),
        "smoothness_penalty": median_num([w["smoothness_penalty"] for w in window_results]),
        "brisque_penalty": median_num([w["brisque_penalty"] for w in window_results]),
        "iqa_penalty": median_num([w["iqa_penalty"] for w in window_results], default=0.0),
        "stability_score": median_num([w["stability_score"] for w in window_results]),
        "x_std": median_num([w["x_std"] for w in window_results]),
        "y_std": median_num([w["y_std"] for w in window_results]),
        "w_std": median_num([w["w_std"] for w in window_results]),
        "h_std": median_num([w["h_std"] for w in window_results]),
        "samples": int(round(median_num([w["samples"] for w in window_results], default=0))),
    }
    agg["window_results"] = window_results
    return agg

def prepare_baselines(case_ctx, windows, run_ctx, stage_cfg, metric_ctx, with_iqa):
    baseline_map = {}
    for w in windows:
        tag = f"{case_ctx['name']}_{stage_cfg['name']}_baseline_{w['name']}"
        avs_path = run_ctx["avs_dir"] / f"{safe(tag)}.avs"
        avs_path.write_text(
            build_raw_avs(case_ctx["archive_path"], w["start_frame"], w["end_frame"]),
            encoding="utf-8",
        )
        metrics = compute_avs_metrics(
            avs_path,
            run_ctx["log_dir"],
            tag,
            metric_ctx,
            stage_cfg,
            with_iqa=with_iqa,
        )
        baseline_map[w["name"]] = metrics
        print(
            f"  Baseline {stage_cfg['name']} {w['name']}: "
            f"stability={metrics['stability_score']:.6f} "
            f"detail={metrics['detail_mean']:.6f} "
            f"temporal={metrics['temporal_mean']:.6f} "
            f"brisque={metrics['brisque_mean']:.6f}"
        )
    return baseline_map


def evaluate_candidate(case_ctx, run_ctx, qtgmc_opts, sharpness, eval_index, stage_cfg, metric_ctx, with_iqa):
    opts = dict(qtgmc_opts)
    opts["Sharpness"] = round(float(sharpness), 4)
    patched_filter = rewrite_filter_qtgmc(case_ctx["filter_text"], opts)

    window_results = []
    for w in run_ctx["windows"]:
        tag = (
            f"{case_ctx['name']}_{stage_cfg['name']}_{run_ctx['profile_name']}_"
            f"{eval_index:03d}_{w['name']}_s{opts['Sharpness']:.4f}"
        )
        avs_path = run_ctx["avs_dir"] / f"{safe(tag)}.avs"
        avs_path.write_text(
            build_avs(case_ctx["archive_path"], w["start_frame"], w["end_frame"], patched_filter),
            encoding="utf-8",
        )
        metrics = compute_avs_metrics(
            avs_path,
            run_ctx["log_dir"],
            tag,
            metric_ctx,
            stage_cfg,
            with_iqa=with_iqa,
        )
        baseline = run_ctx["baseline_by_window"][w["name"]]
        score_bits = score_components(metrics, baseline, stage_cfg["weights"])
        one = dict(metrics)
        one.update(score_bits)
        one.update(
            {
                "window_name": w["name"],
                "window_start_frame": w["start_frame"],
                "window_end_frame": w["end_frame"],
                "avs_path": str(avs_path),
            }
        )
        window_results.append(one)

    agg = aggregate_window_results(window_results)
    agg.update(
        {
            "sharpness": opts["Sharpness"],
            "profile_name": run_ctx["profile_name"],
        }
    )
    return agg


def search_sharpness(case_ctx, run_ctx, qtgmc_opts, low, high, iterations, stage_cfg, metric_ctx, with_iqa):
    cache = {}
    eval_counter = 0

    def eval_at(value):
        nonlocal eval_counter
        key = round(float(value), 4)
        if key not in cache:
            eval_counter += 1
            cache[key] = evaluate_candidate(
                case_ctx,
                run_ctx,
                qtgmc_opts,
                key,
                eval_counter,
                stage_cfg,
                metric_ctx,
                with_iqa=with_iqa,
            )
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


def qtgmc_profile_grid():
    profiles = []
    for preset, source_match, tr2 in itertools.product(PRESETS, SOURCEMATCH_VALUES, TR2_VALUES):
        profiles.append(
            {
                "Preset": preset,
                "EZKeepGrain": 1.0,
                "SourceMatch": source_match,
                "Lossless": 2 if tr2 >= 3 else 1,
                "TR2": tr2,
                "MatchEdi": "NNEDI3",
                "MatchEdi2": "NNEDI3",
            }
        )
    return profiles


def profile_name(profile):
    preset = str(profile["Preset"]).lower().replace(" ", "")
    return f"{preset}_sm{profile['SourceMatch']}_tr2{profile['TR2']}_ll{profile['Lossless']}"


def write_case_results(stage_dir, stage_name, case_results):
    rows = [
        "rank\tprofile\tpreset\tsource_match\ttr2\tlossless\tbest_sharpness\t"
        "score\tstability_ratio\tdetail_ratio\tdetail_penalty\tbrisque_ratio\tbrisque_penalty\t"
        "temporal_ratio\tsmoothness_penalty\tiqa_ratio\tiqa_penalty\tstability_score\t"
        "x_std\ty_std\tw_std\th_std\tsamples\tevals"
    ]
    for i, r in enumerate(case_results, start=1):
        best = r["best"]
        rows.append(
            "\t".join(
                [
                    str(i),
                    r["profile_name"],
                    str(r["profile"]["Preset"]),
                    str(r["profile"]["SourceMatch"]),
                    str(r["profile"]["TR2"]),
                    str(r["profile"]["Lossless"]),
                    f"{best['sharpness']:.4f}",
                    f"{best['score']:.6f}",
                    f"{best['stability_ratio']:.6f}",
                    f"{best['detail_ratio']:.6f}",
                    f"{best['detail_penalty']:.6f}",
                    f"{best['brisque_ratio']:.6f}",
                    f"{best['brisque_penalty']:.6f}",
                    f"{best['temporal_ratio']:.6f}",
                    f"{best['smoothness_penalty']:.6f}",
                    (f"{best['iqa_ratio']:.6f}" if best.get("iqa_ratio") is not None else ""),
                    f"{best['iqa_penalty']:.6f}",
                    f"{best['stability_score']:.6f}",
                    f"{best['x_std']:.6f}",
                    f"{best['y_std']:.6f}",
                    f"{best['w_std']:.6f}",
                    f"{best['h_std']:.6f}",
                    str(best["samples"]),
                    str(len(r["history"])),
                ]
            )
        )
    (stage_dir / f"{stage_name}_rankings.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

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
    (stage_dir / f"{stage_name}_history.json").write_text(
        json.dumps(history_dump, indent=2),
        encoding="utf-8",
    )


def encode_preview(case_ctx, window, qtgmc_opts, sharpness, out_path):
    opts = dict(qtgmc_opts)
    opts["Sharpness"] = round(float(sharpness), 4)
    patched_filter = rewrite_filter_qtgmc(case_ctx["filter_text"], opts)
    avs_text = build_avs(
        case_ctx["archive_path"],
        window["start_frame"],
        window["end_frame"],
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


def encode_control_preview(case_ctx, window, out_path):
    avs_text = build_raw_avs(
        case_ctx["archive_path"],
        window["start_frame"],
        window["end_frame"],
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

def write_archive_summary(out_dir, archive_best_entries):
    rows = [
        "archive\tbest_profile\tpreset\tsource_match\ttr2\tlossless\tmedian_sharpness\tmean_score\tcases"
    ]

    by_archive = {}
    for row in archive_best_entries:
        by_archive.setdefault(row["archive"], []).append(row)

    for archive, rows_for_archive in sorted(by_archive.items()):
        by_profile = {}
        for row in rows_for_archive:
            pname = row["best"]["profile_name"]
            if pname not in by_profile:
                by_profile[pname] = {
                    "profile": row["best"]["profile"],
                    "scores": [],
                    "sharpness": [],
                    "cases": [],
                }
            by_profile[pname]["scores"].append(float(row["best"]["score"]))
            by_profile[pname]["sharpness"].append(float(row["best"]["sharpness"]))
            by_profile[pname]["cases"].append(row["case_name"])

        best_profile_name = None
        best_profile_stats = None
        for pname, stats in by_profile.items():
            mean_score = statistics.fmean(stats["scores"])
            if (best_profile_stats is None) or (mean_score < best_profile_stats["mean_score"]):
                best_profile_name = pname
                best_profile_stats = {
                    "mean_score": mean_score,
                    "median_sharpness": statistics.median(stats["sharpness"]),
                    "cases": sorted(stats["cases"]),
                    "profile": stats["profile"],
                }

        prof = best_profile_stats["profile"]
        rows.append(
            "\t".join(
                [
                    archive,
                    best_profile_name,
                    str(prof["Preset"]),
                    str(prof["SourceMatch"]),
                    str(prof["TR2"]),
                    str(prof["Lossless"]),
                    f"{best_profile_stats['median_sharpness']:.4f}",
                    f"{best_profile_stats['mean_score']:.6f}",
                    ",".join(best_profile_stats["cases"]),
                ]
            )
        )

    (out_dir / "best_settings_per_archive.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def stage_config(name):
    if name == "stage1":
        return {
            "name": "stage1",
            "weights": STAGE1_WEIGHTS,
            "brisque_frame_stride": BRISQUE_FRAME_STRIDE_STAGE1,
            "iqa_frame_stride": IQA_FRAME_STRIDE_STAGE2,
            "search_iters": STAGE1_SEARCH_ITERS,
        }
    return {
        "name": "stage2",
        "weights": STAGE2_WEIGHTS,
        "brisque_frame_stride": BRISQUE_FRAME_STRIDE_STAGE2,
        "iqa_frame_stride": IQA_FRAME_STRIDE_STAGE2,
        "search_iters": STAGE2_SEARCH_ITERS,
    }


def main():
    ensure_ffmpeg_exists()

    metric_ctx = {
        "brisque": init_brisque_context(),
        "iqa": init_iqa_metric(),
    }

    base_dir = CLIPS_DIR / "filter_tests" / "qtgmc_quality_pipeline"
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
    print(f"Stage1 profiles: {len(profiles)} | Stage2 rerank top-N: {STAGE2_TOP_N}")

    overall_rows = [
        "case\tarchive\tchapter\tbest_profile\tbest_sharpness\tscore\tstability_ratio\t"
        "detail_ratio\ttemporal_ratio\tbrisque_ratio\tsamples"
    ]
    archive_best_entries = []

    s1_cfg = stage_config("stage1")
    s2_cfg = stage_config("stage2")

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

        stage1_windows, stage2_windows = build_stage_windows(ch, case)

        case_dir = out_dir / safe(case_name)
        preview_dir = case_dir / "previews"
        s1_dir = case_dir / "stage1"
        s2_dir = case_dir / "stage2"
        for d in (preview_dir, s1_dir, s2_dir):
            d.mkdir(parents=True, exist_ok=True)
        s1_avs_dir = s1_dir / "eval_avs"
        s1_log_dir = s1_dir / "eval_logs"
        s2_avs_dir = s2_dir / "eval_avs"
        s2_log_dir = s2_dir / "eval_logs"
        for d in (s1_avs_dir, s1_log_dir, s2_avs_dir, s2_log_dir):
            d.mkdir(parents=True, exist_ok=True)

        case_ctx = {
            "name": case_name,
            "archive_stem": archive_stem,
            "archive_path": archive_path,
            "chapter_title": ch.get("title", ""),
            "filter_text": filter_path.read_text(encoding="utf-8"),
        }
        print(
            f"\nCase {case_name}: {archive_stem} | {case_ctx['chapter_title']} | "
            f"primary frames {stage1_windows[0]['start_frame']}-{stage1_windows[0]['end_frame']}"
        )

        stage1_base_ctx = {
            "avs_dir": s1_avs_dir,
            "log_dir": s1_log_dir,
        }
        stage1_baselines = prepare_baselines(
            case_ctx,
            stage1_windows,
            stage1_base_ctx,
            s1_cfg,
            metric_ctx,
            with_iqa=False,
        )

        stage1_results = []
        for idx, profile in enumerate(profiles, start=1):
            pname = profile_name(profile)
            print(f"  [S1 {idx:02d}/{len(profiles)}] searching {pname}")
            run_ctx = {
                "profile_name": pname,
                "avs_dir": s1_avs_dir,
                "log_dir": s1_log_dir,
                "windows": stage1_windows,
                "baseline_by_window": stage1_baselines,
            }
            best, history = search_sharpness(
                case_ctx,
                run_ctx,
                profile,
                SHARPNESS_MIN,
                SHARPNESS_MAX,
                s1_cfg["search_iters"],
                s1_cfg,
                metric_ctx,
                with_iqa=False,
            )
            stage1_results.append(
                {
                    "profile_name": pname,
                    "profile": profile,
                    "best": best,
                    "history": history,
                }
            )

        stage1_results.sort(key=lambda r: r["best"]["score"])
        write_case_results(s1_dir, "stage1", stage1_results)

        stage2_candidates = stage1_results[: min(STAGE2_TOP_N, len(stage1_results))]
        print(f"  Stage2 candidates: {len(stage2_candidates)}")

        stage2_base_ctx = {
            "avs_dir": s2_avs_dir,
            "log_dir": s2_log_dir,
        }
        use_iqa_stage2 = metric_ctx.get("iqa") is not None
        stage2_baselines = prepare_baselines(
            case_ctx,
            stage2_windows,
            stage2_base_ctx,
            s2_cfg,
            metric_ctx,
            with_iqa=use_iqa_stage2,
        )

        stage2_results = []
        for idx, item in enumerate(stage2_candidates, start=1):
            profile = item["profile"]
            pname = item["profile_name"]
            print(f"  [S2 {idx:02d}/{len(stage2_candidates)}] searching {pname}")
            run_ctx = {
                "profile_name": pname,
                "avs_dir": s2_avs_dir,
                "log_dir": s2_log_dir,
                "windows": stage2_windows,
                "baseline_by_window": stage2_baselines,
            }
            best, history = search_sharpness(
                case_ctx,
                run_ctx,
                profile,
                SHARPNESS_MIN,
                SHARPNESS_MAX,
                s2_cfg["search_iters"],
                s2_cfg,
                metric_ctx,
                with_iqa=use_iqa_stage2,
            )
            stage2_results.append(
                {
                    "profile_name": pname,
                    "profile": profile,
                    "best": best,
                    "history": history,
                }
            )

        stage2_results.sort(key=lambda r: r["best"]["score"])
        write_case_results(s2_dir, "stage2", stage2_results)

        control_mp4 = preview_dir / "00_control_raw.mp4"
        preview_window = stage1_windows[0]
        encode_control_preview(case_ctx, preview_window, control_mp4)
        preview_inputs = [control_mp4]
        for rank, r in enumerate(stage2_results[:3], start=1):
            s = r["best"]["sharpness"]
            out_name = safe(
                f"{rank:02d}_{r['profile_name']}_s{s:.4f}_score{r['best']['score']:.4f}"
            )
            preview_mp4 = preview_dir / f"{out_name}.mp4"
            encode_preview(case_ctx, preview_window, r["profile"], s, preview_mp4)
            preview_inputs.append(preview_mp4)

        if len(preview_inputs) == 4:
            quadrant = case_dir / "quadrant_top4.mp4"
            make_quadrant(preview_inputs, quadrant)
            print(f"  Wrote quadrant: {quadrant}")

        best = stage2_results[0]
        overall_rows.append(
            "\t".join(
                [
                    case_name,
                    archive_stem,
                    case_ctx["chapter_title"],
                    best["profile_name"],
                    f"{best['best']['sharpness']:.4f}",
                    f"{best['best']['score']:.6f}",
                    f"{best['best']['stability_ratio']:.6f}",
                    f"{best['best']['detail_ratio']:.6f}",
                    f"{best['best']['temporal_ratio']:.6f}",
                    f"{best['best']['brisque_ratio']:.6f}",
                    str(best["best"]["samples"]),
                ]
            )
        )
        archive_best_entries.append(
            {
                "archive": archive_stem,
                "case_name": case_name,
                "best": {
                    "profile_name": best["profile_name"],
                    "profile": best["profile"],
                    "sharpness": best["best"]["sharpness"],
                    "score": best["best"]["score"],
                },
            }
        )
        print(
            f"  Best: {best['profile_name']} sharpness={best['best']['sharpness']:.4f} "
            f"score={best['best']['score']:.6f} "
            f"stability={best['best']['stability_ratio']:.4f} "
            f"detail={best['best']['detail_ratio']:.4f} "
            f"smooth={best['best']['temporal_ratio']:.4f} "
            f"brisque={best['best']['brisque_ratio']:.4f}"
        )

    (out_dir / "overall_best.tsv").write_text("\n".join(overall_rows) + "\n", encoding="utf-8")
    write_archive_summary(out_dir, archive_best_entries)
    print("\nDone.")


if __name__ == "__main__":
    main()
