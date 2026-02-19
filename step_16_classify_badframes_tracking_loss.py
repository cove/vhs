#!/usr/bin/env python3.11
#
# Tracking-loss bad frame classification for VHS archives.
# Uses intra-frame artifact signals that remain effective even when multiple
# consecutive frames are degraded:
#   1) Horizontal edge energy (Sobel Y)
#   2) Scanline luma instability (adjacent row mean differences)
#   3) Field mismatch (even/odd line absolute difference)
#
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import ARCHIVE_DIR, METADATA_DIR


DEFAULT_ARCHIVE = "callahan_01_archive"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Classify VHS frames as good/bad using tracking-loss specific "
            "scanline and field artifact signals."
        )
    )
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE,
        help=f"Archive stem used for default paths (default: {DEFAULT_ARCHIVE}).",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Video path. Default: ../Archive/<archive>.mkv",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output folder. Default: metadata/<archive>/tracking_badframe",
    )
    parser.add_argument(
        "--scores-tsv",
        default="",
        help="Optional existing frame_scores.tsv to reuse (skips video scoring).",
    )
    parser.add_argument(
        "--existing-badframes",
        default="",
        help="Optional badframes TSV for comparison. Default: metadata/<archive>/badframes.tsv",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to evaluate (default: 0).",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        default=-1,
        help="Last frame index to evaluate, inclusive (-1 = video end).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Evaluate every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--crop-top",
        type=int,
        default=0,
        help="Crop this many rows from top before scoring (default: 0).",
    )
    parser.add_argument(
        "--crop-bottom",
        type=int,
        default=0,
        help="Crop this many rows from bottom before scoring (default: 0).",
    )
    parser.add_argument(
        "--crop-left",
        type=int,
        default=0,
        help="Crop this many columns from left before scoring (default: 0).",
    )
    parser.add_argument(
        "--crop-right",
        type=int,
        default=0,
        help="Crop this many columns from right before scoring (default: 0).",
    )
    parser.add_argument(
        "--sobel-ksize",
        type=int,
        default=3,
        help="Sobel kernel size for horizontal edge energy (odd, 1/3/5/7; default: 3).",
    )
    parser.add_argument(
        "--weight-edge",
        type=float,
        default=0.45,
        help="Composite score weight for horizontal edge energy signal (default: 0.45).",
    )
    parser.add_argument(
        "--weight-row",
        type=float,
        default=0.25,
        help="Composite score weight for scanline luma instability signal (default: 0.25).",
    )
    parser.add_argument(
        "--weight-field",
        type=float,
        default=0.30,
        help="Composite score weight for field mismatch signal (default: 0.30).",
    )
    parser.add_argument(
        "--otsu-bins",
        type=int,
        default=256,
        help="Histogram bins for Otsu thresholding (default: 256).",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("otsu", "quantile", "value"),
        default="otsu",
        help="Threshold mode for bad/good split (default: otsu).",
    )
    parser.add_argument(
        "--bad-rate",
        type=float,
        default=-1.0,
        help="When threshold-mode=quantile, label this fraction as bad (0<rate<1).",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=np.nan,
        help="When threshold-mode=value, explicit threshold; score>=threshold => bad.",
    )
    parser.add_argument(
        "--calibrate-bad-rate-from-existing",
        action="store_true",
        help="In quantile mode, infer bad-rate from existing badframes in the evaluated window.",
    )
    parser.add_argument(
        "--export-bad-png-count",
        type=int,
        default=0,
        help="Export up to N evenly spaced predicted bad frames as PNG (default: 0, disabled).",
    )
    parser.add_argument(
        "--png-output-dir",
        default="",
        help="PNG sample output folder. Default: <output-dir>/badframe_samples",
    )
    return parser.parse_args(argv)


def parse_badframe_ranges(tsv_path):
    ranges = []
    if not tsv_path or not Path(tsv_path).exists():
        return ranges

    for raw_line in Path(tsv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            continue
        if end < start:
            start, end = end, start
        ranges.append((max(0, start), max(0, end)))
    return ranges


def expand_ranges_to_set(ranges, min_frame, max_frame):
    out = set()
    if max_frame < min_frame:
        return out
    for start, end in ranges:
        lo = max(min_frame, int(start))
        hi = min(max_frame, int(end))
        if hi < lo:
            continue
        for frame_idx in range(lo, hi + 1):
            out.add(frame_idx)
    return out


def ranges_from_sorted_frames(frame_ids):
    if not frame_ids:
        return []
    ranges = []
    start = prev = int(frame_ids[0])
    for value in frame_ids[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def otsu_threshold(values, bins=256):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("Cannot compute Otsu threshold on empty array.")

    finite_mask = np.isfinite(vals)
    vals = vals[finite_mask]
    if vals.size == 0:
        raise ValueError("Cannot compute Otsu threshold: all values are non-finite.")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmin == vmax:
        return vmin

    hist, edges = np.histogram(vals, bins=max(2, int(bins)), range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]

    between = weight1[:-1] * weight2[1:] * np.square(mean1[:-1] - mean2[1:])
    if between.size == 0:
        return float(np.median(vals))
    best_idx = int(np.argmax(between))
    return float(centers[best_idx])


def robust_zscore(values):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        raise ValueError("Cannot normalize empty signal.")

    center = float(np.median(vals))
    mad = float(np.median(np.abs(vals - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        std = float(np.std(vals))
        scale = std if std > 1e-12 else 1.0

    z = (vals - center) / scale
    return z, center, scale


def sanitize_sobel_ksize(ksize):
    k = int(ksize)
    if k <= 0:
        return 3
    if k % 2 == 0:
        k += 1
    return max(1, min(k, 31))


def crop_frame(gray, top, bottom, left, right):
    h, w = gray.shape[:2]
    t = max(0, int(top))
    b = max(0, int(bottom))
    l = max(0, int(left))
    r = max(0, int(right))

    y0 = min(t, max(0, h - 1))
    y1 = max(y0 + 1, h - b)
    x0 = min(l, max(0, w - 1))
    x1 = max(x0 + 1, w - r)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return gray
    return roi


def compute_tracking_signals(gray, sobel_ksize):
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    edge_energy = float(np.mean(np.abs(sobel_y)))

    row_means = gray.mean(axis=1, dtype=np.float32)
    if row_means.shape[0] > 1:
        row_instability = float(np.mean(np.abs(np.diff(row_means))))
    else:
        row_instability = 0.0

    even = gray[0::2, :].astype(np.float32)
    odd = gray[1::2, :].astype(np.float32)
    paired_rows = min(even.shape[0], odd.shape[0])
    if paired_rows > 0:
        field_mismatch = float(np.mean(np.abs(even[:paired_rows] - odd[:paired_rows])))
    else:
        field_mismatch = 0.0

    return edge_energy, row_instability, field_mismatch


def score_video_frames(
    video_path,
    start_frame,
    max_frame,
    frame_step,
    crop_top,
    crop_bottom,
    crop_left,
    crop_right,
    sobel_ksize,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read frame count for: {video_path}")

    start = max(0, int(start_frame))
    end = total_frames - 1 if int(max_frame) < 0 else min(total_frames - 1, int(max_frame))
    step = max(1, int(frame_step))

    if start > end:
        cap.release()
        raise ValueError(f"start-frame ({start}) is after max-frame ({end}).")

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    target_count = ((end - start) // step) + 1
    pbar = tqdm(total=target_count, desc="Scoring frames", unit="frame")

    indices = []
    edge_scores = []
    row_scores = []
    field_scores = []

    frame_idx = start
    while frame_idx <= end:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if ((frame_idx - start) % step) == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            roi = crop_frame(
                gray,
                top=crop_top,
                bottom=crop_bottom,
                left=crop_left,
                right=crop_right,
            )
            edge_energy, row_instability, field_mismatch = compute_tracking_signals(
                roi, sobel_ksize=sobel_ksize
            )
            indices.append(int(frame_idx))
            edge_scores.append(float(edge_energy))
            row_scores.append(float(row_instability))
            field_scores.append(float(field_mismatch))
            pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()

    if not indices:
        raise RuntimeError("No frame scores produced.")
    return total_frames, start, end, indices, edge_scores, row_scores, field_scores


def combine_signals(edge_scores, row_scores, field_scores, weight_edge, weight_row, weight_field):
    w_edge = float(weight_edge)
    w_row = float(weight_row)
    w_field = float(weight_field)
    weight_sum = w_edge + w_row + w_field
    if weight_sum <= 0.0:
        raise ValueError("At least one signal weight must be > 0.")

    edge_z, edge_center, edge_scale = robust_zscore(edge_scores)
    row_z, row_center, row_scale = robust_zscore(row_scores)
    field_z, field_center, field_scale = robust_zscore(field_scores)

    score = (
        w_edge * edge_z +
        w_row * row_z +
        w_field * field_z
    ) / weight_sum

    signal_norm = {
        "edge": {"center": float(edge_center), "scale": float(edge_scale)},
        "row": {"center": float(row_center), "scale": float(row_scale)},
        "field": {"center": float(field_center), "scale": float(field_scale)},
    }
    return score.astype(np.float64), signal_norm


def _try_parse_float(text):
    try:
        return float(text)
    except Exception:
        return np.nan


def load_scores_tsv(scores_tsv_path):
    rows = []
    header_map = {}

    for raw_line in Path(scores_tsv_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        lowered = [p.strip().lower() for p in parts]
        if lowered and lowered[0] == "frame":
            header_map = {name: idx for idx, name in enumerate(lowered)}
            continue

        if header_map:
            frame_idx_col = header_map.get("frame", 0)
            score_col = header_map.get("score", 1)
            edge_col = header_map.get("edge_energy", -1)
            row_col = header_map.get("row_instability", -1)
            field_col = header_map.get("field_mismatch", -1)
        else:
            frame_idx_col = 0
            score_col = 1
            edge_col = -1
            row_col = -1
            field_col = -1

        if len(parts) <= max(frame_idx_col, score_col):
            continue
        try:
            frame_idx = int(parts[frame_idx_col])
            score = float(parts[score_col])
        except ValueError:
            continue

        edge_energy = _try_parse_float(parts[edge_col]) if edge_col >= 0 and edge_col < len(parts) else np.nan
        row_instability = _try_parse_float(parts[row_col]) if row_col >= 0 and row_col < len(parts) else np.nan
        field_mismatch = _try_parse_float(parts[field_col]) if field_col >= 0 and field_col < len(parts) else np.nan
        rows.append((frame_idx, score, edge_energy, row_instability, field_mismatch))

    if not rows:
        raise ValueError(f"No frame/score rows found in {scores_tsv_path}")

    rows.sort(key=lambda x: x[0])
    indices = [int(r[0]) for r in rows]
    scores = [float(r[1]) for r in rows]
    edge_scores = [float(r[2]) for r in rows]
    row_scores = [float(r[3]) for r in rows]
    field_scores = [float(r[4]) for r in rows]
    return indices, scores, edge_scores, row_scores, field_scores


def pick_evenly_spaced_samples(frame_ids, count):
    ordered = [int(x) for x in frame_ids]
    sample_count = max(0, int(count))
    if sample_count <= 0 or not ordered:
        return []
    if sample_count >= len(ordered):
        return ordered

    chosen = []
    seen = set()
    for idx in np.linspace(0, len(ordered) - 1, num=sample_count, dtype=int).tolist():
        frame_id = int(ordered[int(idx)])
        if frame_id in seen:
            continue
        seen.add(frame_id)
        chosen.append(frame_id)

    if len(chosen) < sample_count:
        for frame_id in ordered:
            if frame_id in seen:
                continue
            seen.add(frame_id)
            chosen.append(frame_id)
            if len(chosen) >= sample_count:
                break
    return chosen


def export_frame_png_samples(video_path, frame_ids, sample_dir, prefix="bad_sample"):
    sample_dir.mkdir(parents=True, exist_ok=True)
    if not frame_ids:
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for PNG export: {video_path}")

    written = []
    failed_frames = []
    try:
        for sample_idx, frame_idx in enumerate(frame_ids, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                failed_frames.append(int(frame_idx))
                continue

            out_path = sample_dir / f"{prefix}_{sample_idx:02d}_frame_{int(frame_idx)}.png"
            if not cv2.imwrite(str(out_path), frame_bgr):
                failed_frames.append(int(frame_idx))
                continue
            written.append(out_path)
    finally:
        cap.release()
    return written, failed_frames


def write_outputs(
    output_dir,
    indices,
    scores,
    edge_scores,
    row_scores,
    field_scores,
    labels,
    bad_ranges,
    summary,
    note="tracking_loss",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_scores_path = output_dir / "frame_scores.tsv"
    with frame_scores_path.open("w", encoding="utf-8") as f:
        f.write("frame\tscore\tedge_energy\trow_instability\tfield_mismatch\tlabel\n")
        for frame_idx, score, edge, row, field, label in zip(
            indices, scores, edge_scores, row_scores, field_scores, labels
        ):
            f.write(
                f"{frame_idx}\t{score:.8f}\t{edge:.8f}\t{row:.8f}\t{field:.8f}\t{label}\n"
            )

    badframes_path = output_dir / "badframes.tsv"
    with badframes_path.open("w", encoding="utf-8") as f:
        f.write("start_frame\tend_frame\tnote\n")
        for start, end in bad_ranges:
            f.write(f"{start}\t{end}\t{note}\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return frame_scores_path, badframes_path, summary_path


def finite_stats(values):
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size <= 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def main(argv=None):
    args = parse_args(argv)

    archive_name = args.archive.strip()
    if not archive_name:
        raise ValueError("--archive cannot be empty.")

    sobel_ksize = sanitize_sobel_ksize(args.sobel_ksize)

    video_path = Path(args.video) if args.video else (ARCHIVE_DIR / f"{archive_name}.mkv")
    output_dir = Path(args.output_dir) if args.output_dir else (METADATA_DIR / archive_name / "tracking_badframe")
    png_output_dir = (
        Path(args.png_output_dir) if args.png_output_dir else (output_dir / "badframe_samples")
    )
    existing_badframes = (
        Path(args.existing_badframes)
        if args.existing_badframes
        else (METADATA_DIR / archive_name / "badframes.tsv")
    )
    scores_tsv_path = Path(args.scores_tsv) if args.scores_tsv else None

    indices = []
    edge_scores = []
    row_scores = []
    field_scores = []
    signal_norm = None
    total_frames = None

    if scores_tsv_path:
        if not scores_tsv_path.exists():
            raise FileNotFoundError(f"Scores TSV not found: {scores_tsv_path}")
        indices, scores, edge_scores, row_scores, field_scores = load_scores_tsv(scores_tsv_path)
        start_frame = min(indices)
        end_frame = max(indices)
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        print(f"Loaded {len(indices)} frame scores from: {scores_tsv_path}")
    else:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        total_frames, start_frame, end_frame, indices, edge_scores, row_scores, field_scores = score_video_frames(
            video_path=video_path,
            start_frame=args.start_frame,
            max_frame=args.max_frame,
            frame_step=args.frame_step,
            crop_top=args.crop_top,
            crop_bottom=args.crop_bottom,
            crop_left=args.crop_left,
            crop_right=args.crop_right,
            sobel_ksize=sobel_ksize,
        )
        scores, signal_norm = combine_signals(
            edge_scores=edge_scores,
            row_scores=row_scores,
            field_scores=field_scores,
            weight_edge=args.weight_edge,
            weight_row=args.weight_row,
            weight_field=args.weight_field,
        )
        scores = scores.astype(np.float64).tolist()

    scores_np = np.asarray(scores, dtype=np.float64)
    evaluated_set = set(indices)

    manual_ranges = []
    manual_bad_eval = set()
    if existing_badframes.exists():
        manual_ranges = parse_badframe_ranges(existing_badframes)
        manual_bad = expand_ranges_to_set(manual_ranges, start_frame, end_frame)
        manual_bad_eval = evaluated_set.intersection(manual_bad)

    threshold_mode = args.threshold_mode
    threshold_source = threshold_mode
    target_bad_rate = None
    if threshold_mode == "otsu":
        threshold = otsu_threshold(np.sort(scores_np), bins=args.otsu_bins)
    elif threshold_mode == "quantile":
        target_bad_rate = float(args.bad_rate)
        if args.calibrate_bad_rate_from_existing:
            if not existing_badframes.exists():
                raise ValueError(
                    "--calibrate-bad-rate-from-existing requested, but existing badframes file does not exist."
                )
            if len(manual_bad_eval) <= 0:
                raise ValueError(
                    "--calibrate-bad-rate-from-existing requested, but no manual bad frames were found in the evaluated window."
                )
            target_bad_rate = float(len(manual_bad_eval) / max(1, len(evaluated_set)))
            threshold_source = "quantile_from_existing_badframes"
        if not (0.0 < target_bad_rate < 1.0):
            raise ValueError(
                "For --threshold-mode quantile, use --bad-rate between 0 and 1 "
                "(or pass --calibrate-bad-rate-from-existing)."
            )
        threshold = float(np.quantile(scores_np, 1.0 - target_bad_rate))
    elif threshold_mode == "value":
        threshold = float(args.threshold_value)
        if not np.isfinite(threshold):
            raise ValueError("--threshold-mode value requires a finite --threshold-value.")
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    labels = ["good" if score < threshold else "bad" for score in scores]
    bad_frames = sorted(frame for frame, label in zip(indices, labels) if label == "bad")
    bad_ranges = ranges_from_sorted_frames(bad_frames)

    requested_png_count = max(0, int(args.export_bad_png_count))
    selected_sample_frames = []
    sample_png_paths = []
    failed_sample_frames = []
    if requested_png_count > 0:
        if not video_path.exists():
            print(f"WARNING: skipping PNG export; video file not found: {video_path}")
        else:
            selected_sample_frames = pick_evenly_spaced_samples(bad_frames, requested_png_count)
            try:
                sample_png_paths, failed_sample_frames = export_frame_png_samples(
                    video_path=video_path,
                    frame_ids=selected_sample_frames,
                    sample_dir=png_output_dir,
                )
            except Exception as exc:
                print(f"WARNING: PNG export failed: {exc}")

    summary = {
        "archive": archive_name,
        "video_path": str(video_path),
        "detector": "tracking_loss_scanline_field",
        "total_video_frames": (None if total_frames is None else int(total_frames)),
        "evaluated_frame_start": int(start_frame),
        "evaluated_frame_end": int(end_frame),
        "evaluated_frame_step": int(max(1, args.frame_step)),
        "evaluated_frames": int(len(indices)),
        "crop": {
            "top": int(max(0, args.crop_top)),
            "bottom": int(max(0, args.crop_bottom)),
            "left": int(max(0, args.crop_left)),
            "right": int(max(0, args.crop_right)),
        },
        "signals": {
            "sobel_ksize": int(sobel_ksize),
            "weights": {
                "edge": float(args.weight_edge),
                "row": float(args.weight_row),
                "field": float(args.weight_field),
            },
            "normalization": signal_norm,
            "edge_energy": finite_stats(edge_scores),
            "row_instability": finite_stats(row_scores),
            "field_mismatch": finite_stats(field_scores),
        },
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        "score_threshold": float(threshold),
        "target_bad_rate": (None if target_bad_rate is None else float(target_bad_rate)),
        "score_min": float(np.min(scores_np)),
        "score_max": float(np.max(scores_np)),
        "score_mean": float(np.mean(scores_np)),
        "good_frames": int(sum(1 for x in labels if x == "good")),
        "bad_frames": int(sum(1 for x in labels if x == "bad")),
        "predicted_bad_ranges": int(len(bad_ranges)),
        "png_samples": {
            "requested_bad_samples": int(requested_png_count),
            "selected_bad_frames": int(len(selected_sample_frames)),
            "written_pngs": int(len(sample_png_paths)),
            "failed_frames": [int(x) for x in failed_sample_frames],
            "output_dir": str(png_output_dir) if requested_png_count > 0 else None,
        },
    }
    if threshold_mode == "otsu":
        summary["score_threshold_otsu"] = float(threshold)

    if existing_badframes.exists():
        predicted_bad = set(bad_frames)
        tp = len(predicted_bad.intersection(manual_bad_eval))
        fp = len(predicted_bad - manual_bad_eval)
        fn = len(manual_bad_eval - predicted_bad)
        tn = len(evaluated_set) - tp - fp - fn

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        summary["comparison_to_existing_badframes"] = {
            "path": str(existing_badframes),
            "manual_bad_ranges": int(len(manual_ranges)),
            "manual_bad_frames_in_window": int(len(manual_bad_eval)),
            "predicted_bad_frames_in_window": int(len(predicted_bad)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    else:
        summary["comparison_to_existing_badframes"] = {
            "path": str(existing_badframes),
            "note": "file not found; comparison skipped",
        }

    frame_scores_path, badframes_path, summary_path = write_outputs(
        output_dir=output_dir,
        indices=indices,
        scores=scores_np.tolist(),
        edge_scores=edge_scores,
        row_scores=row_scores,
        field_scores=field_scores,
        labels=labels,
        bad_ranges=bad_ranges,
        summary=summary,
        note=f"tracking_loss_{threshold_mode}",
    )

    print(f"Frame scores: {frame_scores_path}")
    print(f"Predicted badframe ranges: {badframes_path}")
    print(f"Summary: {summary_path}")
    if requested_png_count > 0:
        print(f"PNG samples: {png_output_dir} ({len(sample_png_paths)} written)")


if __name__ == "__main__":
    main()
