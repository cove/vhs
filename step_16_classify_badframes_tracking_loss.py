#!/usr/bin/env python3.11
#
# Tracking-loss bad frame classification for VHS archives.
# Uses intra-frame artifact signals that remain effective even when multiple
# consecutive frames are degraded:
#   1) Horizontal edge energy (Sobel Y)
#   2) Scanline luma instability (adjacent row mean differences)
#   3) Field mismatch (even/odd line absolute difference)
#

"""
## Key Parameters to Experiment With

### The Three Signals & Their Weights
These are your biggest levers:

- **`weight_edge`** (default 0.45) — Sobel Y energy; detects horizontal tearing/banding from tracking loss
- **`weight_row`** (default 0.25) — Adjacent row mean differences; catches luma instability across scanlines
- **`weight_field`** (default 0.30) — Even/odd field mismatch; catches interlacing artifacts when heads mistrack

Try zeroing out two weights and leaving one at 1.0 to see what each signal is actually "seeing" in isolation. This tells you which signal is doing the real work (or not).

### Threshold Mode
This controls *how* the bad/good line is drawn:

- **`otsu`** (default) — Automatic, finds the natural valley in the score histogram. Good starting point but assumes a bimodal distribution, which may not hold for your tape.
- **`quantile` + `bad_rate`** — You say "I expect X% of frames to be bad." More direct control. Try `bad_rate=0.05` if you think ~5% are bad.
- **`value` + `threshold_value`** — Hard-code a threshold after inspecting the `frame_scores.tsv` output.

### Crop Parameters
`crop_top`, `crop_bottom`, `crop_left`, `crop_right` — VHS tracking noise is heavily concentrated in the top and bottom edges of the frame. Cropping those out before scoring can dramatically clean up your signals, since otherwise stable content edge artifacts may be drowning out real tracking events.

---

## Why Sobel Ksize Doesn't Matter Much

This is actually expected behavior. The edge *energy* signal (`mean(abs(sobel_y))`) is an **average over the whole frame** — it's a scalar summary, not a spatial detection. Large vs. small kernels both pick up the same dominant horizontal energy pattern; the kernel size mainly affects sensitivity to fine vs. coarse edges, but at the aggregate mean level these wash out.

More importantly, your three signals are all **highly correlated by design** — a bad tracking frame will simultaneously spike all three. So if the same frames are bad across ksize 1 vs. 31, it likely means:

1. The bad frames are genuinely obvious (large magnitude artifacts) and any reasonable signal catches them, or
2. The **threshold** is the real discriminator, not signal sensitivity — Otsu may be drawing the line at roughly the same place regardless

**What to actually try:** Rather than tweaking ksize, use the `intution` mode across a known-bad segment, then open the `frame_scores.tsv` and plot or inspect the score distribution. If you see a clean bimodal split, Otsu is fine. If scores are smeared, switch to `quantile` mode and tune `bad_rate` based on your ground-truth expectation.
"""

import argparse
from dataclasses import dataclass
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import ARCHIVE_DIR, METADATA_DIR, apply_config_overrides, require_non_empty


DEFAULT_ARCHIVE = "callahan_01_archive"


@dataclass(frozen=True)
class TrackingLossConfig:
    archive: str = DEFAULT_ARCHIVE
    video: str | Path = ""
    output_dir: str | Path = ""
    scores_tsv: str | Path = ""
    existing_badframes: str | Path = ""
    start_frame: int = 0
    max_frame: int = -1
    frame_step: int = 1
    crop_top: int = 50
    crop_bottom: int = 50
    crop_left: int = 50
    crop_right: int = 50
    sobel_ksize: int = 3
    weight_edge: float = 0.45
    weight_row: float = 0.25
    weight_field: float = 0.30
    otsu_bins: int = 256
    threshold_mode: str = "value"
    bad_rate: float = -1.0
    threshold_value: float = 5.5
    calibrate_bad_rate_from_existing: bool = False
    export_bad_png_count: int = -1
    export_good_png_count: int = -1
    export_review_png_count: int = 0
    png_output_dir: str | Path = ""
    metadata_copy_dir: str | Path = ""


DEFAULT_CONFIG = TrackingLossConfig()


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


def _resolve_export_count(requested_count, fallback_count):
    requested = int(requested_count)
    if requested >= 0:
        return requested
    return max(0, int(fallback_count))


def export_frame_png_samples(video_path, frame_ids, sample_dir, label):
    sample_dir.mkdir(parents=True, exist_ok=True)
    if not frame_ids:
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for PNG export: {video_path}")

    written = []
    failed_frames = []
    try:
        for frame_idx in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                failed_frames.append(int(frame_idx))
                continue

            out_path = sample_dir / f"{label}_frame_{int(frame_idx):08d}.png"
            if not cv2.imwrite(str(out_path), frame_bgr):
                failed_frames.append(int(frame_idx))
                continue
            written.append(out_path)
    finally:
        cap.release()
    return written, failed_frames


def write_review_png_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("frame\tlabel\tscore\tpng_path\n")
        for row in rows:
            f.write(
                f"{int(row['frame'])}\t{row['label']}\t{float(row['score']):.8f}\t{row['png_path']}\n"
            )


def mirror_metadata_outputs(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".tsv", ".json"}:
            continue
        rel = path.relative_to(source_dir)
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied.append(dst)
    return copied


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


def _run_with_config(config: TrackingLossConfig):
    archive_name = require_non_empty(config.archive, "archive")

    sobel_ksize = sanitize_sobel_ksize(config.sobel_ksize)

    video_path = Path(config.video) if config.video else (ARCHIVE_DIR / f"{archive_name}.mkv")
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else (ARCHIVE_DIR / f"{archive_name}_tracking_badframe")
    )
    png_output_dir = (
        Path(config.png_output_dir) if config.png_output_dir else (output_dir / "review_png")
    )
    metadata_copy_dir = (
        Path(config.metadata_copy_dir)
        if config.metadata_copy_dir
        else (METADATA_DIR / archive_name / "tracking_badframe")
    )
    existing_badframes = (
        Path(config.existing_badframes)
        if config.existing_badframes
        else (METADATA_DIR / archive_name / "badframes.tsv")
    )
    scores_tsv_path = Path(config.scores_tsv) if config.scores_tsv else None

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
            start_frame=config.start_frame,
            max_frame=config.max_frame,
            frame_step=config.frame_step,
            crop_top=config.crop_top,
            crop_bottom=config.crop_bottom,
            crop_left=config.crop_left,
            crop_right=config.crop_right,
            sobel_ksize=sobel_ksize,
        )
        scores, signal_norm = combine_signals(
            edge_scores=edge_scores,
            row_scores=row_scores,
            field_scores=field_scores,
            weight_edge=config.weight_edge,
            weight_row=config.weight_row,
            weight_field=config.weight_field,
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

    threshold_mode = config.threshold_mode
    threshold_source = threshold_mode
    target_bad_rate = None
    if threshold_mode == "otsu":
        threshold = otsu_threshold(np.sort(scores_np), bins=config.otsu_bins)
    elif threshold_mode == "quantile":
        target_bad_rate = float(config.bad_rate)
        if config.calibrate_bad_rate_from_existing:
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
        threshold = float(config.threshold_value)
        if not np.isfinite(threshold):
            raise ValueError("--threshold-mode value requires a finite --threshold-value.")
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    labels = ["good" if score < threshold else "bad" for score in scores]
    score_by_frame = {int(frame_idx): float(score) for frame_idx, score in zip(indices, scores)}
    good_frames = sorted(frame for frame, label in zip(indices, labels) if label == "good")
    bad_frames = sorted(frame for frame, label in zip(indices, labels) if label == "bad")
    bad_ranges = ranges_from_sorted_frames(bad_frames)

    requested_bad_png_count = _resolve_export_count(
        config.export_bad_png_count,
        config.export_review_png_count,
    )
    requested_good_png_count = _resolve_export_count(
        config.export_good_png_count,
        config.export_review_png_count,
    )

    selected_bad_frames = []
    selected_good_frames = []
    written_bad_pngs = []
    written_good_pngs = []
    failed_bad_frames = []
    failed_good_frames = []
    review_manifest_path = None
    if requested_bad_png_count > 0 or requested_good_png_count > 0:
        if not video_path.exists():
            print(f"WARNING: skipping PNG export; video file not found: {video_path}")
        else:
            selected_bad_frames = pick_evenly_spaced_samples(bad_frames, requested_bad_png_count)
            selected_good_frames = pick_evenly_spaced_samples(good_frames, requested_good_png_count)
            try:
                written_bad_pngs, failed_bad_frames = export_frame_png_samples(
                    video_path=video_path,
                    frame_ids=selected_bad_frames,
                    sample_dir=png_output_dir / "bad",
                    label="bad",
                )
                written_good_pngs, failed_good_frames = export_frame_png_samples(
                    video_path=video_path,
                    frame_ids=selected_good_frames,
                    sample_dir=png_output_dir / "good",
                    label="good",
                )

                review_rows = []
                for png_path in written_bad_pngs:
                    frame_match = re.search(r"frame_(\d+)\.png$", png_path.name)
                    if not frame_match:
                        continue
                    frame_idx = int(frame_match.group(1))
                    review_rows.append(
                        {
                            "frame": frame_idx,
                            "label": "bad",
                            "score": score_by_frame.get(frame_idx, np.nan),
                            "png_path": png_path.relative_to(png_output_dir).as_posix(),
                        }
                    )
                for png_path in written_good_pngs:
                    frame_match = re.search(r"frame_(\d+)\.png$", png_path.name)
                    if not frame_match:
                        continue
                    frame_idx = int(frame_match.group(1))
                    review_rows.append(
                        {
                            "frame": frame_idx,
                            "label": "good",
                            "score": score_by_frame.get(frame_idx, np.nan),
                            "png_path": png_path.relative_to(png_output_dir).as_posix(),
                        }
                    )
                review_rows.sort(key=lambda row: (int(row["frame"]), str(row["label"])))
                review_manifest_path = png_output_dir / "review_manifest.tsv"
                write_review_png_manifest(review_manifest_path, review_rows)
            except Exception as exc:
                print(f"WARNING: PNG export failed: {exc}")

    summary = {
        "archive": archive_name,
        "video_path": str(video_path),
        "detector": "tracking_loss_scanline_field",
        "total_video_frames": (None if total_frames is None else int(total_frames)),
        "evaluated_frame_start": int(start_frame),
        "evaluated_frame_end": int(end_frame),
        "evaluated_frame_step": int(max(1, config.frame_step)),
        "evaluated_frames": int(len(indices)),
        "crop": {
            "top": int(max(0, config.crop_top)),
            "bottom": int(max(0, config.crop_bottom)),
            "left": int(max(0, config.crop_left)),
            "right": int(max(0, config.crop_right)),
        },
        "signals": {
            "sobel_ksize": int(sobel_ksize),
            "weights": {
                "edge": float(config.weight_edge),
                "row": float(config.weight_row),
                "field": float(config.weight_field),
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
        "good_frames": int(len(good_frames)),
        "bad_frames": int(len(bad_frames)),
        "predicted_bad_ranges": int(len(bad_ranges)),
        "png_samples": {
            "review_output_dir": (
                str(png_output_dir)
                if (requested_bad_png_count > 0 or requested_good_png_count > 0)
                else None
            ),
            "review_manifest": (
                str(review_manifest_path)
                if review_manifest_path is not None
                else None
            ),
            "bad": {
                "requested_samples": int(requested_bad_png_count),
                "available_predicted_frames": int(len(bad_frames)),
                "selected_frames": int(len(selected_bad_frames)),
                "written_pngs": int(len(written_bad_pngs)),
                "failed_frames": [int(x) for x in failed_bad_frames],
                "output_dir": (
                    str(png_output_dir / "bad")
                    if requested_bad_png_count > 0
                    else None
                ),
            },
            "good": {
                "requested_samples": int(requested_good_png_count),
                "available_predicted_frames": int(len(good_frames)),
                "selected_frames": int(len(selected_good_frames)),
                "written_pngs": int(len(written_good_pngs)),
                "failed_frames": [int(x) for x in failed_good_frames],
                "output_dir": (
                    str(png_output_dir / "good")
                    if requested_good_png_count > 0
                    else None
                ),
            },
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
    mirrored_files = mirror_metadata_outputs(output_dir, metadata_copy_dir)
    print(f"Mirrored metadata copy: {metadata_copy_dir} ({len(mirrored_files)} file(s))")
    if requested_bad_png_count > 0 or requested_good_png_count > 0:
        print(
            "PNG samples: "
            f"{png_output_dir} "
            f"(bad={len(written_bad_pngs)} written, good={len(written_good_pngs)} written)"
        )
        if review_manifest_path is not None:
            print(f"Review manifest: {review_manifest_path}")
    return {
        "frame_scores_path": frame_scores_path,
        "badframes_path": badframes_path,
        "summary_path": summary_path,
        "output_dir": output_dir,
        "png_output_dir": png_output_dir,
        "requested_bad_png_count": int(requested_bad_png_count),
        "requested_good_png_count": int(requested_good_png_count),
        "written_bad_pngs": int(len(written_bad_pngs)),
        "written_good_pngs": int(len(written_good_pngs)),
        "review_manifest_path": review_manifest_path,
        "metadata_copy_dir": metadata_copy_dir,
        "mirrored_file_count": int(len(mirrored_files)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Classify bad VHS frames with the tracking-loss detector and export "
            "review PNG sets for model training."
        )
    )
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--video", default="", help="Default: ../Archive/<archive>.mkv")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: ../Archive/<archive>_tracking_badframe",
    )
    parser.add_argument(
        "--scores-tsv",
        default="",
        help="Optional frame_scores TSV generated earlier; skips re-scoring video.",
    )
    parser.add_argument(
        "--existing-badframes",
        default="",
        help="Optional comparison TSV. Default: metadata/<archive>/badframes.tsv",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frame", type=int, default=-1)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-right", type=int, default=0)
    parser.add_argument("--sobel-ksize", type=int, default=3)
    parser.add_argument("--weight-edge", type=float, default=0.45)
    parser.add_argument("--weight-row", type=float, default=0.25)
    parser.add_argument("--weight-field", type=float, default=0.30)
    parser.add_argument("--otsu-bins", type=int, default=256)
    parser.add_argument(
        "--threshold-mode",
        choices=("otsu", "quantile", "value"),
        default="otsu",
    )
    parser.add_argument(
        "--bad-rate",
        type=float,
        default=-1.0,
        help="Only used by --threshold-mode quantile.",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=np.nan,
        help="Only used by --threshold-mode value.",
    )
    parser.add_argument(
        "--calibrate-bad-rate-from-existing",
        action="store_true",
        help="Quantile mode only.",
    )
    parser.add_argument(
        "--export-review-png-count",
        type=int,
        default=1000,
        help=(
            "Default sample count per label for review PNG export "
            "(applies to bad and good unless overridden)."
        ),
    )
    parser.add_argument(
        "--export-bad-png-count",
        type=int,
        default=-1,
        help="Override bad PNG sample count. Use -1 to inherit --export-review-png-count.",
    )
    parser.add_argument(
        "--export-good-png-count",
        type=int,
        default=-1,
        help="Override good PNG sample count. Use -1 to inherit --export-review-png-count.",
    )
    parser.add_argument(
        "--png-output-dir",
        default="",
        help="Default: <output-dir>/review_png (creates bad/ and good/ subfolders).",
    )
    parser.add_argument(
        "--metadata-copy-dir",
        default="",
        help="Default: metadata/<archive>/tracking_badframe (mirror copy of .tsv/.json files).",
    )
    return parser.parse_args(argv)


def args_to_config(args):
    return TrackingLossConfig(
        archive=str(args.archive),
        video=str(args.video or ""),
        output_dir=str(args.output_dir or ""),
        scores_tsv=str(args.scores_tsv or ""),
        existing_badframes=str(args.existing_badframes or ""),
        start_frame=int(args.start_frame),
        max_frame=int(args.max_frame),
        frame_step=int(args.frame_step),
        crop_top=int(args.crop_top),
        crop_bottom=int(args.crop_bottom),
        crop_left=int(args.crop_left),
        crop_right=int(args.crop_right),
        sobel_ksize=int(args.sobel_ksize),
        weight_edge=float(args.weight_edge),
        weight_row=float(args.weight_row),
        weight_field=float(args.weight_field),
        otsu_bins=int(args.otsu_bins),
        threshold_mode=str(args.threshold_mode),
        bad_rate=float(args.bad_rate),
        threshold_value=float(args.threshold_value),
        calibrate_bad_rate_from_existing=bool(args.calibrate_bad_rate_from_existing),
        export_bad_png_count=int(args.export_bad_png_count),
        export_good_png_count=int(args.export_good_png_count),
        export_review_png_count=int(args.export_review_png_count),
        png_output_dir=str(args.png_output_dir or ""),
        metadata_copy_dir=str(args.metadata_copy_dir or ""),
    )


def run_tracking_loss_classification(
    config: TrackingLossConfig | None = None,
    **overrides,
):
    resolved = config or DEFAULT_CONFIG
    if overrides:
        resolved = apply_config_overrides(resolved, **overrides)
    return _run_with_config(resolved)


def main(argv=None, config: TrackingLossConfig | None = None):
    if config is not None:
        run_tracking_loss_classification(config=config)
        return

    args = parse_args(argv)
    run_tracking_loss_classification(config=args_to_config(args))


if __name__ == "__main__":
    main()

