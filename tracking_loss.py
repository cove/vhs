#!/usr/bin/env python3.11
#
# Tracking-loss bad frame classification for VHS archives.
#
# Signals — designed for the two VHS tracking-loss artifacts:
#   1) chroma_loss  — mean HSV saturation drop.  Chroma loss makes frames go
#                     grey/noisy; the S channel collapses toward 0.
#                     Signal = 1 - mean(S)/255  (0 = good, 1 = fully grey)
#
#   2) noise_energy — std of per-row pixel variance, normalised by mean row
#                     variance.  Tracking noise creates a handful of rows with
#                     wildly high variance sitting next to normal rows; the
#                     std of variances spikes even when only a few rows are hit.
#
#   3) row_tear     — 95th-percentile of per-row mean-absolute-difference with
#                     the adjacent row.  Tearing shifts rows horizontally; the
#                     diff with their neighbour becomes enormous for torn rows.
#                     95th-pct rather than mean so a few torn rows dominate
#                     without normal scene content washing them out.
#
# Thresholding: Tukey-style fence (Q3 + iqr_mult*IQR) on chapter-aligned
# windows.  iqr_mult defaults to 3.5 but is now a proper config parameter so
# the interactive tuner can pass the value it found through directly.
# Larger overlapping chapters are excluded so umbrella chapters do not
# dominate labelling windows.
#

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
    video: str = ""
    chapters_file: str = ""        # default: METADATA_DIR/<archive>/chapters.ffmetadata
    output_dir: str = ""
    scores_tsv: str = ""
    existing_badframes: str = ""
    start_frame: int = 0
    max_frame: int = -1
    frame_step: int = 1
    crop_top: int = 50
    crop_bottom: int = 50
    crop_left: int = 50
    crop_right: int = 50
    weight_chroma: float = 0.25    # chroma_loss signal weight
    weight_noise: float = 0.25     # noise_energy signal weight
    weight_tear: float = 0.25      # row_tear signal weight
    weight_wave: float = 0.25      # wave_energy signal weight
    iqr_mult: float = 3.5          # Tukey fence multiplier: threshold = Q3 + iqr_mult * IQR
    threshold_window_size: int = 1000
    export_bad_png_count: int = -1
    export_good_png_count: int = -1
    export_review_png_count: int = 0
    png_output_dir: str = ""
    metadata_copy_dir: str = ""
    metadata_badframes_tsv: str = ""


DEFAULT_CONFIG = TrackingLossConfig()


# ---------------------------------------------------------------------------
# Artifact-specific signals
# ---------------------------------------------------------------------------

def compute_tracking_signals(frame_bgr, crop_top, crop_bottom, crop_left, crop_right):
    """
    Compute three artifact-specific signals from a raw BGR frame.
    Works on the full colour frame — grayscale conversion only happens
    where needed, after chroma is measured.

    Returns
    -------
    chroma_loss  : float  0 = fully saturated (good), 1 = fully grey (bad)
    noise_energy : float  normalised std of per-row variance
    row_tear     : float  95th-percentile row-to-neighbour absolute difference
    """
    h, w = frame_bgr.shape[:2]

    # Crop — clamp so we always have at least 1 row/col
    y0 = min(int(crop_top),    max(0, h - 1))
    y1 = max(y0 + 1,           h - int(crop_bottom))
    x0 = min(int(crop_left),   max(0, w - 1))
    x1 = max(x0 + 1,           w - int(crop_right))

    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = frame_bgr

    # 1. Chroma loss — measure before converting to grey
    hsv         = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s           = hsv[:, :, 1].astype(np.float32)   # saturation 0–255
    chroma_loss = 1.0 - float(np.mean(s) / 255.0)   # invert: high = bad

    # 2. Noise energy — per-row variance spread
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_vars = np.var(gray, axis=1)
    mean_var = float(np.mean(row_vars))
    noise_energy = float(np.std(row_vars) / mean_var) if mean_var > 1e-6 else 0.0

    # 3. Row tear — 95th-percentile of row-to-neighbour difference
    if gray.shape[0] > 1:
        row_diffs = np.abs(gray[1:] - gray[:-1]).mean(axis=1)
        row_tear  = float(np.percentile(row_diffs, 95))
    else:
        row_tear = 0.0

    # 4. Wave energy — std of per-row horizontal centre-of-mass.
    #    VHS horizontal sync instability displaces rows left/right in a wave
    #    pattern.  The CoM of each row's luminance oscillates; stable frames
    #    have near-constant CoM across rows.  High-pass the CoM series first
    #    (subtract a 5-row rolling mean) so slow scene content gradients don't
    #    inflate the score — only the rapid row-to-row oscillation counts.
    gray_f   = gray.astype(np.float32)
    row_sums = gray_f.sum(axis=1)
    cols_idx = np.arange(gray_f.shape[1], dtype=np.float32)
    row_com  = (gray_f @ cols_idx) / np.maximum(row_sums, 1e-6)
    if row_com.shape[0] >= 5:
        kernel   = np.ones(5, dtype=np.float32) / 5
        trend    = np.convolve(row_com, kernel, mode="same")
        wave_energy = float(np.std(row_com - trend))
    else:
        wave_energy = float(np.std(row_com))

    return chroma_loss, noise_energy, row_tear, wave_energy


# ---------------------------------------------------------------------------
# Chapter parsing
# ---------------------------------------------------------------------------

def parse_chapters_ffmetadata(path):
    chapters, current = [], {}
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line == "[CHAPTER]":
            if current.get("start") is not None and current.get("end") is not None:
                chapters.append(current)
            current = {}
            continue
        if "=" not in line or line.startswith(";"):
            continue
        key, _, value = line.partition("=")
        key   = key.strip().upper()
        value = value.strip()
        if key == "START":
            try:
                current["start"] = int(value)
            except ValueError:
                pass
        elif key == "END":
            try:
                current["end"] = int(value)
            except ValueError:
                pass
        elif key == "TITLE":
            current["title"] = value
    if current.get("start") is not None and current.get("end") is not None:
        chapters.append(current)
    return chapters


def resolve_overlapping_chapters(chapters):
    """Drop the larger chapter when two chapters overlap."""
    kept  = list(chapters or [])
    if len(kept) <= 1:
        return kept, {"original_count": len(kept), "excluded_count": 0, "kept_count": len(kept)}
    spans   = [max(1, int(ch["end"]) - int(ch["start"]) + 1) for ch in kept]
    exclude = set()
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a0, a1 = int(kept[i]["start"]), int(kept[i]["end"])
            b0, b1 = int(kept[j]["start"]), int(kept[j]["end"])
            if a1 < b0 or b1 < a0:
                continue
            if spans[i] > spans[j]:
                exclude.add(i)
            elif spans[j] > spans[i]:
                exclude.add(j)
    filtered = [ch for idx, ch in enumerate(kept) if idx not in exclude]
    return filtered, {"original_count": len(kept),
                      "excluded_count": len(exclude),
                      "kept_count": len(filtered)}


def assign_frames_to_chapters(indices, chapters):
    spans = [max(1, int(ch["end"]) - int(ch["start"]) + 1) for ch in chapters]
    result = []
    for fi in indices:
        cands = [ci for ci, ch in enumerate(chapters)
                 if int(ch["start"]) <= int(fi) <= int(ch["end"])]
        if not cands:
            result.append(-1)
        else:
            result.append(min(cands, key=lambda ci: (spans[ci], int(chapters[ci]["start"]))))
    return result


# ---------------------------------------------------------------------------
# Windowed IQR thresholding
# ---------------------------------------------------------------------------

def iqr_threshold_for_window(window_scores, iqr_mult=3.5):
    """Tukey fence: Q3 + iqr_mult x IQR."""
    w = np.asarray(window_scores, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.inf
    q1 = float(np.percentile(w, 25))
    q3 = float(np.percentile(w, 75))
    return q3 + float(iqr_mult) * (q3 - q1)


def compute_per_frame_thresholds(scores, indices, chapters, window_size, iqr_mult=3.5):
    scores_np = np.asarray(scores, dtype=np.float64)
    n         = len(scores_np)
    thresholds = np.full(n, np.nan)
    win_size   = max(1, int(window_size))

    chapter_ids          = assign_frames_to_chapters(indices, chapters)
    positions_by_chapter = {}
    for pos, cid in enumerate(chapter_ids):
        positions_by_chapter.setdefault(cid, []).append(pos)

    window_info = []

    # Chapter-aligned windows
    for cid, ch_positions in positions_by_chapter.items():
        if cid < 0 or cid >= len(chapters):
            continue
        ch      = chapters[cid]
        ch_s    = int(ch["start"])
        ch_e    = int(ch["end"])
        title   = ch.get("title", f"chapter_{cid}")
        for win_s in range(ch_s, ch_e + 1, win_size):
            win_e     = min(ch_e, win_s + win_size - 1)
            positions = [p for p in ch_positions if win_s <= int(indices[p]) <= win_e]
            if not positions:
                continue
            wscores = scores_np[positions]
            thresh  = iqr_threshold_for_window(wscores, iqr_mult=iqr_mult)
            for p in positions:
                thresholds[p] = thresh
            finite = wscores[np.isfinite(wscores)]
            q1 = float(np.percentile(finite, 25)) if finite.size else float("nan")
            q3 = float(np.percentile(finite, 75)) if finite.size else float("nan")
            window_info.append({
                "chapter_id": int(cid), "title": title, "window_type": "chapter",
                "window_start_frame": int(win_s), "window_end_frame": int(win_e),
                "frame_count_in_window": len(positions),
                "q1": q1, "q3": q3, "iqr": q3 - q1,
                "iqr_mult": float(iqr_mult),
                "threshold": float(thresh),
            })

    # Fallback for frames outside chapters
    fallback_positions = positions_by_chapter.get(-1, [])
    fallback_groups    = {}
    for pos in fallback_positions:
        fi    = int(indices[pos])
        key   = (fi // win_size) * win_size
        fallback_groups.setdefault(key, []).append(pos)
    for win_s, positions in sorted(fallback_groups.items()):
        wscores = scores_np[positions]
        thresh  = iqr_threshold_for_window(wscores, iqr_mult=iqr_mult)
        for p in positions:
            thresholds[p] = thresh
        finite = wscores[np.isfinite(wscores)]
        q1 = float(np.percentile(finite, 25)) if finite.size else float("nan")
        q3 = float(np.percentile(finite, 75)) if finite.size else float("nan")
        window_info.append({
            "chapter_id": -1, "title": "no_chapter_fallback", "window_type": "fallback",
            "window_start_frame": int(win_s), "window_end_frame": int(win_s + win_size - 1),
            "frame_count_in_window": len(positions),
            "q1": q1, "q3": q3, "iqr": q3 - q1,
            "iqr_mult": float(iqr_mult),
            "threshold": float(thresh),
        })

    global_fallback = iqr_threshold_for_window(scores_np, iqr_mult=iqr_mult)
    thresholds = np.where(np.isfinite(thresholds), thresholds, global_fallback)
    return thresholds, chapter_ids, window_info


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_video_frames(video_path, start_frame, max_frame, frame_step,
                       crop_top, crop_bottom, crop_left, crop_right):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read frame count: {video_path}")

    start = max(0, int(start_frame))
    end   = total_frames - 1 if int(max_frame) < 0 else min(total_frames - 1, int(max_frame))
    step  = max(1, int(frame_step))
    if start > end:
        cap.release()
        raise ValueError(f"start-frame ({start}) is after max-frame ({end}).")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    indices, chroma_scores, noise_scores, tear_scores, wave_scores = [], [], [], [], []
    pbar = tqdm(total=((end - start) // step) + 1, desc="Scoring frames", unit="frame")
    frame_idx = start
    while frame_idx <= end:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if ((frame_idx - start) % step) == 0:
            ch, no, te, wa = compute_tracking_signals(
                frame_bgr, crop_top, crop_bottom, crop_left, crop_right
            )
            indices.append(int(frame_idx))
            chroma_scores.append(ch)
            noise_scores.append(no)
            tear_scores.append(te)
            wave_scores.append(wa)
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()
    if not indices:
        raise RuntimeError("No frame scores produced.")
    return total_frames, start, end, indices, chroma_scores, noise_scores, tear_scores, wave_scores


def robust_zscore(values):
    vals   = np.asarray(values, dtype=np.float64)
    center = float(np.median(vals))
    mad    = float(np.median(np.abs(vals - center)))
    scale  = 1.4826 * mad
    if scale <= 1e-12:
        std   = float(np.std(vals))
        scale = std if std > 1e-12 else 1.0
    return (vals - center) / scale, center, scale


def combine_signals(chroma_scores, noise_scores, tear_scores, wave_scores,
                    weight_chroma, weight_noise, weight_tear, weight_wave):
    w_sum = float(weight_chroma) + float(weight_noise) + float(weight_tear) + float(weight_wave)
    if w_sum <= 0:
        raise ValueError("At least one signal weight must be > 0.")
    chroma_z, cc, cs = robust_zscore(chroma_scores)
    noise_z,  nc, ns = robust_zscore(noise_scores)
    tear_z,   tc, ts = robust_zscore(tear_scores)
    wave_z,   wc, ws = robust_zscore(wave_scores)
    score = (float(weight_chroma) * chroma_z +
             float(weight_noise)  * noise_z  +
             float(weight_tear)   * tear_z   +
             float(weight_wave)   * wave_z) / w_sum
    norm  = {
        "chroma": {"center": float(cc), "scale": float(cs)},
        "noise":  {"center": float(nc), "scale": float(ns)},
        "tear":   {"center": float(tc), "scale": float(ts)},
        "wave":   {"center": float(wc), "scale": float(ws)},
    }
    return score.astype(np.float64), norm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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
            s, e = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if e < s:
            s, e = e, s
        ranges.append((max(0, s), max(0, e)))
    return ranges


def expand_ranges_to_set(ranges, min_frame, max_frame):
    out = set()
    if max_frame < min_frame:
        return out
    for s, e in ranges:
        for fi in range(max(min_frame, s), min(max_frame, e) + 1):
            out.add(fi)
    return out


def ranges_from_sorted_frames(frame_ids):
    if not frame_ids:
        return []
    result = []
    start = prev = int(frame_ids[0])
    for v in frame_ids[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        result.append((start, prev))
        start = prev = v
    result.append((start, prev))
    return result


def _try_parse_float(text):
    try:
        return float(text)
    except Exception:
        return np.nan


def load_scores_tsv(path):
    """Load a previously saved frame_scores.tsv (supports new and old column names)."""
    rows, header_map = [], {}
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line  = raw_line.strip()
        if not line:
            continue
        parts   = line.split("\t")
        lowered = [p.strip().lower() for p in parts]
        if lowered and lowered[0] == "frame":
            header_map = {name: idx for idx, name in enumerate(lowered)}
            continue
        fi_col = header_map.get("frame", 0)
        sc_col = header_map.get("score", 1)
        c_col  = header_map.get("chroma_loss",   header_map.get("edge_energy",    -1))
        n_col  = header_map.get("noise_energy",  header_map.get("row_instability", -1))
        t_col  = header_map.get("row_tear",      header_map.get("field_mismatch",  -1))
        w_col  = header_map.get("wave_energy", -1)
        if len(parts) <= max(fi_col, sc_col):
            continue
        try:
            fi = int(parts[fi_col])
            sc = float(parts[sc_col])
        except ValueError:
            continue
        c = _try_parse_float(parts[c_col]) if 0 <= c_col < len(parts) else np.nan
        n = _try_parse_float(parts[n_col]) if 0 <= n_col < len(parts) else np.nan
        t = _try_parse_float(parts[t_col]) if 0 <= t_col < len(parts) else np.nan
        w = _try_parse_float(parts[w_col]) if 0 <= w_col < len(parts) else 0.0
        rows.append((fi, sc, c, n, t, w))
    if not rows:
        raise ValueError(f"No frame/score rows found in {path}")
    rows.sort(key=lambda x: x[0])
    return ([r[0] for r in rows], [r[1] for r in rows],
            [r[2] for r in rows], [r[3] for r in rows],
            [r[4] for r in rows], [r[5] for r in rows])


def pick_evenly_spaced_samples(frame_ids, count):
    ordered = [int(x) for x in frame_ids]
    count   = max(0, int(count))
    if count <= 0 or not ordered:
        return []
    if count >= len(ordered):
        return ordered
    seen, chosen = set(), []
    for idx in np.linspace(0, len(ordered) - 1, num=count, dtype=int):
        fi = ordered[int(idx)]
        if fi not in seen:
            seen.add(fi)
            chosen.append(fi)
    return chosen


def _resolve_export_count(requested, fallback):
    return max(0, int(fallback)) if int(requested) < 0 else max(0, int(requested))


def export_frame_png_samples(video_path, frame_ids, sample_dir, label):
    sample_dir.mkdir(parents=True, exist_ok=True)
    if not frame_ids:
        return [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for PNG export: {video_path}")
    written, failed = [], []
    try:
        for fi in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                failed.append(int(fi))
                continue
            out = sample_dir / f"{label}_frame_{int(fi):08d}.png"
            if cv2.imwrite(str(out), frame_bgr):
                written.append(out)
            else:
                failed.append(int(fi))
    finally:
        cap.release()
    return written, failed


def write_review_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("frame\tlabel\tscore\tthreshold\tpng_path\n")
        for row in rows:
            f.write(f"{int(row['frame'])}\t{row['label']}\t"
                    f"{float(row['score']):.8f}\t{float(row['threshold']):.8f}\t"
                    f"{row['png_path']}\n")


def mirror_metadata_outputs(source_dir, target_dir):
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".tsv", ".json"}:
            dst = target_dir / path.relative_to(source_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied.append(dst)
    return copied


def finite_stats(values):
    v = np.asarray(values, dtype=np.float64)
    f = v[np.isfinite(v)]
    if f.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(np.min(f)), "max": float(np.max(f)), "mean": float(np.mean(f))}


def write_outputs(output_dir, indices, scores, chroma_scores, noise_scores, tear_scores,
                  wave_scores, thresholds, labels, bad_ranges, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_scores_path = output_dir / "frame_scores.tsv"
    with frame_scores_path.open("w", encoding="utf-8") as f:
        f.write("frame\tscore\tthreshold\tchroma_loss\tnoise_energy\trow_tear\twave_energy\tlabel\n")
        for fi, sc, thr, ch, no, te, wa, lbl in zip(
                indices, scores, thresholds, chroma_scores, noise_scores,
                tear_scores, wave_scores, labels):
            f.write(f"{fi}\t{sc:.8f}\t{thr:.8f}\t{ch:.8f}\t{no:.8f}\t{te:.8f}\t{wa:.8f}\t{lbl}\n")
    badframes_path = output_dir / "badframes.tsv"
    with badframes_path.open("w", encoding="utf-8") as f:
        f.write("start_frame\tend_frame\tnote\n")
        for s, e in bad_ranges:
            f.write(f"{s}\t{e}\ttracking_loss_chapter_iqr\n")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return frame_scores_path, badframes_path, summary_path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _run_with_config(config: TrackingLossConfig):
    archive_name = require_non_empty(config.archive, "archive")

    video_path    = Path(config.video)    if config.video    else (ARCHIVE_DIR / f"{archive_name}.mkv")
    output_dir    = Path(config.output_dir) if config.output_dir else (METADATA_DIR / archive_name / "tracking_badframe")
    png_dir       = Path(config.png_output_dir)       if config.png_output_dir       else (output_dir / "review_png")
    meta_copy     = Path(config.metadata_copy_dir)    if config.metadata_copy_dir    else (METADATA_DIR / archive_name / "tracking_badframe")
    meta_bf_tsv   = Path(config.metadata_badframes_tsv) if config.metadata_badframes_tsv else (METADATA_DIR / archive_name / "badframes.tsv")
    existing_bf   = Path(config.existing_badframes)   if config.existing_badframes   else (METADATA_DIR / archive_name / "badframes.tsv")
    chapters_file = Path(config.chapters_file)        if config.chapters_file        else (METADATA_DIR / archive_name / "chapters.ffmetadata")
    scores_tsv    = Path(config.scores_tsv)           if config.scores_tsv           else None

    iqr_mult = float(config.iqr_mult)

    # Load or compute raw scores
    signal_norm  = None
    total_frames = None

    if scores_tsv:
        if not scores_tsv.exists():
            raise FileNotFoundError(f"Scores TSV not found: {scores_tsv}")
        indices, scores, chroma_scores, noise_scores, tear_scores, wave_scores = load_scores_tsv(scores_tsv)
        start_frame, end_frame = min(indices), max(indices)
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        print(f"Loaded {len(indices)} frame scores from: {scores_tsv}")
    else:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        (total_frames, start_frame, end_frame,
         indices, chroma_scores, noise_scores, tear_scores, wave_scores) = score_video_frames(
            video_path=video_path,
            start_frame=config.start_frame,
            max_frame=config.max_frame,
            frame_step=config.frame_step,
            crop_top=config.crop_top,
            crop_bottom=config.crop_bottom,
            crop_left=config.crop_left,
            crop_right=config.crop_right,
        )
        scores_np, signal_norm = combine_signals(
            chroma_scores, noise_scores, tear_scores, wave_scores,
            config.weight_chroma, config.weight_noise, config.weight_tear, config.weight_wave,
        )
        scores = scores_np.astype(np.float64).tolist()

    scores_np     = np.asarray(scores, dtype=np.float64)
    evaluated_set = set(indices)

    # Load chapters
    chapters        = []
    chapter_overlap = {"original_count": 0, "excluded_count": 0, "kept_count": 0}
    chapters_source = None
    if chapters_file.exists():
        raw_chapters = parse_chapters_ffmetadata(chapters_file)
        chapters, chapter_overlap = resolve_overlapping_chapters(raw_chapters)
        chapters_source = str(chapters_file)
        print(f"Loaded {len(raw_chapters)} chapters from: {chapters_file} "
              f"(kept {len(chapters)}, excluded overlaps: {chapter_overlap['excluded_count']})")
        for ch in chapters:
            print(f"  [{ch['start']:>7} - {ch['end']:>7}]  {ch.get('title', '?')}")
    else:
        print(f"WARNING: chapters file not found ({chapters_file}); "
              "using single window over all frames.")

    # Per-chapter IQR thresholds — now uses config.iqr_mult
    thresholds, chapter_ids, window_info = compute_per_frame_thresholds(
        scores_np, indices, chapters,
        window_size=config.threshold_window_size,
        iqr_mult=iqr_mult,
    )

    print(f"\nThreshold windows (Q3 + {iqr_mult}xIQR, window={config.threshold_window_size} frames):")
    for wi in window_info:
        print(f"  {wi['title'][:55]:<55}  "
              f"frames={wi['frame_count_in_window']:>6}  "
              f"threshold={wi['threshold']:>8.4f}")

    # Classify
    labels             = ["bad" if s >= t else "good" for s, t in zip(scores_np, thresholds)]
    score_by_frame     = {int(fi): float(sc) for fi, sc in zip(indices, scores_np)}
    threshold_by_frame = {int(fi): float(t)  for fi, t  in zip(indices, thresholds)}

    good_frames = sorted(fi for fi, lbl in zip(indices, labels) if lbl == "good")
    bad_frames  = sorted(fi for fi, lbl in zip(indices, labels) if lbl == "bad")
    bad_ranges  = ranges_from_sorted_frames(bad_frames)

    # Ground-truth comparison
    manual_ranges, manual_bad_eval = [], set()
    if existing_bf.exists():
        manual_ranges   = parse_badframe_ranges(existing_bf)
        manual_bad      = expand_ranges_to_set(manual_ranges, start_frame, end_frame)
        manual_bad_eval = evaluated_set.intersection(manual_bad)

    # PNG export
    req_bad  = _resolve_export_count(config.export_bad_png_count,  config.export_review_png_count)
    req_good = _resolve_export_count(config.export_good_png_count, config.export_review_png_count)
    w_bad, w_good, f_bad, f_good = [], [], [], []
    manifest_path = None

    if (req_bad > 0 or req_good > 0) and video_path.exists():
        sel_bad  = pick_evenly_spaced_samples(bad_frames,  req_bad)
        sel_good = pick_evenly_spaced_samples(good_frames, req_good)
        try:
            w_bad,  f_bad  = export_frame_png_samples(video_path, sel_bad,  png_dir / "bad",  "bad")
            w_good, f_good = export_frame_png_samples(video_path, sel_good, png_dir / "good", "good")
            review_rows = []
            for png_path in w_bad + w_good:
                m = re.search(r"frame_(\d+)\.png$", png_path.name)
                if not m:
                    continue
                fi  = int(m.group(1))
                lbl = "bad" if "bad" in png_path.parts else "good"
                review_rows.append({
                    "frame":     fi,
                    "label":     lbl,
                    "score":     score_by_frame.get(fi, np.nan),
                    "threshold": threshold_by_frame.get(fi, np.nan),
                    "png_path":  png_path.relative_to(png_dir).as_posix(),
                })
            review_rows.sort(key=lambda r: int(r["frame"]))
            manifest_path = png_dir / "review_manifest.tsv"
            write_review_manifest(manifest_path, review_rows)
        except Exception as exc:
            print(f"WARNING: PNG export failed: {exc}")

    # Summary
    summary = {
        "archive": archive_name,
        "video_path": str(video_path),
        "detector": "tracking_loss_chapter_iqr",
        "total_video_frames": (None if total_frames is None else int(total_frames)),
        "evaluated_frame_start": int(start_frame),
        "evaluated_frame_end":   int(end_frame),
        "evaluated_frame_step":  int(max(1, config.frame_step)),
        "evaluated_frames":      int(len(indices)),
        "crop": {k: int(getattr(config, f"crop_{k}")) for k in ("top","bottom","left","right")},
        "signals": {
            "description": {
                "chroma_loss":  "1 - mean(HSV saturation)/255; high = desaturated/grey",
                "noise_energy": "std(row_variance)/mean(row_variance); high = noisy rows present",
                "row_tear":     "95th-pct row-to-neighbour abs diff; high = rows horizontally torn",
                "wave_energy":  "std of high-passed per-row horizontal CoM; high = wavy/wobbly rows",
            },
            "weights": {
                "chroma": float(config.weight_chroma),
                "noise":  float(config.weight_noise),
                "tear":   float(config.weight_tear),
                "wave":   float(config.weight_wave),
            },
            "normalization": signal_norm,
            "chroma_loss":  finite_stats(chroma_scores),
            "noise_energy": finite_stats(noise_scores),
            "row_tear":     finite_stats(tear_scores),
            "wave_energy":  finite_stats(wave_scores),
        },
        "thresholding": {
            "method":      "chapter_aligned_window_iqr",
            "formula":     f"Q3 + {iqr_mult} x IQR (per chapter-aligned window)",
            "iqr_mult":    iqr_mult,
            "threshold_window_size": int(config.threshold_window_size),
            "chapters_file":         chapters_source,
            "chapter_count":         len(chapters),
            "chapter_overlap_resolution": chapter_overlap,
            "windows": window_info,
        },
        "score_min":  float(np.min(scores_np)),
        "score_max":  float(np.max(scores_np)),
        "score_mean": float(np.mean(scores_np)),
        "good_frames": int(len(good_frames)),
        "bad_frames":  int(len(bad_frames)),
        "predicted_bad_ranges": int(len(bad_ranges)),
        "png_samples": {
            "review_output_dir": str(png_dir) if (req_bad > 0 or req_good > 0) else None,
            "review_manifest":   str(manifest_path) if manifest_path else None,
            "bad":  {"requested": req_bad,  "written": len(w_bad),  "failed": [int(x) for x in f_bad]},
            "good": {"requested": req_good, "written": len(w_good), "failed": [int(x) for x in f_good]},
        },
    }

    if existing_bf.exists():
        predicted_bad = set(bad_frames)
        tp   = len(predicted_bad & manual_bad_eval)
        fp   = len(predicted_bad - manual_bad_eval)
        fn   = len(manual_bad_eval - predicted_bad)
        tn   = len(evaluated_set) - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        summary["comparison_to_existing_badframes"] = {
            "path": str(existing_bf),
            "manual_bad_ranges": int(len(manual_ranges)),
            "manual_bad_frames_in_window": int(len(manual_bad_eval)),
            "predicted_bad_frames": int(len(predicted_bad)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
        }
    else:
        summary["comparison_to_existing_badframes"] = {
            "path": str(existing_bf), "note": "file not found; comparison skipped",
        }

    frame_scores_path, badframes_path, summary_path = write_outputs(
        output_dir=output_dir,
        indices=indices,
        scores=scores_np.tolist(),
        chroma_scores=chroma_scores,
        noise_scores=noise_scores,
        tear_scores=tear_scores,
        wave_scores=wave_scores,
        thresholds=thresholds.tolist(),
        labels=labels,
        bad_ranges=bad_ranges,
        summary=summary,
    )

    print(f"\nFrame scores:           {frame_scores_path}")
    print(f"Predicted bad ranges:   {badframes_path}")
    print(f"Summary:                {summary_path}")
    mirrored = mirror_metadata_outputs(output_dir, meta_copy)
    print(f"Mirrored metadata:      {meta_copy} ({len(mirrored)} file(s))")
    meta_bf_tsv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(badframes_path, meta_bf_tsv)
    print(f"Metadata badframes TSV: {meta_bf_tsv}")
    if req_bad > 0 or req_good > 0:
        print(f"PNG samples:            {png_dir}  (bad={len(w_bad)}, good={len(w_good)})")
        if manifest_path:
            print(f"Review manifest:        {manifest_path}")

    return {
        "frame_scores_path":      frame_scores_path,
        "badframes_path":         badframes_path,
        "summary_path":           summary_path,
        "output_dir":             output_dir,
        "png_output_dir":         png_dir,
        "written_bad_pngs":       int(len(w_bad)),
        "written_good_pngs":      int(len(w_good)),
        "review_manifest_path":   manifest_path,
        "metadata_copy_dir":      meta_copy,
        "metadata_badframes_tsv": meta_bf_tsv,
        "mirrored_file_count":    int(len(mirrored)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Classify bad VHS frames using per-chapter IQR thresholding."
    )
    p.add_argument("--archive",       default=DEFAULT_ARCHIVE)
    p.add_argument("--video",         default="")
    p.add_argument("--chapters-file", default="")
    p.add_argument("--output-dir",    default="")
    p.add_argument("--scores-tsv",    default="")
    p.add_argument("--existing-badframes", default="")
    p.add_argument("--start-frame",  type=int,   default=0)
    p.add_argument("--max-frame",    type=int,   default=-1)
    p.add_argument("--frame-step",   type=int,   default=1)
    p.add_argument("--crop-top",     type=int,   default=50)
    p.add_argument("--crop-bottom",  type=int,   default=50)
    p.add_argument("--crop-left",    type=int,   default=50)
    p.add_argument("--crop-right",   type=int,   default=50)
    p.add_argument("--weight-chroma", type=float, default=0.25)
    p.add_argument("--weight-noise",  type=float, default=0.25)
    p.add_argument("--weight-tear",   type=float, default=0.25)
    p.add_argument("--weight-wave",   type=float, default=0.25,
                   help="Weight for wave-energy signal (horizontal sync instability, default: 0.25)")
    p.add_argument("--iqr-mult",      type=float, default=3.5,
                   help="Tukey fence multiplier: threshold = Q3 + k*IQR (default: 3.5)")
    p.add_argument("--threshold-window-size", type=int, default=1000)
    p.add_argument("--export-review-png-count", type=int, default=1000)
    p.add_argument("--export-bad-png-count",    type=int, default=-1)
    p.add_argument("--export-good-png-count",   type=int, default=-1)
    p.add_argument("--png-output-dir",          default="")
    p.add_argument("--metadata-copy-dir",       default="")
    p.add_argument("--metadata-badframes-tsv",  default="")
    return p.parse_args(argv)


def args_to_config(args):
    return TrackingLossConfig(
        archive=args.archive,
        video=args.video or "",
        chapters_file=args.chapters_file or "",
        output_dir=args.output_dir or "",
        scores_tsv=args.scores_tsv or "",
        existing_badframes=args.existing_badframes or "",
        start_frame=int(args.start_frame),
        max_frame=int(args.max_frame),
        frame_step=int(args.frame_step),
        crop_top=int(args.crop_top),
        crop_bottom=int(args.crop_bottom),
        crop_left=int(args.crop_left),
        crop_right=int(args.crop_right),
        weight_chroma=float(args.weight_chroma),
        weight_noise=float(args.weight_noise),
        weight_tear=float(args.weight_tear),
        weight_wave=float(args.weight_wave),
        iqr_mult=float(args.iqr_mult),
        threshold_window_size=max(1, int(args.threshold_window_size)),
        export_bad_png_count=int(args.export_bad_png_count),
        export_good_png_count=int(args.export_good_png_count),
        export_review_png_count=int(args.export_review_png_count),
        png_output_dir=args.png_output_dir or "",
        metadata_copy_dir=args.metadata_copy_dir or "",
        metadata_badframes_tsv=args.metadata_badframes_tsv or "",
    )


def run_tracking_loss_classification(config: TrackingLossConfig | None = None, **overrides):
    resolved = config or DEFAULT_CONFIG
    if overrides:
        resolved = apply_config_overrides(resolved, **overrides)
    return _run_with_config(resolved)


def main(argv=None, config: TrackingLossConfig | None = None):
    if config is not None:
        run_tracking_loss_classification(config=config)
        return
    run_tracking_loss_classification(config=args_to_config(parse_args(argv)))


if __name__ == "__main__":
    main()
