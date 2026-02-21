#!/usr/bin/env python3.11
"""
sweep_tracking_loss.py

Sweeps threshold modes, bad_rates, and signal weights against your existing
badframes.tsv ground truth, ranks by F1, and renders a side-by-side
comparison video so you can visually inspect where configs agree/disagree.

Usage:
    # Score the video once, sweep all configs, write comparison video:
    python sweep_tracking_loss.py --archive callahan_01_archive

    # Re-sweep using pre-computed scores (much faster — skips video decoding):
    python sweep_tracking_loss.py --scores-tsv path/to/frame_scores.tsv

    # Limit comparison video to top-N configs:
    python sweep_tracking_loss.py --scores-tsv scores.tsv --video-configs 6
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── direct imports from tracking_loss ────────────────────────────────────────
from step_16_classify_badframes_tracking_loss import (
    combine_signals,
    expand_ranges_to_set,
    load_scores_tsv,
    otsu_threshold,
    parse_badframe_ranges,
    ranges_from_sorted_frames,
    score_video_frames,
    sanitize_sobel_ksize,
)
from common import ARCHIVE_DIR, METADATA_DIR
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Sweep space — edit to taste
# ---------------------------------------------------------------------------

THRESHOLD_CONFIGS = [
    # (threshold_mode, bad_rate_or_None)
    ("otsu",     None),
    ("quantile", 0.01),
    ("quantile", 0.03),
    ("quantile", 0.05),
    ("quantile", 0.10),
    ("quantile", 0.20),
]

WEIGHT_COMBOS = [
    # (label,          w_edge, w_row, w_field)
    ("default",        0.45,   0.25,  0.30),
    ("edge_only",      1.00,   0.00,  0.00),
    ("row_only",       0.00,   1.00,  0.00),
    ("field_only",     0.00,   0.00,  1.00),
    ("balanced",       0.33,   0.33,  0.34),
    ("edge_heavy",     0.60,   0.20,  0.20),
    ("row_heavy",      0.20,   0.60,  0.20),
    ("field_heavy",    0.20,   0.20,  0.60),
    ("edge_row",       0.50,   0.50,  0.00),
    ("edge_field",     0.50,   0.00,  0.50),
    ("row_field",      0.00,   0.50,  0.50),
    ("edge_dominant",  0.70,   0.15,  0.15),
]

SOBEL_KSIZES = [3]  # add more e.g. [1, 3, 5] if re-scoring from video

# Comparison video defaults
TILE_WIDTH  = 320   # px per config tile
TILE_HEIGHT = 240   # px per config tile
PANEL_H     = 60    # px for the verdict bar under each tile
VIDEO_FPS   = 10.0  # fps of the output comparison video
SAMPLE_STEP = 30    # write every Nth frame to the comparison video (1 = all)

# ---------------------------------------------------------------------------


def short_label(weight_label, t_mode, bad_rate):
    br = f"_br{bad_rate:.2f}" if bad_rate is not None else ""
    return f"{weight_label}|{t_mode}{br}"


def build_configs(edge_scores_raw, row_scores_raw, field_scores_raw, indices):
    """
    For every (weight_combo × threshold_config) pair, compute the composite
    score array in-memory and return config descriptors.  No file I/O happens.
    """
    configs = []
    for _ksize in SOBEL_KSIZES:
        for w_label, w_edge, w_row, w_field in WEIGHT_COMBOS:
            scores_np, _ = combine_signals(
                edge_scores_raw, row_scores_raw, field_scores_raw,
                w_edge, w_row, w_field,
            )
            scores_np = scores_np.astype(np.float64)

            for t_mode, bad_rate in THRESHOLD_CONFIGS:
                if t_mode == "otsu":
                    threshold = otsu_threshold(np.sort(scores_np))
                elif t_mode == "quantile":
                    if bad_rate is None or not (0.0 < bad_rate < 1.0):
                        continue
                    threshold = float(np.quantile(scores_np, 1.0 - bad_rate))
                else:
                    continue

                labels = np.array(
                    ["bad" if s >= threshold else "good" for s in scores_np]
                )
                configs.append({
                    "label":     short_label(w_label, t_mode, bad_rate),
                    "w_label":   w_label,
                    "w_edge":    w_edge,
                    "w_row":     w_row,
                    "w_field":   w_field,
                    "t_mode":    t_mode,
                    "bad_rate":  bad_rate,
                    "threshold": float(threshold),
                    "scores":    scores_np,   # shape (N,) — kept for video render
                    "labels":    labels,      # shape (N,)
                })
    return configs


def evaluate_configs(configs, indices, existing_badframes_path):
    """Add F1/precision/recall to every config dict in-place. Returns has_gt bool."""
    evaluated_set = set(indices)
    start_f = min(indices)
    end_f   = max(indices)

    manual_bad_eval = set()
    if existing_badframes_path and Path(existing_badframes_path).exists():
        manual_ranges   = parse_badframe_ranges(existing_badframes_path)
        manual_bad      = expand_ranges_to_set(manual_ranges, start_f, end_f)
        manual_bad_eval = evaluated_set.intersection(manual_bad)

    has_gt = len(manual_bad_eval) > 0

    for cfg in configs:
        bad_frames    = sorted(
            indices[i] for i, lbl in enumerate(cfg["labels"]) if lbl == "bad"
        )
        predicted_bad = set(bad_frames)

        if has_gt:
            tp   = len(predicted_bad & manual_bad_eval)
            fp   = len(predicted_bad - manual_bad_eval)
            fn   = len(manual_bad_eval - predicted_bad)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        else:
            tp = fp = fn = -1
            prec = rec = f1 = float("nan")

        cfg.update({
            "bad_frames":        bad_frames,
            "tp": tp, "fp": fp, "fn": fn,
            "precision":         prec,
            "recall":            rec,
            "f1":                f1,
            "manual_bad_count":  len(manual_bad_eval),
        })

    return has_gt


def print_ranking(configs, top_n=20):
    ranked = sorted(
        [c for c in configs if not np.isnan(c["f1"])],
        key=lambda c: (-c["f1"], -c["precision"]),
    )
    no_gt = [c for c in configs if np.isnan(c["f1"])]

    print(f"\n{'─'*90}")
    print(f"TOP {min(top_n, len(ranked))} CONFIGURATIONS BY F1")
    print(f"{'─'*90}")
    print(f"{'Rk':<4} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TP':>5} {'FP':>5} {'FN':>5}  Label")
    print(f"{'─'*90}")
    for rank, cfg in enumerate(ranked[:top_n], 1):
        star = "★" if cfg["f1"] >= 0.80 else " "
        print(
            f"{rank:<4} {cfg['f1']:>6.3f} {cfg['precision']:>6.3f} {cfg['recall']:>6.3f}"
            f" {cfg['tp']:>5} {cfg['fp']:>5} {cfg['fn']:>5} {star} {cfg['label']}"
        )

    if no_gt:
        print(f"\n(No ground-truth comparison for {len(no_gt)} configs — "
              "check existing_badframes path)")

    if ranked:
        best = ranked[0]
        print(f"\n{'─'*90}")
        print(f"Best : {best['label']}")
        print(f"  F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")
        print(f"  Threshold={best['threshold']:.6f}")
        print(f"  Bad frames predicted: {len(best['bad_frames'])} / "
              f"manual: {best['manual_bad_count']}")
    print()
    return ranked


# ---------------------------------------------------------------------------
# Comparison video
# ---------------------------------------------------------------------------

def _render_tile(frame_bgr, cfg, pos, tile_w, tile_h, panel_h):
    """
    Resize frame to tile_w × tile_h, then add a verdict bar (panel_h px tall).
    Returns an image of shape (tile_h + panel_h, tile_w, 3).
    """
    score   = float(cfg["scores"][pos])
    is_bad  = score >= cfg["threshold"]
    verdict = "BAD" if is_bad else "good"

    thumb = cv2.resize(frame_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

    # Coloured border: red = bad, green = good
    border_col = (0, 0, 200) if is_bad else (0, 180, 0)
    cv2.rectangle(thumb, (0, 0), (tile_w - 1, tile_h - 1), border_col, 4)

    # Verdict panel below the thumbnail
    bar = np.zeros((panel_h, tile_w, 3), dtype=np.uint8)
    bar[:] = (40, 20, 20) if is_bad else (20, 40, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, cfg["label"][:28],               (4, 16), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, f"s={score:+.3f} t={cfg['threshold']:+.3f}", (4, 32), font, 0.33, (180, 180,  80), 1, cv2.LINE_AA)
    cv2.putText(bar, verdict,                          (4, 50), font, 0.45,
                (80, 80, 255) if is_bad else (80, 220, 80), 1, cv2.LINE_AA)

    return np.vstack([thumb, bar])


def build_comparison_video(
    video_path,
    indices,
    selected_configs,
    output_path,
    tile_w=TILE_WIDTH,
    tile_h=TILE_HEIGHT,
    panel_h=PANEL_H,
    fps=VIDEO_FPS,
    sample_step=SAMPLE_STEP,
):
    """
    For every sampled frame, render one tile per config horizontally tiled
    and write to an mp4.  Layout: tile_w*N wide, (tile_h+panel_h) tall.
    """
    if not selected_configs:
        print("No configs for comparison video — skipping.")
        return

    n_cols  = len(selected_configs)
    frame_w = tile_w * n_cols
    frame_h = tile_h + panel_h

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    # Map frame_idx → position in the indices list for score lookup
    idx_pos = {fi: pos for pos, fi in enumerate(indices)}

    sampled = [fi for fi in indices if (fi - indices[0]) % sample_step == 0]

    print(f"\nBuilding comparison video  ({len(sampled)} frames × {n_cols} configs)")
    print(f"  Output : {output_path}")
    pbar = tqdm(total=len(sampled), desc="Rendering", unit="frame")

    for frame_idx in sampled:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            pbar.update(1)
            continue

        pos   = idx_pos.get(frame_idx, 0)
        tiles = [_render_tile(frame_bgr, cfg, pos, tile_w, tile_h, panel_h)
                 for cfg in selected_configs]
        row   = np.hstack(tiles)

        # Frame index stamp top-left
        cv2.putText(
            row, f"frame {frame_idx}", (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(row)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"Comparison video written → {output_path}")


# ---------------------------------------------------------------------------
# Main sweep orchestrator
# ---------------------------------------------------------------------------

def run_sweep(
    archive,
    video_path,
    scores_tsv_path,
    existing_badframes_path,
    output_dir,
    start_frame,
    max_frame,
    top_n,
    video_configs,
    sample_step,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Get raw per-signal scores ─────────────────────────────────────────
    if scores_tsv_path and Path(scores_tsv_path).exists():
        print(f"Loading pre-computed scores from: {scores_tsv_path}")
        indices, _, edge_raw, row_raw, field_raw = load_scores_tsv(scores_tsv_path)
        edge_raw  = [float(v) for v in edge_raw]
        row_raw   = [float(v) for v in row_raw]
        field_raw = [float(v) for v in field_raw]
    else:
        if not video_path or not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        print(f"Scoring video (once): {video_path}")
        _, _sf, _ef, indices, edge_raw, row_raw, field_raw = score_video_frames(
            video_path=video_path,
            start_frame=start_frame,
            max_frame=max_frame,
            frame_step=1,
            crop_top=0, crop_bottom=0, crop_left=0, crop_right=0,
            sobel_ksize=sanitize_sobel_ksize(3),
        )
        # Save raw scores so future runs can skip decoding
        raw_tsv = output_dir / "frame_scores_raw.tsv"
        with raw_tsv.open("w", encoding="utf-8") as f:
            f.write("frame\tscore\tedge_energy\trow_instability\tfield_mismatch\n")
            for fi, e, r, fld in zip(indices, edge_raw, row_raw, field_raw):
                f.write(f"{fi}\t0.0\t{e:.8f}\t{r:.8f}\t{fld:.8f}\n")
        print(f"Raw scores saved → {raw_tsv}")
        print("TIP: pass --scores-tsv to this path on future runs to skip video decoding.\n")

    indices = list(indices)

    # ── 2. Build all configs in-memory (no I/O per config) ───────────────────
    n_total = len(WEIGHT_COMBOS) * len(THRESHOLD_CONFIGS)
    print(f"Building {len(WEIGHT_COMBOS)} weight combos × {len(THRESHOLD_CONFIGS)} "
          f"threshold configs = up to {n_total} sweep configs...")
    configs = build_configs(edge_raw, row_raw, field_raw, indices)
    print(f"  {len(configs)} valid configs after filtering.\n")

    # ── 3. Evaluate against ground truth ─────────────────────────────────────
    has_gt = evaluate_configs(configs, indices, existing_badframes_path)
    if not has_gt:
        print("WARNING: no ground-truth bad frames found — F1 metrics unavailable.\n")

    # ── 4. Rank and print table ───────────────────────────────────────────────
    ranked = print_ranking(configs, top_n=top_n)

    # ── 5. Save sweep results JSON ────────────────────────────────────────────
    results_path = output_dir / "sweep_results.json"
    serialisable = []
    for cfg in ranked:
        row = {k: v for k, v in cfg.items() if k not in ("scores", "labels", "bad_frames")}
        row["bad_frame_count"] = len(cfg["bad_frames"])
        serialisable.append(row)
    results_path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
    print(f"Sweep results JSON → {results_path}")

    # ── 6. Comparison video ───────────────────────────────────────────────────
    vpath = Path(video_path) if video_path else None
    if vpath and vpath.exists() and video_configs > 0:
        # Use top-ranked configs; fall back to unranked list if no ground truth
        video_cfgs = (ranked if ranked else configs)[:video_configs]
        comp_path  = output_dir / "comparison.mp4"
        build_comparison_video(
            video_path       = vpath,
            indices          = indices,
            selected_configs = video_cfgs,
            output_path      = comp_path,
            sample_step      = sample_step,
        )
    elif video_configs > 0:
        print("Skipping comparison video: video file not available.")

    return ranked, configs


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep tracking_loss classifier parameters + comparison video"
    )
    p.add_argument("--archive",    default="callahan_01_archive")
    p.add_argument("--video",      default="",
                   help="Path to video file. Defaults to ARCHIVE_DIR/<archive>.mkv")
    p.add_argument("--scores-tsv", default="",
                   help="Pre-computed frame_scores.tsv — skips video decoding.")
    p.add_argument("--existing-badframes", default="",
                   help="Ground-truth badframes.tsv. Defaults to archive metadata dir.")
    p.add_argument("--output-dir", default="",
                   help="Where to write sweep outputs. Defaults to metadata dir.")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frame",   type=int, default=-1)
    p.add_argument("--top",         type=int, default=20,
                   help="How many top results to print in the ranking table.")
    p.add_argument("--video-configs", type=int, default=6,
                   help="How many top configs to show side-by-side in the comparison video. "
                        "0 = skip video generation.")
    p.add_argument("--sample-step", type=int, default=SAMPLE_STEP,
                   help=f"Write every Nth frame to the comparison video (default: {SAMPLE_STEP}).")
    return p.parse_args()


def main():
    args = parse_args()
    archive = args.archive

    video_path = args.video or str(ARCHIVE_DIR / f"{archive}.mkv")
    existing_badframes = (
        args.existing_badframes
        or str(METADATA_DIR / archive / "badframes.tsv")
    )
    output_dir = (
        args.output_dir
        or str(METADATA_DIR / archive / "tracking_loss_sweep")
    )

    run_sweep(
        archive                 = archive,
        video_path              = video_path,
        scores_tsv_path         = args.scores_tsv or None,
        existing_badframes_path = existing_badframes,
        output_dir              = output_dir,
        start_frame             = args.start_frame,
        max_frame               = args.max_frame,
        top_n                   = args.top,
        video_configs           = args.video_configs,
        sample_step             = args.sample_step,
    )


if __name__ == "__main__":
    main()
