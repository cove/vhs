#!/usr/bin/env python3.11
"""
VHS Bad Frame Tuner — Gradio edition
=====================================
Run:  python vhs_tuner.py

Requires:  pip install gradio opencv-python-headless numpy pillow pandas

Metadata layout (all under metadata/<archive>/)
────────────────────────────────────────────────
  tracking_badframe/
    <slug>_signals.tsv          raw per-frame signals cache (accumulates across sessions)
    <slug>_overrides.tsv        manual good/bad overrides from the tuner UI
    <slug>_tuner_config.json    last-saved weight / IQR settings for this chapter
    <slug>_badframes.tsv        chapter-specific bad ranges (auto + overrides merged)
  badframes.tsv                 ARCHIVE-LEVEL canonical file — read by step_6_make_videos
                                 = union of all chapters' results with overrides applied

step_6_make_videos.py reads  metadata/<archive>/badframes.tsv
                              which is kept up-to-date by "Apply & Regenerate"
"""

from __future__ import annotations

import ast
import base64
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

# ── Project paths ─────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "scripts" else _HERE
sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE_DIR  = PROJECT_ROOT / "../Archive"
METADATA_DIR = PROJECT_ROOT / "metadata"
STEP6        = PROJECT_ROOT / "step_6_make_videos.py"
FPS          = 30000 / 1001
BORDER       = 3

try:
    from tracking_loss import TrackingLossConfig, run_tracking_loss_classification
    _HAS_TRACKING = True
except ImportError:
    TrackingLossConfig = None            # type: ignore
    run_tracking_loss_classification = None  # type: ignore
    _HAS_TRACKING = False


# ═══════════════════════════════════════════════════════════════════════════════
# Chapter / metadata helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ffmetadata_chapters(path: Path) -> list[dict]:
    chapters, current = [], {}
    default_tb = 1.0 / 1_000_000_000
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line == "[CHAPTER]":
            if "start_raw" in current:
                chapters.append(current)
            current = {"_tb": default_tb}
            continue
        if "=" not in line or line.startswith(";"):
            continue
        key, _, val = line.partition("=")
        key, val = key.strip().upper(), val.strip()
        if key == "TIMEBASE":
            try:
                n, d = val.split("/"); current["_tb"] = float(n) / float(d)
            except Exception:
                pass
        elif key == "START":
            try: current["start_raw"] = int(val)
            except Exception: pass
        elif key == "END":
            try: current["end_raw"] = int(val)
            except Exception: pass
        elif key == "TITLE":
            current["title"] = val
    if "start_raw" in current:
        chapters.append(current)
    result = []
    for ch in chapters:
        tb    = ch.get("_tb", default_tb)
        s_sec = ch.get("start_raw", 0) * tb
        e_sec = ch.get("end_raw",   0) * tb
        result.append({
            "title":       ch.get("title", "Untitled"),
            "start_sec":   s_sec,  "end_sec":   e_sec,
            "start_frame": int(round(s_sec * FPS)),
            "end_frame":   int(round(e_sec * FPS)),
        })
    return result


def slugify(title: str) -> str:
    return re.sub(r"[^\w]+", "_", str(title).strip()).strip("_").lower()


def _tracking_dir(archive: str) -> Path:
    d = METADATA_DIR / archive / "tracking_badframe"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _signals_cache_path(archive: str, ch_title: str) -> Path:
    return _tracking_dir(archive) / f"{slugify(ch_title)}_signals.tsv"

def _overrides_path(archive: str, ch_title: str) -> Path:
    return _tracking_dir(archive) / f"{slugify(ch_title)}_overrides.tsv"

def _tuner_config_path(archive: str, ch_title: str) -> Path:
    return _tracking_dir(archive) / f"{slugify(ch_title)}_tuner_config.json"

def _chapter_badframes_path(archive: str, ch_title: str) -> Path:
    return _tracking_dir(archive) / f"{slugify(ch_title)}_badframes.tsv"

def _archive_badframes_path(archive: str) -> Path:
    p = METADATA_DIR / archive / "badframes.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Bad-frame range I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_badframes_tsv(path: Path) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("start_frame"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            s, e   = int(parts[0]), int(parts[1])
            note   = parts[2].strip() if len(parts) > 2 else ""
            rows.append((s, e, note))
        except ValueError:
            continue
    return rows


def _ranges_from_sorted_frames(frame_ids: list[int]) -> list[tuple[int, int]]:
    if not frame_ids:
        return []
    result: list[tuple[int, int]] = []
    start = prev = frame_ids[0]
    for v in frame_ids[1:]:
        if v == prev + 1:
            prev = v
        else:
            result.append((start, prev))
            start = prev = v
    result.append((start, prev))
    return result


def merge_chapter_into_archive_badframes(
    archive: str,
    ch_start: int,
    ch_end: int,
    chapter_ranges: list[tuple[int, int]],
    overrides: dict[int, str],
) -> Path:
    """
    Merge a chapter's auto-detected bad ranges + manual overrides into the
    archive-level badframes.tsv without disturbing other chapters.

    Algorithm
    ─────────
    1. Read existing archive badframes.tsv.
    2. Drop rows that overlap [ch_start, ch_end] (stale from previous run).
    3. Build new chapter bad-frame set from chapter_ranges.
    4. Apply manual overrides: "bad" forces a frame in; "good" forces it out.
    5. Rebuild ranges and merge back with surviving rows from other chapters.
    6. Write and return the archive path.
    """
    archive_path = _archive_badframes_path(archive)

    existing = _parse_badframes_tsv(archive_path)
    other_rows: list[tuple[int, int, str]] = [
        (s, e, note) for s, e, note in existing
        if e < ch_start or s > ch_end
    ]

    chapter_bad: set[int] = set()
    for s, e in chapter_ranges:
        chapter_bad.update(range(s, e + 1))

    manual_bad_fids:  list[int] = []
    manual_good_fids: list[int] = []
    for fid, label in overrides.items():
        if label == "bad":
            chapter_bad.add(fid)
            manual_bad_fids.append(fid)
        elif label == "good":
            chapter_bad.discard(fid)
            manual_good_fids.append(fid)

    ch_note     = "tracking_loss+manual" if (manual_bad_fids or manual_good_fids) else "tracking_loss_chapter_iqr"
    new_ch_rows = [(s, e, ch_note) for s, e in _ranges_from_sorted_frames(sorted(chapter_bad))]

    all_rows = sorted(other_rows + new_ch_rows, key=lambda r: r[0])

    with archive_path.open("w", encoding="utf-8") as f:
        f.write("start_frame\tend_frame\tnote\n")
        for s, e, note in all_rows:
            f.write(f"{s}\t{e}\t{note}\n")
        if manual_good_fids:
            f.write(f"# Manual good overrides (forced good): {sorted(manual_good_fids)}\n")
        if manual_bad_fids:
            f.write(f"# Manual bad overrides  (forced bad):  {sorted(manual_bad_fids)}\n")

    return archive_path


# ═══════════════════════════════════════════════════════════════════════════════
# Signal cache
# ═══════════════════════════════════════════════════════════════════════════════

def load_cached_signals(archive: str, ch_title: str) -> tuple[list[int] | None, dict | None]:
    p = _signals_cache_path(archive, ch_title)
    if not p.exists():
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(p, sep="\t", comment="#")
        df.columns = [c.strip().lower() for c in df.columns]
        if not {"frame","chroma_loss","noise_energy","row_tear","wave_energy"}.issubset(df.columns):
            return None, None
        fids = df["frame"].astype(int).tolist()
        return fids, {
            "chroma": df["chroma_loss"].values.astype(np.float64),
            "noise":  df["noise_energy"].values.astype(np.float64),
            "tear":   df["row_tear"].values.astype(np.float64),
            "wave":   df["wave_energy"].values.astype(np.float64),
        }
    except Exception:
        return None, None


def save_cached_signals(archive: str, ch_title: str,
                        fids: list[int], sigs: dict) -> None:
    p = _signals_cache_path(archive, ch_title)
    with p.open("w", encoding="utf-8") as f:
        f.write("frame\tchroma_loss\tnoise_energy\trow_tear\twave_energy\n")
        for i, fid in enumerate(fids):
            f.write(f"{fid}\t{sigs['chroma'][i]:.8f}\t{sigs['noise'][i]:.8f}"
                    f"\t{sigs['tear'][i]:.8f}\t{sigs['wave'][i]:.8f}\n")


def load_overrides(archive: str, ch_title: str) -> dict[int, str]:
    p = _overrides_path(archive, ch_title)
    if not p.exists():
        return {}
    out: dict[int, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("frame"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            fid   = int(parts[0])
            label = parts[1].strip().lower()
            if label in ("good", "bad"):
                out[fid] = label
        except Exception:
            pass
    return out


def save_overrides(archive: str, ch_title: str, overrides: dict[int, str]) -> None:
    p = _overrides_path(archive, ch_title)
    lines = ["frame\toverride"]
    for fid in sorted(overrides):
        lines.append(f"{fid}\t{overrides[fid]}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_overrides_from_archive_badframes(
    archive: str,
    ch_start: int,
    ch_end: int,
) -> dict[int, str]:
    """
    Read manual override comments from the archive-level badframes.tsv and
    return any that fall within [ch_start, ch_end].  These comments look like:
      # Manual good overrides (forced good): [12345, 12399]
      # Manual bad overrides  (forced bad):  [12350]
    """
    out: dict[int, str] = {}
    p = _archive_badframes_path(archive)
    if not p.exists():
        return out

    def _parse_list_tail(line: str) -> list[int]:
        if ":" not in line:
            return []
        tail = line.split(":", 1)[1].strip()
        if not tail:
            return []
        try:
            vals = ast.literal_eval(tail)
        except Exception:
            return []
        if not isinstance(vals, list):
            return []
        clean = []
        for v in vals:
            try:
                clean.append(int(v))
            except Exception:
                pass
        return clean

    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if s.startswith("# Manual good overrides"):
            for fid in _parse_list_tail(s):
                if ch_start <= fid <= ch_end:
                    out[fid] = "good"
        elif s.startswith("# Manual bad overrides"):
            for fid in _parse_list_tail(s):
                if ch_start <= fid <= ch_end:
                    out[fid] = "bad"
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Signal computation + frame extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_signals(bgr: np.ndarray, crop: int = 50) -> tuple[float, float, float, float]:
    h, w = bgr.shape[:2]
    y0 = min(crop, max(0, h-1)); y1 = max(y0+1, h-crop)
    x0 = min(crop, max(0, w-1)); x1 = max(x0+1, w-crop)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = bgr
    s           = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    chroma_loss = 1.0 - float(np.mean(s) / 255.0)
    gray        = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_vars    = np.var(gray, axis=1)
    mean_var    = float(np.mean(row_vars))
    noise       = float(np.std(row_vars) / mean_var) if mean_var > 1e-6 else 0.0
    tear        = (float(np.percentile(np.abs(gray[1:] - gray[:-1]).mean(axis=1), 95))
                   if gray.shape[0] > 1 else 0.0)
    row_sums    = gray.sum(axis=1)
    cols_idx    = np.arange(gray.shape[1], dtype=np.float32)
    row_com     = (gray @ cols_idx) / np.maximum(row_sums, 1e-6)
    wave        = (float(np.std(row_com - np.convolve(row_com, np.ones(5)/5, mode="same")))
                   if row_com.shape[0] >= 5 else float(np.std(row_com)))
    return chroma_loss, noise, tear, wave


def _bgr_to_jpeg_b64(bgr: np.ndarray, width: int = 160) -> str:
    h, w = bgr.shape[:2]
    thumb = cv2.resize(bgr, (width, int(width * h / max(w, 1))), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def extract_frames(
    video_path: str,
    start: int, end: int, n: int,
    archive: str, ch_title: str,
    progress=None,
) -> tuple[list[int] | None, list[str] | None, dict | None, str]:
    frame_ids: list[int] = np.linspace(int(start), int(end), int(n), dtype=int).tolist()
    frame_set             = set(frame_ids)

    cached_fids, cached_sigs = load_cached_signals(archive, ch_title)
    cached_lookup: dict[int, dict[str, float]] = {}
    if cached_fids and cached_sigs:
        for i, fid in enumerate(cached_fids):
            cached_lookup[fid] = {k: float(v[i]) for k, v in cached_sigs.items()}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None, f"Cannot open video: {video_path}"

    frames_b64: list[str] = []
    chroma_s, noise_s, tear_s, wave_s = [], [], [], []

    for idx, fid in enumerate(frame_ids):
        if progress is not None:
            progress(idx / len(frame_ids), desc=f"Frame {fid}…")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        frames_b64.append(_bgr_to_jpeg_b64(bgr))

        if fid in cached_lookup:
            c = cached_lookup[fid]
            chroma_s.append(c["chroma"]); noise_s.append(c["noise"])
            tear_s.append(c["tear"]);     wave_s.append(c["wave"])
        else:
            ch, no, te, wa = _compute_signals(bgr)
            chroma_s.append(ch); noise_s.append(no)
            tear_s.append(te);   wave_s.append(wa)

    cap.release()

    sigs: dict[str, np.ndarray] = {
        "chroma": np.array(chroma_s, dtype=np.float64),
        "noise":  np.array(noise_s,  dtype=np.float64),
        "tear":   np.array(tear_s,   dtype=np.float64),
        "wave":   np.array(wave_s,   dtype=np.float64),
    }

    # Merge into persistent cache
    all_fids_l: list[int] = list(frame_ids)
    all_sigs_l: dict[str, list[float]] = {k: list(v) for k, v in sigs.items()}
    if cached_fids and cached_sigs:
        for i, fid in enumerate(cached_fids):
            if fid not in frame_set:
                all_fids_l.append(fid)
                for k, arr in cached_sigs.items():
                    all_sigs_l[k].append(float(arr[i]))
    order       = list(np.argsort(all_fids_l))
    sorted_fids = [all_fids_l[i] for i in order]
    sorted_sigs = {k: np.array([v[i] for i in order]) for k, v in all_sigs_l.items()}
    save_cached_signals(archive, ch_title, sorted_fids, sorted_sigs)

    return frame_ids, frames_b64, sigs, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring / thresholding
# ═══════════════════════════════════════════════════════════════════════════════

def _robust_z(v: np.ndarray) -> np.ndarray:
    v   = np.asarray(v, dtype=np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    sc  = 1.4826 * mad
    if sc < 1e-12:
        sc = float(np.std(v)) or 1.0
    return (v - med) / sc


def combined_score(sigs: dict, wc: float, wn: float, wt: float, ww: float) -> np.ndarray:
    wsum = wc + wn + wt + ww or 1.0
    return (
        _robust_z(sigs["chroma"]) * wc +
        _robust_z(sigs["noise"])  * wn +
        _robust_z(sigs["tear"])   * wt +
        _robust_z(sigs["wave"])   * ww
    ) / wsum


def compute_threshold(
    scores: np.ndarray,
    mode: str,
    iqr_mult: float,
    thresh_val: float,
    bad_pct: float,
) -> float:
    v = scores[np.isfinite(scores)]
    if v.size == 0:
        return 0.0
    if mode == "iqr":
        q1, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
        return q3 + iqr_mult * (q3 - q1)
    if mode == "value":
        return float(thresh_val)
    return float(np.quantile(v, 1.0 - bad_pct / 100.0))


def toggle_frame_override(
    fid: int,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
) -> dict[int, str]:
    """
    Toggle one frame between:
    - auto state (no override)
    - forced opposite of auto state
    """
    out = dict(overrides)
    sc = combined_score(sigs, wc, wn, wt, ww)
    thr = compute_threshold(sc, tm, ik, tv, bp)
    idx = fids.index(fid) if fid in fids else -1
    auto_bad = (idx >= 0 and sc[idx] >= thr)

    if fid in out:
        del out[fid]
    else:
        out[fid] = "good" if auto_bad else "bad"
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SVG Sparklines  — timeline charts with horizontal red cut line
# ═══════════════════════════════════════════════════════════════════════════════

def _sparkline_svg(
    values: np.ndarray,
    threshold: float | None = None,
    label: str = "",
    height: int = 38,
    line_color: str = "#27a85a",
) -> str:
    """
    Timeline sparkline: X axis = frame index, Y axis = signal value.
    An optional horizontal red line marks the threshold cut.
    Uses width:100% so it fills whatever column it sits in.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    SVG_W = 200  # viewBox width — browser scales to container

    if v.size == 0:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {height}" '
            f'style="background:#0a0a0a;display:block;width:100%;border-radius:2px;margin-bottom:3px">'
            f'<text x="4" y="14" font-family="Courier New" font-size="9" fill="#444">{label}</text>'
            f'</svg>'
        )

    vmin   = float(v.min())
    vmax   = float(v.max())
    vrange = (vmax - vmin) or 1.0
    n      = len(v)
    PAD    = 3  # top padding so high points aren't clipped

    def _x(i: int) -> float:
        return i / max(n - 1, 1) * SVG_W

    def _y(val: float) -> float:
        return PAD + (height - PAD) * (1.0 - (val - vmin) / vrange)

    pts = " ".join(f"{_x(i):.1f},{_y(val):.1f}" for i, val in enumerate(v))

    # Filled area under the line
    area_pts = f"0,{height} {pts} {SVG_W},{height}"

    # Threshold line + right-edge triangle marker
    tline = ""
    if threshold is not None:
        ty = _y(threshold)
        ty = max(0.0, min(float(height), ty))
        tline = (
            f'<line x1="0" y1="{ty:.1f}" x2="{SVG_W}" y2="{ty:.1f}" '
            f'stroke="#e03030" stroke-width="1.8" opacity="0.95"/>'
            f'<polygon points="{SVG_W},{ty:.1f} {SVG_W-6},{ty-4:.1f} {SVG_W-6},{ty+4:.1f}" '
            f'fill="#e03030" opacity="0.9"/>'
        )

    lbl = (
        f'<text x="3" y="{height - 3}" font-family="Courier New" '
        f'font-size="8" fill="#555">{label}</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {height}" '
        f'style="background:#0a0a0a;display:block;width:100%;border-radius:2px;margin-bottom:3px">'
        f'<polygon points="{area_pts}" fill="{line_color}" opacity="0.12"/>'
        f'<polyline points="{pts}" fill="none" stroke="{line_color}" stroke-width="1.3" opacity="0.85"/>'
        f'{tline}{lbl}'
        f'</svg>'
    )


def build_sparklines_html(
    sigs: dict,
    scores: np.ndarray,
    threshold: float,
    wc: float, wn: float, wt: float, ww: float,
) -> tuple[str, str, str, str, str]:
    """
    Returns (spark_chroma, spark_noise, spark_tear, spark_wave, spark_score).

    Signal sparklines show raw values over the sampled frames.
    The line opacity tracks the current weight (dim when weight is low).
    The composite score sparkline carries the red threshold line.
    """
    def _col(w: float) -> str:
        alpha = 0.25 + 0.75 * min(1.0, w / 0.5)
        return f"rgba(39,168,90,{alpha:.2f})"

    def _unit(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return arr
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - lo) / (hi - lo)

    sc_chroma = _sparkline_svg(
        _unit(sigs.get("chroma", np.array([]))),
        wc,
        f"chroma  w={wc:.2f}",
        line_color=_col(wc),
    )
    sc_noise = _sparkline_svg(
        _unit(sigs.get("noise", np.array([]))),
        wn,
        f"noise   w={wn:.2f}",
        line_color=_col(wn),
    )
    sc_tear = _sparkline_svg(
        _unit(sigs.get("tear", np.array([]))),
        wt,
        f"tear    w={wt:.2f}",
        line_color=_col(wt),
    )
    sc_wave = _sparkline_svg(
        _unit(sigs.get("wave", np.array([]))),
        ww,
        f"wave    w={ww:.2f}",
        line_color=_col(ww),
    )
    sc_score  = _sparkline_svg(scores, threshold, "composite score",
                                height=52, line_color="#5599dd")

    return sc_chroma, sc_noise, sc_tear, sc_wave, sc_score


# ═══════════════════════════════════════════════════════════════════════════════
# Frame grid HTML
# ═══════════════════════════════════════════════════════════════════════════════

# NOTE: _GRID_JS is intentionally empty - the actual JS lives in a static
# gr.HTML component that is never included in event outputs, so it survives
# grid rebuilds. Cells call window.vhsToggleFrame(fid) via inline onclick.
_GRID_JS = ""


def build_grid_html(
    frames_b64: list[str],
    fids: list[int],
    scores: np.ndarray,
    overrides: dict[int, str],
    threshold: float,
    cols: int,
    thumb_w: int,
) -> str:
    if not fids:
        return "<p style='color:#666;font-family:monospace;padding:20px'>No frames loaded.</p>"

    cells = []
    for b64, fid, sc in zip(frames_b64, fids, scores):
        ov    = overrides.get(int(fid))
        auto  = sc >= threshold
        bad   = (ov == "bad") if ov else auto
        if ov == "bad":
            color = "#8b1f1f"   # dark red = manual bad
            badge = " M:BAD"
        elif ov == "good":
            color = "#1f6b3a"   # dark green = manual good
            badge = " M:GOOD"
        else:
            color = "#e03030" if bad else "#30c870"
            badge = ""
        label = f"#{fid} {sc:.2f}{badge}"
        cells.append(
            f'<div class="vhs-cell" data-fid="{fid}" onclick="if(window.vhsToggleFrame){{window.vhsToggleFrame({fid});}} return false;"'
            f' title="frame {fid} · score {sc:.4f} · click to toggle">'
            f'<div class="vhs-wrap" style="border-color:{color}">'
            f'<img src="{b64}" class="vhs-thumb"/></div>'
            f'<div class="vhs-lbl" style="color:{color}">{label}</div>'
            f'</div>'
        )

    return f"""
{_GRID_JS}
<style>
  .vhs-grid {{
    display:grid;
    grid-template-columns:repeat({cols},{thumb_w}px);
    gap:5px; background:#0d0d0d; padding:8px;
  }}
  .vhs-cell {{ display:flex; flex-direction:column; align-items:center;
               cursor:pointer; user-select:none; }}
  .vhs-cell:hover .vhs-wrap {{ opacity:0.75; transform:scale(1.03); }}
  .vhs-wrap {{ border:{BORDER}px solid; line-height:0;
               transition:opacity .1s, transform .1s; }}
  .vhs-thumb {{ display:block; width:{thumb_w}px; }}
  .vhs-lbl {{ font-family:'Courier New',monospace; font-size:9px;
              margin-top:2px; white-space:nowrap; }}
</style>
<div class="vhs-grid">{''.join(cells)}</div>
"""


def build_gallery_items(
    frames_b64: list[str],
    fids: list[int],
    scores: np.ndarray,
    overrides: dict[int, str],
    threshold: float,
) -> list[tuple[Image.Image, str]]:
    items: list[tuple[Image.Image, str]] = []
    for b64, fid, sc in zip(frames_b64, fids, scores):
        # Gradio Gallery.select in v6 rejects data: URIs in event payload.
        # Convert to in-memory PIL images so selected items are cache-safe.
        try:
            payload = b64.split(",", 1)[1] if "," in b64 else b64
            img = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (160, 90), (20, 20, 20))
        ov = overrides.get(int(fid))
        auto_bad = sc >= threshold
        if ov == "bad":
            state = "M:BAD"
            state_short = "MB"
            color = "#8b1f1f"   # dark red = manual bad
        elif ov == "good":
            state = "M:GOOD"
            state_short = "MG"
            color = "#1f6b3a"   # dark green = manual good
        else:
            state = "AUTO:BAD" if auto_bad else "AUTO:GOOD"
            state_short = "AB" if auto_bad else "AG"
            color = "#e03030" if auto_bad else "#30c870"

        # Restore fast visual scanning: colored border per frame state.
        styled = ImageOps.expand(img, border=BORDER, fill=color)
        items.append((styled, f"#{fid}  s={sc:.2f}  {state_short}"))
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Preview video: ffmpeg burn-in  G/L/S  (global frame, local frame, score)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_score_ass(path: Path, fids: list[int], scores: np.ndarray,
                     chapter_start_frame: int, fps: float,
                     total_local_frames: int) -> None:
    score_map   = {int(f): float(s) for f, s in zip(fids, scores)}
    sorted_fids = sorted(score_map)
    if not sorted_fids:
        path.write_text("", encoding="utf-8"); return

    def _t(lf: int) -> str:
        secs = max(0.0, lf / fps)
        h = int(secs//3600); m = int((secs%3600)//60); s = secs%60
        return f"{h}:{m:02d}:{int(s):02d}.{int((s%1)*100):02d}"

    events = []
    for i, fid in enumerate(sorted_fids):
        lo = max(0, fid - chapter_start_frame)
        hi = (sorted_fids[i+1] - chapter_start_frame
              if i+1 < len(sorted_fids) else total_local_frames + 60)
        events.append(f"Dialogue: 0,{_t(lo)},{_t(hi)},Sc,,0,0,0,,S:{score_map[fid]:.2f}")

    path.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,"
        "Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Sc,Courier New,18,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,"
        "1,0,0,0,100,100,0,0,1,2,0,7,6,6,36,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        + "\n".join(events),
        encoding="utf-8",
    )


def make_preview_video(input_path: str | Path, output_path: str | Path,
                       chapter_start_frame: int, fids: list[int],
                       scores: np.ndarray, fps: float = FPS) -> Path:
    input_path  = Path(input_path)
    output_path = Path(output_path)
    ass_path    = output_path.with_suffix(".score_overlay.ass")

    cap   = cv2.VideoCapture(str(input_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()

    if fids and len(scores) == len(fids):
        _write_score_ass(ass_path, fids, scores, chapter_start_frame, fps, total)
    else:
        ass_path.write_text("", encoding="utf-8")

    offset = int(chapter_start_frame)
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C\\:/Windows/Fonts/cour.ttf",
    ]
    font_path = next((f for f in font_candidates if Path(f).exists()), "")
    font_arg  = f"fontfile={font_path}:" if font_path else ""

    vf_parts = [
        "scale=iw/2:ih/2",
        f"subtitles='{ass_path}'",
        (f"drawtext={font_arg}fontsize=16:fontcolor=white:x=6:y=6:"
         f"box=1:boxcolor=black@0.75:boxborderw=3:"
         f"text='G\\:%{{eif\\:n+{offset}\\:d\\:7}} L\\:%{{n}}'"),
    ]

    def _run(vf: str) -> bool:
        return subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(input_path),
             "-vf", vf, "-c:v", "libx264", "-crf", "22", "-preset", "fast",
             "-c:a", "aac", "-b:a", "128k", str(output_path)],
            capture_output=True,
        ).returncode == 0

    if not _run(",".join(vf_parts)):
        if not _run(",".join([vf_parts[0], vf_parts[2]])):
            _run(vf_parts[0])

    try: ass_path.unlink()
    except Exception: pass
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Apply: save config + run tracking_loss + merge overrides into archive badframes
# ═══════════════════════════════════════════════════════════════════════════════

def apply_and_regenerate(
    archive: str, ch_title: str,
    ch_start: int, ch_end: int,
    w_chroma: float, w_noise: float, w_tear: float, w_wave: float,
    iqr_mult: float, frame_step: int,
) -> str:
    if not archive or not ch_title or ch_title.startswith("—"):
        return "❌  No chapter selected."

    logs: list[str] = []

    # 1 ── Save tuner config ────────────────────────────────────────────────
    cfg = {
        "archive": archive, "chapter": ch_title,
        "chapter_start_frame": ch_start, "chapter_end_frame": ch_end,
        "weight_chroma": w_chroma, "weight_noise": w_noise,
        "weight_tear":   w_tear,   "weight_wave":  w_wave,
        "iqr_mult": iqr_mult, "frame_step": frame_step,
    }
    cfg_p = _tuner_config_path(archive, ch_title)
    cfg_p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logs.append(f"✅  Config → {cfg_p.name}")

    if not _HAS_TRACKING:
        return "\n".join(logs) + "\n❌  tracking_loss module not found."

    # 2 ── Locate video ─────────────────────────────────────────────────────
    proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
    mkv   = ARCHIVE_DIR / f"{archive}.mkv"
    video = str(proxy if proxy.exists() else mkv if mkv.exists() else "")
    if not video:
        return "\n".join(logs) + f"\n❌  No video found for '{archive}'."

    # 3 ── Run tracking_loss for this chapter ───────────────────────────────
    logs.append(f"▶️   tracking_loss  frames {ch_start}–{ch_end}  step={frame_step}…")
    ch_output_dir = _tracking_dir(archive) / f"{slugify(ch_title)}_tl_output"
    config = TrackingLossConfig(  # type: ignore[call-arg]
        archive              = archive,
        video                = video,
        output_dir           = str(ch_output_dir),
        start_frame          = ch_start,
        max_frame            = ch_end,
        frame_step           = max(1, frame_step),
        weight_chroma        = w_chroma,
        weight_noise         = w_noise,
        weight_tear          = w_tear,
        weight_wave          = w_wave,
        iqr_mult             = iqr_mult,
        threshold_window_size= 1000,
        export_review_png_count = 0,
        # Keep tracking_loss metadata copy local to this chapter run output.
        metadata_badframes_tsv = str(ch_output_dir / "chapter_badframes.metadata.tsv"),
    )
    try:
        result = run_tracking_loss_classification(config=config)  # type: ignore
        frame_scores_out = Path(result.get("frame_scores_path", ch_output_dir / "frame_scores.tsv"))
        logs.append(f"✅  tracking_loss done  →  {frame_scores_out.name}")
    except Exception as exc:
        logs.append(f"❌  tracking_loss failed: {exc}")
        return "\n".join(logs)

    # 4 ── Load auto bad ranges ─────────────────────────────────────────────
    run_badframes: Path | None = None
    if isinstance(result, dict):
        bf = result.get("badframes_path")
        if bf:
            run_badframes = Path(bf)
        if (not run_badframes or not run_badframes.exists()) and result.get("output_dir"):
            fallback = Path(result["output_dir"]) / "badframes.tsv"
            if fallback.exists():
                run_badframes = fallback

    if not run_badframes or not run_badframes.exists():
        logs.append("❌  tracking_loss did not produce badframes.tsv; archive badframes was not updated.")
        return "\n".join(logs)

    auto_ranges = [(s, e) for s, e, _ in _parse_badframes_tsv(run_badframes)]
    logs.append(f"   Auto bad ranges: {len(auto_ranges)}")

    # 5 ── Merge overrides into archive-level badframes.tsv ─────────────────
    overrides      = load_overrides(archive, ch_title)
    manual_bad     = [f for f, l in overrides.items() if l == "bad"]
    manual_good    = [f for f, l in overrides.items() if l == "good"]
    if overrides:
        logs.append(f"   Overrides: {len(manual_bad)} forced-bad, {len(manual_good)} forced-good")

    archive_bf = merge_chapter_into_archive_badframes(
        archive        = archive,
        ch_start       = ch_start,
        ch_end         = ch_end,
        chapter_ranges = auto_ranges,
        overrides      = overrides,
    )
    logs.append(f"✅  Archive badframes → {archive_bf}")
    logs.append(f"   (step_6_make_videos reads this file)")
    return "\n".join(logs)


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio layout
# ═══════════════════════════════════════════════════════════════════════════════

_DARK_CSS = """
body, .gradio-container { background:#0d0d0d !important; color:#ccc; }
.gr-box, .gr-padded { background:#141414 !important; }
.gr-button-primary { background:#1a6b3a !important; border-color:#27a85a !important; }
.gr-button { background:#222 !important; color:#bbb !important; border-color:#333 !important; }
label { color:#999 !important; font-family:'Courier New',monospace !important; font-size:0.77rem !important; }
input[type=range] { accent-color:#27a85a; }
#vhs-stats { font-family:'Courier New',monospace; font-size:13px;
             background:#111; padding:8px 12px; border-left:3px solid #27a85a; }
#vhs-apply-log textarea  { font-family:'Courier New',monospace; font-size:11px;
                           background:#0a0a0a !important; color:#9fdfb8 !important; }
#vhs-render-log textarea { font-family:'Courier New',monospace; font-size:11px;
                           background:#0a0a0a !important; color:#9fdfb8 !important; }
#vhs-click-recv input, #vhs-click-recv textarea {
  height:26px !important; font-family:monospace; font-size:10px;
  background:#111 !important; color:#555 !important; }
#vhs-grid-gallery [class*="caption"] {
  font-family:'Courier New',monospace !important;
  font-size:10px !important;
  text-align:left !important;
  white-space:normal !important;
  overflow-wrap:anywhere !important;
  line-height:1.15 !important;
  padding-left:2px !important;
}
"""

_STEP_BTN_HTML = """
<div style="display:flex;gap:8px;align-items:center;padding:6px 0;
            font-family:'Courier New',monospace;">
  <button onclick="vhsStep(-1)"
    style="background:#1a1a1a;color:#ccc;border:1px solid #333;
           padding:5px 14px;cursor:pointer;border-radius:3px;font-family:inherit;">
    ◀ −1 frame
  </button>
  <button onclick="vhsStep(1)"
    style="background:#1a1a1a;color:#ccc;border:1px solid #333;
           padding:5px 14px;cursor:pointer;border-radius:3px;font-family:inherit;">
    +1 frame ▶
  </button>
  <span id="vhs-step-info" style="color:#666;font-size:11px;"></span>
</div>
<script>
function vhsStep(dir) {
  const fps = 30000/1001;
  const wrap = document.getElementById('vhs-preview-video');
  const vid  = wrap ? wrap.querySelector('video') : document.querySelector('video');
  if (!vid) { document.getElementById('vhs-step-info').textContent='(no video)'; return; }
  vid.pause();
  vid.currentTime = Math.max(0, Math.min(vid.duration||Infinity, vid.currentTime + dir/fps));
  document.getElementById('vhs-step-info').textContent =
    'local ≈ ' + Math.round(vid.currentTime * fps);
}
</script>
"""


def _get_archives() -> list[str]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(p.stem for p in ARCHIVE_DIR.glob("*.mkv"))


def _get_chapter_titles(archive: str) -> list[str]:
    if not archive:
        return ["— select chapter —"]
    cf = METADATA_DIR / archive / "chapters.ffmetadata"
    if not cf.exists():
        return ["— no chapters file found —"]
    chapters = parse_ffmetadata_chapters(cf)
    return ["— select chapter —"] + [ch["title"] for ch in chapters]


def _find_chapter(chapters: list[dict], title: str) -> dict | None:
    return next((c for c in chapters if c["title"] == title), None)


_E_SIG   = _sparkline_svg(np.array([]), None,  "", height=38)
_E_SCORE = _sparkline_svg(np.array([]), None,  "", height=52)


with gr.Blocks(
    title="📼  VHS Frame Tuner",
) as demo:

    # ── Persistent state ──────────────────────────────────────────────────────
    st_fids      = gr.State([])
    st_b64       = gr.State([])
    st_sigs      = gr.State({})
    st_overrides = gr.State({})
    st_chapters  = gr.State([])

    gr.Markdown("# 📼  VHS Frame Tuner")

    # ── Archive + Chapter ─────────────────────────────────────────────────────
    with gr.Row():
        _archives  = _get_archives()
        archive_dd = gr.Dropdown(
            choices=_archives, value=_archives[0] if _archives else None,
            label="Archive", scale=1, interactive=True,
        )
        chapter_dd = gr.Dropdown(
            choices=["— select chapter —"], value="— select chapter —",
            label="Chapter", scale=3, interactive=True,
        )
        load_btn = gr.Button("🔄  Load Chapter", variant="primary", scale=1)

    status_md = gr.Markdown("*Pick an archive and chapter, then click **Load Chapter**.*")

    # ── Main panel ────────────────────────────────────────────────────────────
    with gr.Row(visible=False) as main_panel:

        # ── LEFT column ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=230):

            gr.Markdown("### 🎞  Range & Sample")
            with gr.Row():
                start_n = gr.Number(label="Start frame", value=0,   precision=0)
                end_n   = gr.Number(label="End frame",   value=1000, precision=0)
            n_sl = gr.Slider(20, 400, value=100, step=10, label="Sample n frames")
            apply_range_btn = gr.Button("Apply Range", variant="secondary")

            gr.Markdown("### ⚖️  Signal Weights")
            wc_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="chroma_loss")
            spark_chroma = gr.HTML(_E_SIG)
            wn_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="noise_energy")
            spark_noise  = gr.HTML(_E_SIG)
            wt_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="row_tear")
            spark_tear   = gr.HTML(_E_SIG)
            ww_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="wave_energy")
            spark_wave   = gr.HTML(_E_SIG)

            gr.Markdown("### 🎚️  Threshold")
            t_mode  = gr.Radio(["iqr", "value", "quantile"], value="iqr",
                                label="Mode", interactive=True)
            iqr_sl  = gr.Slider(1.0, 8.0,   value=3.5,  step=0.05,
                                 label="k  (Q3 + k × IQR)")
            tval_sl = gr.Slider(-5.0, 15.0, value=1.0,  step=0.05,
                                 label="Hard threshold value", visible=False)
            bpct_sl = gr.Slider(1, 60,      value=10,   step=1,
                                 label="Bad %", visible=False)
            spark_score = gr.HTML(_E_SCORE)

            gr.Markdown("### 🖼️  Grid Display")
            with gr.Row():
                cols_sl   = gr.Slider(4,  24,  value=10,  step=1,  label="Cols")
                twidth_sl = gr.Slider(80, 320, value=160, step=20, label="px wide")

            gr.Markdown("### ✅  Apply")
            fstep_sl  = gr.Slider(1, 10, value=1, step=1,
                                   label="Full-scan frame step  (1=accurate, 10=fast)")
            apply_btn = gr.Button("✅  Apply & Regenerate", variant="primary")
            apply_log = gr.Textbox(label="Apply log", lines=8,
                                    interactive=False, elem_id="vhs-apply-log")
            reload_btn = gr.Button("🔄  Reload Frames", variant="secondary")

        # ── RIGHT column ──────────────────────────────────────────────────────
        with gr.Column(scale=4):

            stats_md  = gr.Markdown("", elem_id="vhs-stats")
            grid_gallery = gr.Gallery(
                value=[],
                label="Frames (click a tile to toggle good/bad)",
                show_label=True,
                columns=10,
                object_fit="contain",
                height="auto",
                allow_preview=False,
                elem_id="vhs-grid-gallery",
            )

            # ── Static JS — never included in event outputs, so it persists
            #    across grid rebuilds. We keep inline + delegated click paths
            #    with dedupe so clicks still register if one path is filtered.
            gr.HTML("""
<script>
(function() {
  function _setGradio(elemId, val) {
    var c = document.getElementById(elemId);
    if (!c) return;
    var inp = c.querySelector('textarea') || c.querySelector('input');
    if (!inp) return;
    var proto = (inp.tagName === 'TEXTAREA')
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    var desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && desc.set) desc.set.call(inp, val);
    else inp.value = val;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
  }

  var _lastFid = null;
  var _lastTs = 0;
  function _emit(fid, source) {
    var sid = String(fid || '');
    if (!sid) return;
    var now = Date.now();
    if (_lastFid === sid && (now - _lastTs) < 120) {
      _setGradio('vhs-click-debug-js', 'dedupe ' + source + ' fid=' + sid + ' dt=' + (now - _lastTs) + 'ms');
      return;
    }
    _lastFid = sid;
    _lastTs = now;
    var payload = sid + ':' + now;
    _setGradio('vhs-click-debug-js', 'emit ' + source + ' payload=' + payload);
    _setGradio('vhs-click-recv', payload);
  }

  window.vhsToggleFrame = function(fid) { _emit(fid, 'inline'); };
  document.addEventListener('click', function(e) {
    var cell = e.target.closest('.vhs-cell[data-fid]');
    if (!cell) return;
    _emit(cell.getAttribute('data-fid'), 'delegate');
  }, true);

})();
</script>
""")

            with gr.Row():
                click_recv = gr.Textbox(
                    value="", label="Last clicked frame",
                    interactive=True, scale=1, max_lines=1,
                    elem_id="vhs-click-recv",
                )
                click_debug_js = gr.Textbox(
                    value="", label="JS click debug",
                    interactive=False, scale=2, max_lines=1,
                    elem_id="vhs-click-debug-js",
                )
                click_debug_srv = gr.Textbox(
                    value="", label="Server click debug",
                    interactive=False, scale=2, max_lines=1,
                    elem_id="vhs-click-debug-srv",
                )

            gr.Markdown("---")
            gr.Markdown("### 🎬  Render Chapter")

            preview_vid    = gr.Video(
                label="Preview  (½ size · G/L frame + S score burn-in)",
                visible=False, elem_id="vhs-preview-video",
            )
            step_ctrl_html = gr.HTML(_STEP_BTN_HTML, visible=False)

            with gr.Row():
                render_btn = gr.Button("▶️  Render", variant="primary", scale=1)
                render_log = gr.Textbox(
                    label="", lines=3, interactive=False,
                    scale=4, elem_id="vhs-render-log",
                )

    # =========================================================================
    # Rebuild helper — grid + stats + 5 sparklines
    # =========================================================================

    def _rebuild(fids, b64, sigs, overrides, wc, wn, wt, ww,
                 t_mode, iqr_k, tval, bpct, cols, twidth):
        if not fids or not b64:
            return (gr.update(value=[]), "*(no frames loaded)*",
                    _E_SIG, _E_SIG, _E_SIG, _E_SIG, _E_SCORE)
        sc   = combined_score(sigs, wc, wn, wt, ww)
        thr  = compute_threshold(sc, t_mode, iqr_k, tval, bpct)
        gallery_items = build_gallery_items(b64, fids, sc, overrides, thr)
        gallery_update = gr.update(value=gallery_items, columns=int(cols))
        n_bad  = sum(
            (overrides.get(int(f), "bad" if s >= thr else "good") == "bad")
            for f, s in zip(fids, sc)
        )
        n_ov   = sum(1 for f in fids if int(f) in overrides)
        stats  = (
            f"🔴 **Bad:** {n_bad} ({100*n_bad/max(1,len(fids)):.0f}%)  ·  "
            f"🟢 **Good:** {len(fids)-n_bad}  ·  "
            f"**Threshold:** {thr:.3f}  ·  "
            f"✏ **Overrides:** {n_ov}  ·  n={len(fids)}"
        )
        sc_ch, sc_no, sc_te, sc_wa, sc_sc = build_sparklines_html(
            sigs, sc, thr, wc, wn, wt, ww
        )
        return gallery_update, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc

    _RB_OUTS = [grid_gallery, stats_md,
                spark_chroma, spark_noise, spark_tear, spark_wave, spark_score]

    # ── Archive change ─────────────────────────────────────────────────────
    def on_archive(archive):
        titles   = _get_chapter_titles(archive)
        cf       = METADATA_DIR / archive / "chapters.ffmetadata" if archive else None
        chapters = parse_ffmetadata_chapters(cf) if cf and cf.exists() else []
        return gr.update(choices=titles, value=titles[0]), chapters

    archive_dd.change(on_archive, [archive_dd], [chapter_dd, st_chapters])
    demo.load(on_archive, [archive_dd], [chapter_dd, st_chapters])

    # ── Chapter change → frame range ───────────────────────────────────────
    def on_chapter(title, chapters):
        ch = _find_chapter(chapters, title)
        if not ch:
            return gr.update(), gr.update()
        return gr.update(value=ch["start_frame"]), gr.update(value=ch["end_frame"])

    chapter_dd.change(on_chapter, [chapter_dd, st_chapters], [start_n, end_n])

    # ── Threshold mode ─────────────────────────────────────────────────────
    def on_tmode(mode):
        return (gr.update(visible=(mode=="iqr")),
                gr.update(visible=(mode=="value")),
                gr.update(visible=(mode=="quantile")))

    t_mode.change(on_tmode, [t_mode], [iqr_sl, tval_sl, bpct_sl])

    # ── Load chapter ───────────────────────────────────────────────────────
    def on_load(archive, ch_title, chapters, start, end, n_samp,
                wc, wn, wt, ww, tmode, iqrk, tv, bp, cols, tw,
                progress=gr.Progress()):
        FAIL = (gr.update(visible=False), "❌  No chapter/video found.",
                [], [], {}, {}, gr.update(value=[]), "",
                _E_SIG, _E_SIG, _E_SIG, _E_SIG, _E_SCORE)
        if not archive or not ch_title or ch_title.startswith("—"):
            return FAIL
        proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
        mkv   = ARCHIVE_DIR / f"{archive}.mkv"
        video = proxy if proxy.exists() else mkv if mkv.exists() else None
        if not video:
            return FAIL
        fids, b64, sigs, err = extract_frames(
            str(video), int(start), int(end), int(n_samp),
            archive, ch_title, progress=progress,
        )
        if err or fids is None:
            F2 = list(FAIL); F2[1] = f"❌  {err or 'Extraction failed'}"; return tuple(F2)
        # Load overrides from dedicated file, then merge in any recorded in the
        # archive-level badframes.tsv comment lines (from previous Apply runs).
        overrides = load_overrides(archive, ch_title)
        archive_overrides = load_overrides_from_archive_badframes(
            archive=archive, ch_start=int(start), ch_end=int(end),
        )
        if archive_overrides:
            merged = dict(archive_overrides)
            merged.update(overrides)   # dedicated file wins on conflict
            overrides = merged
        html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc = _rebuild(
            fids, b64, sigs, overrides, wc, wn, wt, ww, tmode, iqrk, tv, bp, cols, tw
        )
        return (gr.update(visible=True),
                f"✅  Loaded **{len(fids)}** frames for **{ch_title}**",
                fids, b64, sigs, overrides,
                html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc)

    _LOAD_OUTS = [main_panel, status_md,
                  st_fids, st_b64, st_sigs, st_overrides] + _RB_OUTS
    _LOAD_INS  = [archive_dd, chapter_dd, st_chapters,
                  start_n, end_n, n_sl,
                  wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
                  cols_sl, twidth_sl]
    load_btn.click(on_load,       _LOAD_INS, _LOAD_OUTS)
    reload_btn.click(on_load,     _LOAD_INS, _LOAD_OUTS)
    apply_range_btn.click(on_load, _LOAD_INS, _LOAD_OUTS)

    # ── Live slider updates ────────────────────────────────────────────────
    def on_sliders(fids, b64, sigs, ovr, wc, wn, wt, ww, tm, ik, tv, bp, cols, tw):
        return _rebuild(fids, b64, sigs, ovr, wc, wn, wt, ww, tm, ik, tv, bp, cols, tw)

    _SL_INS = [st_fids, st_b64, st_sigs, st_overrides,
               wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
               cols_sl, twidth_sl]
    for _s in [wc_sl, wn_sl, wt_sl, ww_sl, iqr_sl, tval_sl, bpct_sl, cols_sl, twidth_sl]:
        _s.change(on_sliders, _SL_INS, _RB_OUTS)
    t_mode.change(on_sliders, _SL_INS, _RB_OUTS)

    # ── Frame click toggle ─────────────────────────────────────────────────
    def on_click(raw_click, fids, b64, sigs, overrides, archive, ch_title, ch_start, ch_end,
                 wc, wn, wt, ww, tm, ik, tv, bp, cols, tw):
        if not raw_click or not raw_click.strip() or not fids:
            return (*[gr.update()] * 7, overrides, "", "ignored: empty click payload or no frames")
        try:
            # Value is "fid:timestamp" — timestamp guarantees every click is unique
            fid = int(raw_click.strip().split(":")[0])
        except ValueError:
            return (*[gr.update()] * 7, overrides, "", f"ignored: invalid payload '{raw_click}'")

        before = overrides.get(fid)

        new_ov = toggle_frame_override(
            fid=fid,
            fids=fids,
            sigs=sigs,
            overrides=overrides,
            wc=wc,
            wn=wn,
            wt=wt,
            ww=ww,
            tm=tm,
            ik=ik,
            tv=tv,
            bp=bp,
        )

        ch_title_str = str(ch_title or "")
        if archive and ch_title_str and "select chapter" not in ch_title_str.lower():
            # Persist override file immediately
            save_overrides(archive, ch_title_str, new_ov)
            # Also merge into archive-level badframes.tsv immediately so nothing is lost
            archive_bf = _archive_badframes_path(archive)
            # Retain auto ranges for this chapter from file (strip overrides, reapply below)
            auto_ranges = [(s, e) for s, e, _ in _parse_badframes_tsv(archive_bf)
                           if s >= int(ch_start) and e <= int(ch_end)]
            merge_chapter_into_archive_badframes(
                archive        = archive,
                ch_start       = int(ch_start),
                ch_end         = int(ch_end),
                chapter_ranges = auto_ranges,
                overrides      = new_ov,
            )

        after = new_ov.get(fid)
        in_sample = fid in {int(x) for x in fids}
        srv_dbg = f"payload={raw_click} fid={fid} before={before} after={after} in_sample={in_sample}"
        html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc = _rebuild(
            fids, b64, sigs, new_ov, wc, wn, wt, ww, tm, ik, tv, bp, cols, tw
        )
        return html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, new_ov, "", srv_dbg

    click_recv.input(
        on_click,
        [click_recv, st_fids, st_b64, st_sigs, st_overrides,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl],
        [*_RB_OUTS, st_overrides, click_recv, click_debug_srv],
    )

    def on_gallery_select(fids, b64, sigs, overrides, archive, ch_title, ch_start, ch_end,
                          wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, evt: gr.SelectData):
        if evt is None or getattr(evt, "index", None) is None:
            return (*[gr.update()] * 7, overrides, "", "ignored: no gallery selection event")
        idx = int(evt.index)
        if idx < 0 or idx >= len(fids):
            return (*[gr.update()] * 7, overrides, "", f"ignored: gallery index out of range ({idx})")
        fid = int(fids[idx])
        payload = f"{fid}:{int(idx)}"
        return on_click(
            payload, fids, b64, sigs, overrides, archive, ch_title, ch_start, ch_end,
            wc, wn, wt, ww, tm, ik, tv, bp, cols, tw
        )

    grid_gallery.select(
        on_gallery_select,
        [st_fids, st_b64, st_sigs, st_overrides,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl],
        [*_RB_OUTS, st_overrides, click_recv, click_debug_srv],
    )

    # ── Apply & Regenerate ─────────────────────────────────────────────────
    def on_apply(archive, ch_title, start, end, wc, wn, wt, ww, iqrk, fstep):
        return apply_and_regenerate(
            archive, ch_title, int(start), int(end),
            wc, wn, wt, ww, iqrk, int(fstep),
        )

    apply_btn.click(
        on_apply,
        [archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, iqr_sl, fstep_sl],
        [apply_log],
    )

    # ── Render + preview ───────────────────────────────────────────────────
    def on_render(archive, ch_title, chapters, fids, sigs, overrides,
                  wc, wn, wt, ww, tm, ik, tv, bp, start, end, fstep,
                  progress=gr.Progress()):
        NA = ("❌  No chapter selected.", gr.update(visible=False), gr.update(visible=False))
        ch_title_text = str(ch_title or "").strip().lower()
        if (not archive or not ch_title
                or "select chapter" in ch_title_text
                or "no chapters" in ch_title_text):
            return NA
        if not STEP6.exists():
            return (f"❌  {STEP6} not found.", gr.update(visible=False), gr.update(visible=False))

        # Always regenerate badframes.tsv with current weights before rendering
        progress(0.05, desc="Regenerating badframes.tsv…")
        regen_log = apply_and_regenerate(
            archive, ch_title, int(start), int(end),
            float(wc), float(wn), float(wt), float(ww), float(ik), int(fstep),
        )
        failure_tokens = ["No chapter selected", "tracking_loss failed",
                          "No video found", "module not found"]
        if any(t in regen_log for t in failure_tokens):
            return (f"❌  Failed to regenerate badframes:\n{regen_log[-3000:]}",
                    gr.update(visible=False), gr.update(visible=False))

        progress(0.2, desc="Running step_6_make_videos…")
        archive_bf = _archive_badframes_path(archive)
        result = subprocess.run(
            [sys.executable, str(STEP6),
             "--archive", archive,
             "--title", ch_title,
             "--title-exact",
             "--badframes-tsv", str(archive_bf),
             "--badframes-archive", archive],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        log = (regen_log + "\n\n" + (result.stdout or "") + (result.stderr or ""))[-5000:]
        if result.returncode != 0:
            return (f"❌  step_6 failed (exit {result.returncode}):\n{log}",
                    gr.update(visible=False), gr.update(visible=False))

        rendered = None
        try:
            from common import VIDEOS_DIR, CLIPS_DIR, safe  # type: ignore
            for d in [VIDEOS_DIR, CLIPS_DIR]:
                c = d / f"{safe(ch_title)}.mp4"
                if c.exists() and c.stat().st_size > 100_000:
                    rendered = c; break
        except Exception:
            pass

        if not rendered:
            return (f"✅  Render done; output file not found.\n{log}",
                    gr.update(visible=False), gr.update(visible=False))

        progress(0.65, desc="Building preview with burn-in…")
        ch       = _find_chapter(chapters, ch_title)
        ch_start = ch["start_frame"] if ch else 0
        sc_np    = combined_score(sigs, wc, wn, wt, ww) if sigs else np.array([])
        preview_p = rendered.parent / f"{rendered.stem}_tuner_preview.mp4"
        try:
            make_preview_video(rendered, preview_p, ch_start, fids or [], sc_np)
            vid_path = str(preview_p)
        except Exception as exc:
            vid_path = str(rendered)
            log += f"\n(Burn-in failed: {exc})"

        progress(1.0)
        return (f"✅  {rendered.name}",
                gr.update(visible=True, value=vid_path),
                gr.update(visible=True))

    render_btn.click(
        on_render,
        [archive_dd, chapter_dd, st_chapters,
         st_fids, st_sigs, st_overrides,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         start_n, end_n, fstep_sl],
        [render_log, preview_vid, step_ctrl_html],
    )


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
        css=_DARK_CSS,
        allowed_paths=["C:/Users/covec/Videos/Clips"],
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
