#!/usr/bin/env python3.11
"""
VHS Bad Frame Tuner - Gradio edition
=====================================
Run:  python vhs_tuner.py

Requires:  pip install gradio opencv-python-headless numpy pillow pandas

Metadata layout (all under metadata/<archive>/)
------------------------------------------------
  chapters.ffmetadata           per-chapter BAD_FRAMES=<csv global frame ids>

step_6_make_videos.py reads chapter BAD_FRAMES lists directly from chapters.ffmetadata.
"""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

# -- Project paths -------------------------------------------------------------
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

from common import (
    parse_bad_frames_csv,
    parse_chapters,
    update_chapter_bad_frames_in_ffmetadata,
)

# ===============================================================================
# Chapter / metadata helpers
# ===============================================================================

def parse_ffmetadata_chapters(path: Path) -> list[dict]:
    # Keep chapter frame mapping identical to step_6_make_videos/common.py.
    _ffm, chapters = parse_chapters(Path(path))
    result = []
    for ch in chapters:
        start_sec = float(ch.get("start", 0.0))
        end_sec = float(ch.get("end", 0.0))
        start_frame = int(round(start_sec * FPS))
        end_frame = int(round(end_sec * FPS))
        if end_frame < start_frame:
            end_frame = start_frame
        result.append(
            {
                "title": str(ch.get("title", "Untitled")),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "bad_frames": parse_bad_frames_csv(ch.get("bad_frames", "")),
            }
        )
    return result


def _normalize_frame_span(ch_start: int, ch_end: int) -> tuple[int, int]:
    # Half-open chapter range: [start, end)
    start = int(ch_start)
    end = int(ch_end)
    if end < start:
        start, end = end, start
    if end == start:
        end = start + 1
    return start, end

def slugify(title: str) -> str:
    return re.sub(r"[^\w]+", "_", str(title).strip()).strip("_").lower()

def _chapters_file_path(archive: str) -> Path:
    return METADATA_DIR / str(archive or "").strip() / "chapters.ffmetadata"

def _chapter_bad_overrides(
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
) -> dict[int, str]:
    # Manual overrides are session-only; chapter metadata stores only BAD_FRAMES.
    return {}


def _persist_visible_bad_frames(
    *,
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
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
) -> tuple[Path | None, int]:
    if not archive or not chapter_title or not fids or not sigs:
        return None, 0
    cf = _chapters_file_path(archive)
    if not cf.exists():
        return None, 0
    chapters = parse_ffmetadata_chapters(cf)
    ch = _find_chapter(chapters, chapter_title)
    if not ch:
        return None, 0

    start, end = _normalize_frame_span(ch_start, ch_end)
    existing_global_bad = {
        int(x)
        for x in ch.get("bad_frames", [])
        if start <= int(x) < end
    }

    scores = combined_score(sigs, wc, wn, wt, ww)
    thr = compute_threshold(scores, tm, ik, tv, bp)

    for fid, sc in zip(fids, scores):
        fid_i = int(fid)
        if not (start <= fid_i < end):
            continue
        ov = overrides.get(fid_i)
        if ov == "bad":
            is_bad = True
        elif ov == "good":
            is_bad = False
        else:
            is_bad = bool(float(sc) >= float(thr))
        if is_bad:
            existing_global_bad.add(int(fid_i))
        else:
            existing_global_bad.discard(int(fid_i))

    out_global = sorted(existing_global_bad)
    update_chapter_bad_frames_in_ffmetadata(cf, {str(chapter_title): out_global})
    return cf, len(out_global)


def persist_bad_frames_for_chapter(
    *,
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
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
    progress=None,
) -> tuple[Path | None, int, int, str]:
    _ = progress
    start, end = _normalize_frame_span(ch_start, ch_end)
    sampled_fids = [int(x) for x in (fids or [])]
    sampled_sigs = sigs or {}
    analyzed = len(sampled_fids)
    if analyzed == 0 or not sampled_sigs:
        return None, 0, analyzed, "No sampled frames loaded."

    path, count = _persist_visible_bad_frames(
        archive=str(archive or ""),
        chapter_title=str(chapter_title or ""),
        ch_start=start,
        ch_end=end,
        fids=sampled_fids,
        sigs=sampled_sigs,
        overrides=overrides or {},
        wc=wc,
        wn=wn,
        wt=wt,
        ww=ww,
        tm=tm,
        ik=ik,
        tv=tv,
        bp=bp,
    )
    return path, int(count), analyzed, ""

def load_cached_signals(archive: str, ch_title: str) -> tuple[list[int] | None, dict | None]:
    return None, None

def save_cached_signals(archive: str, ch_title: str,
                        fids: list[int], sigs: dict) -> None:
    return None

def load_overrides(archive: str, ch_title: str) -> dict[int, str]:
    return {}

def save_overrides(archive: str, ch_title: str, overrides: dict[int, str]) -> None:
    return None

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
    frame_ids: list[int] | None = None,
    include_thumbs: bool = True,
    progress=None,
) -> tuple[list[int] | None, list[str] | None, dict | None, str]:
    start_i, end_i = _normalize_frame_span(start, end)
    if frame_ids is None:
        target_n = max(1, min(int(n), max(1, end_i - start_i)))
        frame_ids = np.linspace(start_i, end_i - 1, target_n, dtype=int).tolist()
    else:
        frame_ids = [int(x) for x in frame_ids if start_i <= int(x) < end_i]
        frame_ids = sorted(set(frame_ids))
        if not frame_ids:
            target_n = max(1, min(int(n), max(1, end_i - start_i)))
            frame_ids = np.linspace(start_i, end_i - 1, target_n, dtype=int).tolist()
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
            progress(idx / len(frame_ids), desc=f"Frame {fid}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        if include_thumbs:
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

# ===============================================================================
# Scoring / thresholding
# ===============================================================================

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
    out = dict(overrides)
    # 3-state manual override:
    # - no override + auto-good -> force bad
    # - no override + auto-bad  -> force good
    # - forced bad/good         -> clear override (back to auto)
    if int(fid) not in out:
        scores = combined_score(sigs, wc, wn, wt, ww)
        thr = compute_threshold(scores, tm, ik, tv, bp)
        pos = {int(f): i for i, f in enumerate(fids)}.get(int(fid))
        auto_bad = bool(pos is not None and float(scores[pos]) >= float(thr))
        out[int(fid)] = "good" if auto_bad else "bad"
    else:
        del out[int(fid)]
    return out

def set_frame_override_mode(
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
    mode: str = "toggle",
) -> dict[int, str]:
    out = dict(overrides or {})
    mode_n = str(mode or "toggle").strip().lower()
    if mode_n == "bad":
        out[int(fid)] = "bad"
        return out
    if mode_n == "good":
        out[int(fid)] = "good"
        return out
    if mode_n in {"clear", "auto"}:
        out.pop(int(fid), None)
        return out
    return toggle_frame_override(
        fid=fid,
        fids=fids,
        sigs=sigs,
        overrides=out,
        wc=wc,
        wn=wn,
        wt=wt,
        ww=ww,
        tm=tm,
        ik=ik,
        tv=tv,
        bp=bp,
    )


def _parse_click_payload(raw_click: str) -> tuple[int, int]:
    text = str(raw_click or "").strip()
    if not text:
        raise ValueError("empty click payload")
    parts = text.split(":")
    fid = int(parts[0])
    # Use server receive time for dedupe; client clocks can differ.
    ts = int(time.time() * 1000)
    return fid, ts


def _should_dedupe_click(
    *,
    fid: int,
    ts: int,
    last_click_event: dict | None,
    window_ms: int = 220,
) -> bool:
    if not isinstance(last_click_event, dict):
        return False
    try:
        last_fid = int(last_click_event.get("fid", -1))
        last_ts = int(last_click_event.get("ts", -1))
    except Exception:
        return False
    if fid != last_fid:
        return False
    dt = int(ts) - int(last_ts)
    return 0 <= dt <= max(0, int(window_ms))


def apply_manual_click_override(
    *,
    raw_click: str,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
    mark_mode: str = "toggle",
    last_click_event: dict | None = None,
) -> tuple[dict[int, str], dict[str, int], str]:
    current = dict(overrides or {})
    try:
        fid, ts = _parse_click_payload(raw_click)
    except Exception:
        return current, dict(last_click_event or {}), f"ignored: invalid payload '{raw_click}'"

    if fid not in {int(x) for x in fids}:
        return current, dict(last_click_event or {}), f"ignored: frame {fid} not in sampled set"

    if _should_dedupe_click(fid=fid, ts=ts, last_click_event=last_click_event):
        return current, dict(last_click_event or {}), (
            f"ignored: duplicate click fid={fid} ts={ts}"
        )

    before = current.get(fid)
    new_ov = set_frame_override_mode(
        fid=fid,
        fids=fids,
        sigs=sigs,
        overrides=current,
        wc=wc,
        wn=wn,
        wt=wt,
        ww=ww,
        tm=tm,
        ik=ik,
        tv=tv,
        bp=bp,
        mode=mark_mode,
    )

    after = new_ov.get(fid)
    srv_dbg = (
        f"payload={raw_click} fid={fid} ts={ts} mode={mark_mode} before={before} after={after} "
        "persisted=False (explicit save required)"
    )
    print(f"[vhs_tuner] {srv_dbg}")
    return new_ov, {"fid": int(fid), "ts": int(ts)}, srv_dbg


def select_focus_frame_ids(
    *,
    start: int,
    end: int,
    max_frames: int,
    coarse_fids: list[int],
    coarse_scores: np.ndarray,
    threshold: float,
    burst_radius: int = 4,
) -> list[int]:
    """
    Build a weighted frame list that prioritizes contiguous context around
    detected bad frames while keeping total count <= max_frames.
    """
    budget = max(1, int(max_frames))
    s, e = _normalize_frame_span(start, end)
    radius = max(1, int(burst_radius))

    bad_candidates: list[tuple[int, float]] = []
    for fid, sc in zip(coarse_fids, coarse_scores):
        if float(sc) >= float(threshold):
            bad_candidates.append((int(fid), float(sc)))
    bad_candidates.sort(key=lambda x: x[1], reverse=True)

    selected: set[int] = set()
    # Add full contiguous neighborhoods (no sampling in chosen windows).
    for fid, _sc in bad_candidates:
        lo = max(s, fid - radius)
        hi = min(e - 1, fid + radius)
        needed = [f for f in range(lo, hi + 1) if f not in selected]
        if len(selected) + len(needed) > budget:
            continue
        selected.update(needed)
        if len(selected) >= budget:
            break

    # Fill remaining budget with uniform samples across the full range.
    if len(selected) < budget:
        fill_n = budget - len(selected)
        baseline = np.linspace(s, e - 1, fill_n, dtype=int).tolist()
        for f in baseline:
            selected.add(int(f))
            if len(selected) >= budget:
                break

    # Final clamp by deterministic order.
    ordered = sorted(selected)
    if len(ordered) > budget:
        ordered = ordered[:budget]
    return ordered

# ===============================================================================
# SVG Sparklines  - timeline charts with horizontal red cut line
# ===============================================================================

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
    SVG_W = 200  # viewBox width - browser scales to container

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

# ===============================================================================
# Frame grid HTML
# ===============================================================================

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
    chapter_start_frame: int = 0,
) -> str:
    if not fids:
        return "<p style='color:#666;font-family:monospace;padding:20px'>No frames loaded.</p>"

    cells = []
    for b64, fid, sc in zip(frames_b64, fids, scores):
        local_fid = int(fid) - int(chapter_start_frame)
        ov    = overrides.get(int(fid))
        auto_bad = bool(float(sc) >= float(threshold))
        bad = (ov == "bad") or (ov != "good" and auto_bad)
        color = "#e03030" if bad else "#30c870"
        if ov == "bad":
            badge = " MANUAL_BAD"
        elif ov == "good":
            badge = " MANUAL_GOOD"
        else:
            badge = " AUTO_BAD" if auto_bad else " AUTO_GOOD"
        label = f"#{local_fid} {sc:.2f}{badge}"
        cells.append(
            f'<div class="vhs-cell" data-fid="{fid}" onclick="if(window.vhsToggleFrame){{window.vhsToggleFrame({fid});}} return false;"'
            f' title="local {local_fid} | global {fid} | score {sc:.4f} | click to toggle">'
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
    chapter_start_frame: int,
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
        auto_bad = bool(float(sc) >= float(threshold))
        is_bad = (ov == "bad") or (ov != "good" and auto_bad)
        if ov == "bad":
            state_short = "MB"   # manual bad override
        elif ov == "good":
            state_short = "MG"   # manual good override
        else:
            state_short = "AB" if auto_bad else "AG"
        color = "#e03030" if is_bad else "#30c870"

        # Restore fast visual scanning: colored border per frame state.
        styled = ImageOps.expand(img, border=BORDER, fill=color)
        items.append((styled, f"#{int(fid)}  s={sc:.2f}  {state_short}"))
    return items

# ===============================================================================
# Preview video: ffmpeg burn-in  G/L/S  (global frame, local frame, score)
# ===============================================================================

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

# ===============================================================================
# Apply: run tracking_loss and write BAD_FRAMES into chapters.ffmetadata
# ===============================================================================

def apply_and_regenerate(
    archive: str, ch_title: str,
    ch_start: int, ch_end: int,
    w_chroma: float, w_noise: float, w_tear: float, w_wave: float,
    iqr_mult: float, frame_step: int,
) -> str:
    ch_text = str(ch_title or "").strip().lower()
    if (not archive or not ch_title
            or "select chapter" in ch_text
            or "no chapters" in ch_text):
        return "No chapter selected."

    logs: list[str] = []

    if not _HAS_TRACKING:
        return "\n".join(logs) + "\ntracking_loss module not found."

    proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
    mkv   = ARCHIVE_DIR / f"{archive}.mkv"
    video = str(proxy if proxy.exists() else mkv if mkv.exists() else "")
    if not video:
        return "\n".join(logs) + f"\nNo video found for '{archive}'."

    logs.append(f"tracking_loss frames {ch_start}-{ch_end} step={frame_step}...")
    config = TrackingLossConfig(  # type: ignore[call-arg]
        archive=archive,
        video=video,
        chapters_file=str(_chapters_file_path(archive)),
        start_frame=ch_start,
        max_frame=ch_end,
        frame_step=max(1, frame_step),
        weight_chroma=w_chroma,
        weight_noise=w_noise,
        weight_tear=w_tear,
        weight_wave=w_wave,
        iqr_mult=iqr_mult,
        threshold_window_size=1000,
    )
    try:
        result = run_tracking_loss_classification(config=config)  # type: ignore
        logs.append("tracking_loss wrote BAD_FRAMES in chapters.ffmetadata")
        logs.append(f"Updated chapter blocks: {int(result.get('updated_chapters', 0))}")
    except Exception as exc:
        logs.append(f"tracking_loss failed: {exc}")
        return "\n".join(logs)

    logs.append("step_6_make_videos will read BAD_FRAMES from chapters.ffmetadata")
    return "\n".join(logs)

# ===============================================================================
# Gradio layout
# ===============================================================================

_DARK_CSS = """
html, body {
  margin: 0 !important;
  padding: 0 !important;
  background:#0d0d0d !important;
  height: 100vh !important;
  overflow: hidden !important;
}
body, .gradio-container { background:#0d0d0d !important; color:#ccc; }
/* Full-bleed layout for maximum usable area. */
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  min-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}
.gradio-container .main {
  margin: 0 !important;
  padding: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
}
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .app {
  margin: 0 !important;
  padding: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
}
.gr-row, .gr-form, .gr-group { gap: 3px !important; margin: 0 !important; padding: 0 !important; }
.gr-box, .gr-padded { background:#141414 !important; }
.gr-button-primary { background:#1a6b3a !important; border-color:#27a85a !important; }
.gr-button { background:#222 !important; color:#bbb !important; border-color:#333 !important; }
label { color:#999 !important; font-family:'Courier New',monospace !important; font-size:0.72rem !important; }
input[type=range] { accent-color:#27a85a; }
# Compact common controls to help fit one screen.
.gradio-container input,
.gradio-container textarea,
.gradio-container .gr-button,
.gradio-container .gr-markdown,
.gradio-container .gr-form,
.gradio-container .gr-slider,
.gradio-container .gr-number,
.gradio-container .gr-dropdown {
  font-size: 11px !important;
}
.gradio-container .gr-button {
  min-height: 26px !important;
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}
.gradio-container .gr-slider,
.gradio-container .gr-number,
.gradio-container .gr-dropdown,
.gradio-container .gr-radio {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
.gradio-container .gr-accordion {
  margin: 0 !important;
}
.gradio-container .gr-accordion summary {
  padding: 2px 8px !important;
  min-height: 22px !important;
  font-size: 11px !important;
}
.gradio-container .gr-accordion > div {
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}
.gradio-container .gr-block.gr-box,
.gradio-container .block {
  margin: 0 !important;
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}
.gradio-container .prose,
.gradio-container .gr-markdown {
  margin: 0 !important;
  padding: 0 !important;
}
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p {
  margin-top: 2px !important;
  margin-bottom: 2px !important;
}
#vhs-stats { font-family:'Courier New',monospace; font-size:11px;
             background:#111; padding:5px 8px; border-left:2px solid #27a85a; }
#vhs-apply-log textarea  { font-family:'Courier New',monospace; font-size:11px;
                           background:#0a0a0a !important; color:#9fdfb8 !important; }
#vhs-render-log textarea { font-family:'Courier New',monospace; font-size:11px;
                           background:#0a0a0a !important; color:#9fdfb8 !important; }
#vhs-grid-gallery [class*="caption"] {
  font-family:'Courier New',monospace !important;
  font-size:9px !important;
  text-align:left !important;
  white-space:normal !important;
  overflow-wrap:anywhere !important;
  line-height:1.15 !important;
  padding-left:2px !important;
}
#vhs-hover-preview {
  position: fixed;
  z-index: 9999;
  display: none;
  pointer-events: none;
  border: 2px solid #2a2a2a;
  background: #050505;
  box-shadow: 0 8px 24px rgba(0,0,0,.6);
  border-radius: 3px;
  overflow: hidden;
}
#vhs-hover-preview img {
  display: block;
  max-width: min(70vw, 1280px);
  max-height: min(70vh, 720px);
  width: auto;
  height: auto;
}
#vhs-main-panel {
  display: flex !important;
  flex-direction: column !important;
  height: 100dvh !important;
  overflow: hidden !important;
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}
#vhs-right-col {
  order: 1 !important;
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  align-self: stretch !important;
  height: 100% !important;
  min-height: 0 !important;
  width: 100% !important;
  overflow: hidden !important;
}
#vhs-left-col {
  order: 2 !important;
  flex: 0 0 auto !important;
  width: 100% !important;
  margin-top: auto !important;
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}
#vhs-main-tabs {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 0 !important;
}
#vhs-main-tabs > div {
  min-height: 0 !important;
}
#vhs-main-tabs > .tab-wrapper {
  flex: 0 0 auto !important;
}
#vhs-main-tabs [role="tablist"] {
  flex: 0 0 auto !important;
}
#vhs-main-tabs > .tabitem[style*="display: none"],
#vhs-main-tabs > [role="tabpanel"][style*="display: none"] {
  display: none !important;
}
#vhs-main-tabs > .tabitem:not([style*="display: none"]),
#vhs-main-tabs > [role="tabpanel"]:not([style*="display: none"]) {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-main-tabs [role="tabpanel"] {
  min-height: 0 !important;
  overflow: hidden !important;
}
/* Gradio inserts one or two nested .column wrappers inside the active tab panel. */
#vhs-main-tabs > .tabitem:not([style*="display: none"]) > .column,
#vhs-main-tabs > [role="tabpanel"]:not([style*="display: none"]) > .column {
  flex-direction: column !important;
  display: flex !important;
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-main-tabs > .tabitem:not([style*="display: none"]) > .column > .column,
#vhs-main-tabs > [role="tabpanel"]:not([style*="display: none"]) > .column > .column {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
}
#vhs-main-tabs [role="tabpanel"] > .column {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 0 !important;
}
#vhs-stats {
  flex: 0 0 auto !important;
}
#vhs-grid-gallery {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  width: 100% !important;
  min-height: 0 !important;
  max-height: none !important;
  height: 100% !important;
  overflow: hidden !important;
}
#vhs-grid-gallery > div {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-grid-gallery .gallery-container {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-grid-gallery .grid-wrap,
#vhs-grid-gallery .grid-wrap.fixed-height {
  display: block !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: 100% !important;
  max-height: none !important;
  overflow: auto !important;
}
#vhs-grid-gallery .grid-container {
  min-height: 0 !important;
  align-content: start !important;
}
#vhs-grid-gallery [class*="gallery"] {
  min-height: 0 !important;
}
#vhs-grid-gallery [class*="grid"] {
  min-height: 0 !important;
}
/* Reduce distracting gallery redraw effects on click. */
#vhs-grid-gallery,
#vhs-grid-gallery *,
#vhs-grid-gallery [class*="image"],
#vhs-grid-gallery [class*="gallery"] {
  transition: none !important;
  animation: none !important;
}
#vhs-loader-toggle button {
  min-height: 20px !important;
  padding: 1px 6px !important;
  font-size: 10px !important;
}
#vhs-runtime-js {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
"""

def _get_archives() -> list[str]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(p.stem for p in ARCHIVE_DIR.glob("*.mkv"))


def _resolve_archive_video(archive: str) -> Path | None:
    proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
    mkv = ARCHIVE_DIR / f"{archive}.mkv"
    return mkv if mkv.exists() else proxy if proxy.exists() else None

CHAPTER_SELECT_LABEL = "-- select chapter --"
CHAPTER_MISSING_LABEL = "-- no chapters file found --"

def _get_chapter_titles(archive: str) -> list[str]:
    if not archive:
        return [CHAPTER_SELECT_LABEL]
    cf = METADATA_DIR / archive / "chapters.ffmetadata"
    if not cf.exists():
        return [CHAPTER_MISSING_LABEL]
    chapters = parse_ffmetadata_chapters(cf)
    return [CHAPTER_SELECT_LABEL] + [ch["title"] for ch in chapters]

def _find_chapter(chapters: list[dict], title: str) -> dict | None:
    return next((c for c in chapters if c["title"] == title), None)

_E_SIG   = _sparkline_svg(np.array([]), None,  "", height=24)
_E_SCORE = _sparkline_svg(np.array([]), None,  "", height=32)

with gr.Blocks(
    title="  VHS Frame Tuner",
) as demo:

    # -- Persistent state ------------------------------------------------------
    st_fids      = gr.State([])
    st_b64       = gr.State([])
    st_sigs      = gr.State({})
    st_overrides = gr.State({})
    st_visible_fids = gr.State([])
    st_last_click = gr.State({"fid": -1, "ts": -1})
    st_chapters  = gr.State([])

    # -- Archive + Chapter -----------------------------------------------------
    with gr.Row() as load_row:
        _archives  = _get_archives()
        archive_dd = gr.Dropdown(
            choices=_archives, value=_archives[0] if _archives else None,
            label="Archive", scale=1, interactive=True,
        )
        chapter_dd = gr.Dropdown(
            choices=[CHAPTER_SELECT_LABEL], value=CHAPTER_SELECT_LABEL,
            label="Chapter", scale=3, interactive=True,
        )
        load_btn = gr.Button("  Load Chapter", variant="primary", scale=1)

    with gr.Row() as load_status_row:
        status_md = gr.Markdown("`Select archive/chapter and load.`")
        show_loader_btn = gr.Button("Loader", variant="secondary", visible=False, elem_id="vhs-loader-toggle")

    # -- Main panel ------------------------------------------------------------
    with gr.Column(visible=False, elem_id="vhs-main-panel") as main_panel:

        with gr.Tabs(elem_id="vhs-main-tabs", selected="frames-tab"):
            with gr.Tab("Frames", id="frames-tab"):
                with gr.Column(scale=4, elem_id="vhs-right-col"):
                    stats_md  = gr.Markdown("", elem_id="vhs-stats")
                    with gr.Row():
                        apply_btn = gr.Button("Apply", variant="primary")
                    grid_gallery = gr.Gallery(
                        value=[],
                        label="Frames (click a tile to toggle good/bad)",
                        show_label=True,
                        columns=7,
                        object_fit="contain",
                        height="auto",
                        allow_preview=False,
                        elem_id="vhs-grid-gallery",
                    )

            with gr.Tab("Chapters", id="chapters-tab"):
                chapter_list = gr.Radio(
                    choices=[],
                    value=None,
                    label="Chapters",
                    interactive=True,
                    elem_id="vhs-chapter-list",
                )
                chapters_load_btn = gr.Button("Load Selected Chapter", variant="primary")

            with gr.Tab("Tuning", id="tuning-tab"):
                with gr.Column(scale=1, min_width=210, elem_id="vhs-left-col"):
                    with gr.Accordion("Range & Sample", open=False):
                        with gr.Row():
                            start_n = gr.Number(label="Start", value=0, precision=0, elem_id="vhs-start-frame")
                            end_n   = gr.Number(label="End (exclusive)", value=10000, precision=0)
                            n_sl = gr.Slider(20, 10000, value=400, step=10, label="n")
                            context_sl = gr.Slider(0, 200, value=10, step=1, label="Frames Around Bad")
                        with gr.Row():
                            strict_sampling_cb = gr.Checkbox(label="Strict Sampling", value=True)
                            apply_range_btn = gr.Button("Apply Range", variant="secondary")
                            reload_btn = gr.Button("Reload", variant="secondary")
                    with gr.Accordion("Manual Marking", open=False):
                        mark_mode_dd = gr.Dropdown(
                            choices=["toggle", "bad", "good", "clear"],
                            value="toggle",
                            label="Click Action",
                            interactive=True,
                        )

                    with gr.Accordion("Signal Weights", open=False):
                        wc_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="chroma")
                        spark_chroma = gr.HTML(_E_SIG)
                        wn_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="noise")
                        spark_noise  = gr.HTML(_E_SIG)
                        wt_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="tear")
                        spark_tear   = gr.HTML(_E_SIG)
                        ww_sl        = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="wave")
                        spark_wave   = gr.HTML(_E_SIG)

                    with gr.Accordion("Threshold", open=False):
                        t_mode  = gr.Radio(["iqr", "value", "quantile"], value="iqr",
                                            label="Mode", interactive=True)
                        iqr_sl  = gr.Slider(1.0, 8.0, value=3.5, step=0.05, label="k")
                        tval_sl = gr.Slider(-5.0, 15.0, value=1.0, step=0.05,
                                             label="Hard value", visible=False)
                        bpct_sl = gr.Slider(1, 60, value=10, step=1,
                                             label="Bad %", visible=False)
                        spark_score = gr.HTML(_E_SCORE)

                    with gr.Accordion("Grid", open=False):
                        with gr.Row():
                            cols_sl   = gr.Slider(4, 16, value=7, step=1, label="Cols")
                            twidth_sl = gr.Slider(64, 220, value=120, step=8, label="Width")

        # Keep the runtime JS component out of layout flow to avoid panel overlap.
        gr.HTML("", elem_id="vhs-runtime-js", visible=False)

        click_recv = gr.Textbox(
            value="", label="",
            interactive=True, max_lines=1, visible=False,
            elem_id="vhs-click-recv",
        )

    # =========================================================================
    # Rebuild helper - grid + stats + 5 sparklines
    # =========================================================================

    def _frame_is_bad(fid, score, threshold, overrides):
        ov = (overrides or {}).get(int(fid))
        if ov == "bad":
            return True
        if ov == "good":
            return False
        return bool(float(score) >= float(threshold))

    def _select_visible_indices(fids, bad_fids, context):
        if not fids or not bad_fids:
            return []
        ctx = max(0, int(context))
        if ctx <= 0:
            bad_set = {int(f) for f in bad_fids}
            return [i for i, fid in enumerate(fids) if int(fid) in bad_set]
        spans = sorted((int(fid) - ctx, int(fid) + ctx) for fid in bad_fids)
        merged = []
        for lo, hi in spans:
            if not merged or lo > merged[-1][1] + 1:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(merged[-1][1], hi)
        vis = []
        j = 0
        for i, fid in enumerate(fids):
            x = int(fid)
            while j < len(merged) and x > merged[j][1]:
                j += 1
            if j >= len(merged):
                break
            if merged[j][0] <= x <= merged[j][1]:
                vis.append(i)
        return vis

    def _rebuild(fids, b64, sigs, overrides, wc, wn, wt, ww,
                 t_mode, iqr_k, tval, bpct, cols, twidth, context, ch_start):
        if not fids or not b64:
            return (gr.update(value=[]), "*(no frames loaded)*",
                    _E_SIG, _E_SIG, _E_SIG, _E_SIG, _E_SCORE, [])
        sc = combined_score(sigs, wc, wn, wt, ww)
        thr = compute_threshold(sc, t_mode, iqr_k, tval, bpct)
        bad_fids = [
            int(fid)
            for fid, s in zip(fids, sc)
            if _frame_is_bad(fid, s, thr, overrides)
        ]
        vis_idx = _select_visible_indices(fids, bad_fids, context)
        vis_fids = [int(fids[i]) for i in vis_idx]
        vis_b64 = [b64[i] for i in vis_idx]
        vis_sc = np.array([sc[i] for i in vis_idx], dtype=np.float64)
        gallery_items = build_gallery_items(
            vis_b64, vis_fids, vis_sc, overrides, thr, chapter_start_frame=int(ch_start)
        )
        gallery_update = gr.update(value=gallery_items, columns=int(cols))
        n_bad = sum(_frame_is_bad(f, s, thr, overrides) for f, s in zip(fids, sc))
        n_ov = sum(1 for f in fids if int(f) in (overrides or {}))
        stats = (
            f" **Bad:** {n_bad} ({100*n_bad/max(1,len(fids)):.0f}%) | "
            f" **Good:** {len(fids)-n_bad} | "
            f"**Threshold:** {thr:.3f} | "
            f" **Overrides:** {n_ov} | n={len(fids)} | shown={len(vis_fids)}"
        )
        sc_ch, sc_no, sc_te, sc_wa, sc_sc = build_sparklines_html(
            sigs, sc, thr, wc, wn, wt, ww
        )
        return gallery_update, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids

    _RB_OUTS = [grid_gallery, stats_md,
                spark_chroma, spark_noise, spark_tear, spark_wave, spark_score, st_visible_fids]

    # -- Archive change -----------------------------------------------------
    def on_archive(archive):
        titles   = _get_chapter_titles(archive)
        cf       = METADATA_DIR / archive / "chapters.ffmetadata" if archive else None
        chapters = parse_ffmetadata_chapters(cf) if cf and cf.exists() else []
        chapter_titles = [str(ch.get("title", "")) for ch in chapters if str(ch.get("title", ""))]
        chapter_value = chapter_titles[0] if chapter_titles else None
        return (
            gr.update(choices=titles, value=(chapter_value or titles[0])),
            chapters,
            gr.update(choices=chapter_titles, value=chapter_value),
        )

    archive_dd.change(on_archive, [archive_dd], [chapter_dd, st_chapters, chapter_list])
    demo.load(on_archive, [archive_dd], [chapter_dd, st_chapters, chapter_list])

    # -- Chapter change -> frame range ---------------------------------------
    def on_chapter(title, chapters):
        ch = _find_chapter(chapters, title)
        if not ch:
            return gr.update(), gr.update()
        return gr.update(value=ch["start_frame"]), gr.update(value=ch["end_frame"])

    chapter_dd.change(on_chapter, [chapter_dd, st_chapters], [start_n, end_n])

    def on_chapter_list_pick(title):
        if not title:
            return gr.update()
        return gr.update(value=title)

    chapter_list.change(on_chapter_list_pick, [chapter_list], [chapter_dd]).then(
        on_chapter, [chapter_dd, st_chapters], [start_n, end_n]
    )

    def on_show_loader():
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

    show_loader_btn.click(on_show_loader, outputs=[load_row, load_status_row, show_loader_btn])

    # -- Threshold mode -----------------------------------------------------
    def on_tmode(mode):
        return (gr.update(visible=(mode=="iqr")),
                gr.update(visible=(mode=="value")),
                gr.update(visible=(mode=="quantile")))

    t_mode.change(on_tmode, [t_mode], [iqr_sl, tval_sl, bpct_sl])

    # -- Load chapter -------------------------------------------------------
    def on_load(archive, ch_title, chapters, start, end, n_samp, strict_sampling,
                wc, wn, wt, ww, tmode, iqrk, tv, bp, cols, tw, context,
                progress=gr.Progress()):
        FAIL = (gr.update(visible=False), "ERROR:  No chapter/video found.",
                [], [], {}, {}, {"fid": -1, "ts": -1}, gr.update(value=[]), "",
                _E_SIG, _E_SIG, _E_SIG, _E_SIG, _E_SCORE, [],
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))
        if not archive or not ch_title or ch_title in {CHAPTER_SELECT_LABEL, CHAPTER_MISSING_LABEL}:
            return FAIL
        video = _resolve_archive_video(str(archive or ""))
        if not video:
            return FAIL
        # Pass 1: uniform sample for coarse bad-frame detection.
        fids, b64, sigs, err = extract_frames(
            str(video), int(start), int(end), int(n_samp),
            archive, ch_title, progress=progress,
        )
        if err or fids is None:
            F2 = list(FAIL); F2[1] = f"ERROR:  {err or 'Extraction failed'}"; return tuple(F2)

        if not bool(strict_sampling):
            # Pass 2: if coarse sample detects bad frames, prioritize contiguous
            # neighbors around them (no sampling inside those local windows).
            sc0 = combined_score(sigs, wc, wn, wt, ww)
            thr0 = compute_threshold(sc0, tmode, iqrk, tv, bp)
            focus_fids = select_focus_frame_ids(
                start=int(start),
                end=int(end),
                max_frames=int(n_samp),
                coarse_fids=fids,
                coarse_scores=sc0,
                threshold=thr0,
                burst_radius=4,
            )
            if focus_fids != fids:
                fids, b64, sigs, err = extract_frames(
                    str(video), int(start), int(end), int(n_samp),
                    archive, ch_title, frame_ids=focus_fids, progress=progress,
                )
                if err or fids is None:
                    F2 = list(FAIL); F2[1] = f"ERROR:  {err or 'Extraction failed'}"; return tuple(F2)
        # Seed overrides from chapter BAD_FRAMES in chapters.ffmetadata.
        overrides = _chapter_bad_overrides(
            archive=archive,
            chapter_title=ch_title,
            ch_start=int(start),
            ch_end=int(end),
        )
        html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids = _rebuild(
            fids, b64, sigs, overrides, wc, wn, wt, ww, tmode, iqrk, tv, bp, cols, tw, context, int(start)
        )
        return (gr.update(visible=True),
                f"OK:  Loaded **{len(fids)}** sampled frames for **{ch_title}**. Click `Apply` to write chapter metadata.",
                fids, b64, sigs, overrides, {"fid": -1, "ts": -1},
                html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

    _LOAD_OUTS = [main_panel, status_md,
                  st_fids, st_b64, st_sigs, st_overrides, st_last_click] + _RB_OUTS + [load_row, load_status_row, show_loader_btn]
    _LOAD_INS  = [archive_dd, chapter_dd, st_chapters,
                  start_n, end_n, n_sl, strict_sampling_cb,
                  wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
                  cols_sl, twidth_sl, context_sl]
    load_btn.click(on_load,       _LOAD_INS, _LOAD_OUTS)
    chapters_load_btn.click(on_load, _LOAD_INS, _LOAD_OUTS)
    reload_btn.click(on_load,     _LOAD_INS, _LOAD_OUTS)
    apply_range_btn.click(on_load, _LOAD_INS, _LOAD_OUTS)

    def on_save_bad_frames(
        archive,
        ch_title,
        ch_start,
        ch_end,
        fids,
        sigs,
        overrides,
        wc,
        wn,
        wt,
        ww,
        tm,
        ik,
        tv,
        bp,
        progress=gr.Progress(),
    ):
        ch_text = str(ch_title or "").strip().lower()
        if (not archive or not ch_title
                or "select chapter" in ch_text
                or "no chapters" in ch_text):
            return "ERROR:  No chapter selected."
        path, count, analyzed, err = persist_bad_frames_for_chapter(
            archive=str(archive or ""),
            chapter_title=str(ch_title or ""),
            ch_start=int(ch_start),
            ch_end=int(ch_end),
            fids=[int(x) for x in (fids or [])],
            sigs=sigs or {},
            overrides=overrides or {},
            wc=wc,
            wn=wn,
            wt=wt,
            ww=ww,
            tm=tm,
            ik=ik,
            tv=tv,
            bp=bp,
            progress=progress,
        )
        if err:
            return f"ERROR:  {err}"
        return (
            f"Saved:  Saved BAD_FRAMES for **{ch_title}** from loaded frame set "
            f"({analyzed} frame(s) analyzed, {count} marked bad)."
        )

    _SAVE_INS = [
        archive_dd,
        chapter_dd,
        start_n,
        end_n,
        st_fids,
        st_sigs,
        st_overrides,
        wc_sl,
        wn_sl,
        wt_sl,
        ww_sl,
        t_mode,
        iqr_sl,
        tval_sl,
        bpct_sl,
    ]
    apply_btn.click(on_save_bad_frames, _SAVE_INS, [status_md])

    # -- Live slider updates ------------------------------------------------
    def on_sliders(ch_start, fids, b64, sigs, ovr,
                   wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context):
        out = _rebuild(
            fids, b64, sigs, ovr, wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, int(ch_start)
        )
        return out

    _SL_INS = [start_n, st_fids, st_b64, st_sigs, st_overrides,
               wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
               cols_sl, twidth_sl, context_sl]
    for _s in [wc_sl, wn_sl, wt_sl, ww_sl, iqr_sl, tval_sl, bpct_sl, cols_sl, twidth_sl, context_sl]:
        _s.change(on_sliders, _SL_INS, _RB_OUTS)
    t_mode.change(on_sliders, _SL_INS, _RB_OUTS)

    # -- Frame click toggle -------------------------------------------------
    def on_click(raw_click, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
                 wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, mark_mode):
        if not raw_click or not raw_click.strip() or not fids:
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", last_click_event)

        new_ov, new_last_click, srv_dbg = apply_manual_click_override(
            raw_click=raw_click,
            fids=fids,
            sigs=sigs,
            overrides=overrides,
            archive=str(archive or ""),
            chapter_title=str(ch_title or ""),
            ch_start=int(ch_start),
            ch_end=int(ch_end),
            wc=wc,
            wn=wn,
            wt=wt,
            ww=ww,
            tm=tm,
            ik=ik,
            tv=tv,
            bp=bp,
            mark_mode=mark_mode,
            last_click_event=last_click_event,
        )
        if str(srv_dbg).startswith("ignored:"):
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", new_last_click)

        html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids = _rebuild(
            fids, b64, sigs, new_ov, wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, int(ch_start)
        )
        return html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids, new_ov, "", new_last_click

    click_recv.input(
        on_click,
        [click_recv, st_fids, st_b64, st_sigs, st_overrides, st_last_click,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl, context_sl, mark_mode_dd],
        [*_RB_OUTS, st_overrides, click_recv, st_last_click],
        show_progress="minimal",
    )

    def on_gallery_select(vis_fids, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
                          wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, mark_mode, evt: gr.SelectData):
        if evt is None or getattr(evt, "index", None) is None:
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", last_click_event)
        idx = int(evt.index)
        if idx < 0 or idx >= len(vis_fids):
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", last_click_event)
        fid = int(vis_fids[idx])
        payload = f"{fid}:{int(time.time() * 1000)}"
        return on_click(
            payload, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
            wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, mark_mode
        )

    grid_gallery.select(
        on_gallery_select,
        [st_visible_fids, st_fids, st_b64, st_sigs, st_overrides, st_last_click,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl, context_sl, mark_mode_dd],
        [*_RB_OUTS, st_overrides, click_recv, st_last_click],
        show_progress="minimal",
    )

# -- Launch --------------------------------------------------------------------
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





