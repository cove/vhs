#!/usr/bin/env python3.11
"""
VHS Bad Frame Tuner — Gradio edition
=====================================
Run:  python vhs_tuner.py

Requires:  pip install gradio opencv-python-headless numpy pillow pandas

Metadata layout (all under metadata/<archive>/)
────────────────────────────────────────────────
  frame_quality.tsv              archive-level canonical frame labels/scores
                                 columns: frame, score, bad_frame, manual_override
  frame_quality_settings.tsv     chapter ranges + tracking-loss parameters used

step_6_make_videos.py reads  metadata/<archive>/frame_quality.tsv
                              which is kept up-to-date by "Apply & Regenerate"
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import time
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

def _archive_frame_quality_path(archive: str) -> Path:
    p = METADATA_DIR / archive / "frame_quality.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _archive_frame_quality_settings_path(archive: str) -> Path:
    p = METADATA_DIR / archive / "frame_quality_settings.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _archive_tracking_tmp_frame_quality_path(archive: str) -> Path:
    p = METADATA_DIR / archive / "_frame_quality_tracking_tmp.tsv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def upsert_frame_quality_settings(
    *,
    archive: str,
    chapter: str,
    start_frame: int,
    end_frame: int,
    weight_chroma: float,
    weight_noise: float,
    weight_tear: float,
    weight_wave: float,
    iqr_mult: float,
    frame_step: int,
    threshold_window_size: int = 1000,
) -> Path:
    def _chapter_key(text: str) -> str:
        # Canonical chapter key so Apply replaces the same chapter row reliably.
        return " ".join(str(text or "").strip().lower().split())

    path = _archive_frame_quality_settings_path(archive)
    rows: list[dict[str, str]] = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split("\t")]
            if parts and parts[0].lower() == "chapter":
                continue
            if len(parts) < 10:
                continue
            rows.append({
                "chapter": parts[0],
                "start_frame": parts[1],
                "end_frame": parts[2],
                "weight_chroma": parts[3],
                "weight_noise": parts[4],
                "weight_tear": parts[5],
                "weight_wave": parts[6],
                "iqr_mult": parts[7],
                "frame_step": parts[8],
                "threshold_window_size": parts[9],
            })

    key = _chapter_key(chapter)
    new_row = {
        "chapter": str(chapter),
        "start_frame": str(int(start_frame)),
        "end_frame": str(int(end_frame)),
        "weight_chroma": f"{float(weight_chroma):.6f}",
        "weight_noise": f"{float(weight_noise):.6f}",
        "weight_tear": f"{float(weight_tear):.6f}",
        "weight_wave": f"{float(weight_wave):.6f}",
        "iqr_mult": f"{float(iqr_mult):.6f}",
        "frame_step": str(int(frame_step)),
        "threshold_window_size": str(int(threshold_window_size)),
    }

    # Drop all prior rows for this chapter and keep only the latest settings.
    deduped = [r for r in rows if _chapter_key(r.get("chapter", "")) != key]
    deduped.append(new_row)

    deduped.sort(key=lambda r: (int(r["start_frame"]), int(r["end_frame"]), r["chapter"]))
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "chapter\tstart_frame\tend_frame\tweight_chroma\tweight_noise\t"
            "weight_tear\tweight_wave\tiqr_mult\tframe_step\tthreshold_window_size\n"
        )
        for r in deduped:
            f.write(
                f"{r['chapter']}\t{r['start_frame']}\t{r['end_frame']}\t"
                f"{r['weight_chroma']}\t{r['weight_noise']}\t{r['weight_tear']}\t"
                f"{r['weight_wave']}\t{r['iqr_mult']}\t{r['frame_step']}\t"
                f"{r['threshold_window_size']}\n"
            )
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# Bad-frame range I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _read_frame_quality_rows(path: Path) -> dict[int, dict[str, float | int]]:
    rows: dict[int, dict[str, float | int]] = {}
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("\t")]
        low = [p.lower() for p in parts]
        if low and low[0] in {"frame", "start_frame"}:
            continue
        if len(parts) < 3:
            continue
        try:
            fid = int(parts[0])
        except Exception:
            continue
        score = np.nan
        try:
            if len(parts) > 1 and parts[1] != "":
                score = float(parts[1])
        except Exception:
            score = np.nan
        try:
            bad_frame = int(parts[2])
        except Exception:
            bad_frame = 0
        manual_override = 0
        if len(parts) > 3:
            try:
                manual_override = int(parts[3])
            except Exception:
                manual_override = 0
        rows[fid] = {
            "score": score,
            "bad_frame": 1 if int(bad_frame) else 0,
            "manual_override": 1 if int(manual_override) else 0,
        }
    return rows

def _write_frame_quality_rows(path: Path, rows: dict[int, dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("frame\tscore\tbad_frame\tmanual_override\n")
        for fid in sorted(rows):
            row = rows[fid]
            sc = float(row.get("score", np.nan))
            sc_txt = "" if not np.isfinite(sc) else f"{sc:.8f}"
            bad = 1 if int(row.get("bad_frame", 0)) else 0
            man = 1 if int(row.get("manual_override", 0)) else 0
            f.write(f"{int(fid)}\t{sc_txt}\t{bad}\t{man}\n")

def load_manual_overrides_from_archive_frame_quality(
    archive: str,
    ch_start: int,
    ch_end: int,
) -> dict[int, str]:
    out: dict[int, str] = {}
    rows = _read_frame_quality_rows(_archive_frame_quality_path(archive))
    for fid, row in rows.items():
        if fid < int(ch_start) or fid > int(ch_end):
            continue
        if int(row.get("manual_override", 0)) != 1:
            continue
        out[int(fid)] = "bad" if int(row.get("bad_frame", 0)) == 1 else "good"
    return out

def _load_tracking_frame_scores(path: Path) -> dict[int, tuple[float, int]]:
    out: dict[int, tuple[float, int]] = {}
    if not path.exists():
        return out
    header: dict[str, int] | None = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("\t")]
        low = [p.lower() for p in parts]
        if header is None and "frame" in low and "score" in low:
            header = {name: idx for idx, name in enumerate(low)}
            continue
        if header is None:
            continue
        try:
            fid = int(parts[header["frame"]])
            score = float(parts[header["score"]])
        except Exception:
            continue
        bad = 0
        if "bad_frame" in header and header["bad_frame"] < len(parts):
            try:
                bad = 1 if int(parts[header["bad_frame"]]) == 1 else 0
            except Exception:
                bad = 0
        elif "label" in header and header["label"] < len(parts):
            label = parts[header["label"]].strip().lower()
            bad = 1 if label == "bad" else 0
        out[fid] = (score, bad)
    return out

def merge_chapter_into_archive_frame_quality(
    archive: str,
    ch_start: int,
    ch_end: int,
    scored_rows: dict[int, tuple[float, int]],
    overrides: dict[int, str],
) -> Path:
    path = _archive_frame_quality_path(archive)
    existing = _read_frame_quality_rows(path)
    merged: dict[int, dict[str, float | int]] = {
        fid: row
        for fid, row in existing.items()
        if fid < int(ch_start) or fid > int(ch_end)
    }
    for fid, (score, bad) in scored_rows.items():
        if fid < int(ch_start) or fid > int(ch_end):
            continue
        merged[int(fid)] = {"score": float(score), "bad_frame": int(bad), "manual_override": 0}

    for fid, label in overrides.items():
        if fid < int(ch_start) or fid > int(ch_end):
            continue
        base = merged.get(int(fid), existing.get(int(fid), {"score": np.nan, "bad_frame": 0, "manual_override": 0}))
        base["bad_frame"] = 1 if label == "bad" else 0
        base["manual_override"] = 1
        merged[int(fid)] = base

    _write_frame_quality_rows(path, merged)
    return path

def upsert_frame_quality_row(
    archive: str,
    frame: int,
    score: float,
    bad_frame: int,
    manual_override: int,
) -> Path:
    path = _archive_frame_quality_path(archive)
    rows = _read_frame_quality_rows(path)
    rows[int(frame)] = {
        "score": float(score) if np.isfinite(score) else np.nan,
        "bad_frame": 1 if int(bad_frame) else 0,
        "manual_override": 1 if int(manual_override) else 0,
    }
    _write_frame_quality_rows(path, rows)
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# Signal cache
# ═══════════════════════════════════════════════════════════════════════════════

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
    progress=None,
) -> tuple[list[int] | None, list[str] | None, dict | None, str]:
    if frame_ids is None:
        frame_ids = np.linspace(int(start), int(end), int(n), dtype=int).tolist()
    else:
        frame_ids = [int(x) for x in frame_ids if int(start) <= int(x) <= int(end)]
        frame_ids = sorted(set(frame_ids))
        if not frame_ids:
            frame_ids = np.linspace(int(start), int(end), int(n), dtype=int).tolist()
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
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
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
    new_ov = toggle_frame_override(
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
    )

    sc = combined_score(sigs, wc, wn, wt, ww)
    thr = compute_threshold(sc, tm, ik, tv, bp)
    idx = fids.index(fid) if fid in fids else -1
    auto_bad = bool(idx >= 0 and sc[idx] >= thr)
    score_val = float(sc[idx]) if idx >= 0 else float("nan")
    manual_state = new_ov.get(fid)
    resolved_bad = (manual_state == "bad") if manual_state else auto_bad
    manual_flag = 1 if manual_state else 0

    persisted = False
    persisted_path = ""
    if archive:
        out_path = upsert_frame_quality_row(
            archive=archive,
            frame=int(fid),
            score=score_val,
            bad_frame=1 if resolved_bad else 0,
            manual_override=manual_flag,
        )
        persisted = True
        persisted_path = str(out_path)

    after = new_ov.get(fid)
    srv_dbg = (
        f"payload={raw_click} fid={fid} ts={ts} before={before} after={after} "
        f"persisted={persisted} path={persisted_path}"
    )
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
    s = int(start)
    e = int(end)
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
        hi = min(e, fid + radius)
        needed = [f for f in range(lo, hi + 1) if f not in selected]
        if len(selected) + len(needed) > budget:
            continue
        selected.update(needed)
        if len(selected) >= budget:
            break

    # Fill remaining budget with uniform samples across the full range.
    if len(selected) < budget:
        fill_n = budget - len(selected)
        baseline = np.linspace(s, e, fill_n, dtype=int).tolist()
        for f in baseline:
            selected.add(int(f))
            if len(selected) >= budget:
                break

    # Final clamp by deterministic order.
    ordered = sorted(selected)
    if len(ordered) > budget:
        ordered = ordered[:budget]
    return ordered

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
# Apply: run tracking_loss + merge overrides into archive frame quality
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
    archive_fq_path = _archive_frame_quality_path(archive)

    # 1 ── Save chapter settings row ────────────────────────────────────────
    settings_tsv = upsert_frame_quality_settings(
        archive=archive,
        chapter=ch_title,
        start_frame=int(ch_start),
        end_frame=int(ch_end),
        weight_chroma=float(w_chroma),
        weight_noise=float(w_noise),
        weight_tear=float(w_tear),
        weight_wave=float(w_wave),
        iqr_mult=float(iqr_mult),
        frame_step=int(frame_step),
        threshold_window_size=1000,
    )
    logs.append(f"✅  Settings → {settings_tsv.name}")

    # Preserve current manual overrides before any re-scoring writes occur.
    preserved_overrides = load_manual_overrides_from_archive_frame_quality(
        archive, int(ch_start), int(ch_end)
    )
    if preserved_overrides:
        manual_bad = [f for f, l in preserved_overrides.items() if l == "bad"]
        manual_good = [f for f, l in preserved_overrides.items() if l == "good"]
        logs.append(
            f"   Preserved overrides: {len(manual_bad)} forced-bad, {len(manual_good)} forced-good"
        )

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
    tracking_tmp_fq = _archive_tracking_tmp_frame_quality_path(archive)
    tracking_tmp_fq.unlink(missing_ok=True)
    config = TrackingLossConfig(  # type: ignore[call-arg]
        archive              = archive,
        video                = video,
        start_frame          = ch_start,
        max_frame            = ch_end,
        frame_step           = max(1, frame_step),
        weight_chroma        = w_chroma,
        weight_noise         = w_noise,
        weight_tear          = w_tear,
        weight_wave          = w_wave,
        iqr_mult             = iqr_mult,
        threshold_window_size= 1000,
        # Write tracking output to a temp TSV; we merge into canonical file after
        # applying preserved manual overrides.
        metadata_frame_quality_tsv = str(tracking_tmp_fq),
    )
    try:
        result = run_tracking_loss_classification(config=config)  # type: ignore
        fq_out = Path(result.get("frame_quality_path", tracking_tmp_fq))
        logs.append(f"✅  tracking_loss done  →  {fq_out.name}")
    except Exception as exc:
        logs.append(f"❌  tracking_loss failed: {exc}")
        return "\n".join(logs)

    # 4 ── Load auto bad ranges ─────────────────────────────────────────────
    scored_rows = _load_tracking_frame_scores(fq_out)
    if not scored_rows:
        logs.append("ERROR: tracking_loss did not produce usable frame scores; frame_quality.tsv was not updated.")
        return "\n".join(logs)
    logs.append(f"   Scored frames: {len(scored_rows)}")

    # 5 ── Merge overrides into archive-level frame_quality.tsv ─────────────────
    overrides = preserved_overrides

    archive_fq = merge_chapter_into_archive_frame_quality(
        archive        = archive,
        ch_start       = ch_start,
        ch_end         = ch_end,
        scored_rows    = scored_rows,
        overrides      = overrides,
    )
    tracking_tmp_fq.unlink(missing_ok=True)
    logs.append(f"OK: Archive frame quality -> {archive_fq}")
    logs.append("   (step_6_make_videos reads frame_quality.tsv)")
    return "\n".join(logs)

# ═══════════════════════════════════════════════════════════════════════════════
# Gradio layout
# ═══════════════════════════════════════════════════════════════════════════════

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
  height: 100vh !important;
  overflow: hidden !important;
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}
#vhs-right-col {
  order: 1 !important;
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
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
#vhs-grid-gallery {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  max-height: none !important;
  height: auto !important;
  overflow: auto !important;
}
#vhs-grid-gallery > div {
  height: 100% !important;
}
#vhs-grid-gallery [class*="grid"] {
  min-height: 100% !important;
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

_E_SIG   = _sparkline_svg(np.array([]), None,  "", height=24)
_E_SCORE = _sparkline_svg(np.array([]), None,  "", height=32)

with gr.Blocks(
    title="📼  VHS Frame Tuner",
) as demo:

    # ── Persistent state ──────────────────────────────────────────────────────
    st_fids      = gr.State([])
    st_b64       = gr.State([])
    st_sigs      = gr.State({})
    st_overrides = gr.State({})
    st_last_click = gr.State({"fid": -1, "ts": -1})
    st_chapters  = gr.State([])

    # ── Archive + Chapter ─────────────────────────────────────────────────────
    with gr.Row() as load_row:
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

    with gr.Row() as load_status_row:
        status_md = gr.Markdown("`Select archive/chapter and load.`")
        show_loader_btn = gr.Button("Loader", variant="secondary", visible=False, elem_id="vhs-loader-toggle")

    # ── Main panel ────────────────────────────────────────────────────────────
    with gr.Column(visible=False, elem_id="vhs-main-panel") as main_panel:

        # ── LEFT column ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=210, elem_id="vhs-left-col"):

            with gr.Accordion("Range & Sample", open=True):
                with gr.Row():
                    start_n = gr.Number(label="Start", value=0, precision=0)
                    end_n   = gr.Number(label="End", value=1000, precision=0)
                    n_sl = gr.Slider(20, 300, value=90, step=10, label="n")
                with gr.Row():
                    apply_range_btn = gr.Button("Apply Range", variant="secondary")
                    reload_btn = gr.Button("Reload", variant="secondary")

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

            with gr.Accordion("Grid + Apply", open=True):
                with gr.Row():
                    cols_sl   = gr.Slider(4, 16, value=7, step=1, label="Cols")
                    twidth_sl = gr.Slider(64, 220, value=120, step=8, label="Width")
                    fstep_sl  = gr.Slider(1, 10, value=1, step=1, label="Step")
                with gr.Row():
                    apply_btn = gr.Button("Apply & Regenerate", variant="primary")
                apply_log = gr.Textbox(label="Apply log", lines=2,
                                        interactive=False, elem_id="vhs-apply-log")

        # ── RIGHT column ──────────────────────────────────────────────────────
        with gr.Column(scale=4, elem_id="vhs-right-col"):

            stats_md  = gr.Markdown("", elem_id="vhs-stats")
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

            # ── Static JS — never included in event outputs, so it persists
            #    across grid rebuilds. We keep inline + delegated click paths
            #    with dedupe so clicks still register if one path is filtered.
            gr.HTML("""
<script>
(function() {
  function _ensureHoverPreview() {
    var root = document.getElementById('vhs-hover-preview');
    if (root) return root;
    root = document.createElement('div');
    root.id = 'vhs-hover-preview';
    var img = document.createElement('img');
    img.alt = 'hover preview';
    root.appendChild(img);
    document.body.appendChild(root);
    return root;
  }

  var _hover = _ensureHoverPreview();
  var _hoverImg = _hover.querySelector('img');
  var _hoverSrc = '';
  function _hideHover() {
    _hover.style.display = 'none';
  }
  function _showHover(src, x, y) {
    if (!src) return;
    if (_hoverSrc !== src) {
      _hoverImg.src = src;
      _hoverSrc = src;
    }
    _hover.style.display = 'block';
    var pad = 18;
    var left = x + pad;
    var top = y + pad;
    var rect = _hover.getBoundingClientRect();
    var maxLeft = Math.max(0, window.innerWidth - rect.width - 6);
    var maxTop = Math.max(0, window.innerHeight - rect.height - 6);
    if (left > maxLeft) left = Math.max(0, x - rect.width - pad);
    if (top > maxTop) top = Math.max(0, y - rect.height - pad);
    _hover.style.left = left + 'px';
    _hover.style.top = top + 'px';
  }

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
      return;
    }
    _lastFid = sid;
    _lastTs = now;
    var payload = sid + ':' + now;
    _setGradio('vhs-click-recv', payload);
  }

  window.vhsToggleFrame = function(fid) { _emit(fid, 'inline'); };
  document.addEventListener('mousemove', function(e) {
    var img = e.target.closest('#vhs-grid-gallery img');
    if (!img) { _hideHover(); return; }
    _showHover(img.currentSrc || img.src, e.clientX, e.clientY);
  }, true);
  // Direct click handling for Gallery tiles so repeated clicks on the same
  // frame still emit toggle events reliably.
  document.addEventListener('click', function(e) {
    var inGallery = e.target.closest('#vhs-grid-gallery');
    if (!inGallery) return;
    var box = e.target.closest('[class*="gallery"], [class*="item"], [class*="thumbnail"], figure, li, div');
    if (!box) return;
    var txt = String((box.textContent || '')).replace(/\\s+/g, ' ').trim();
    var m = txt.match(/#\\s*(\\d+)/);
    if (!m) return;
    _emit(m[1], 'gallery-click');
  }, true);
  document.addEventListener('mouseleave', _hideHover, true);
  document.addEventListener('scroll', _hideHover, true);
  document.addEventListener('click', function(e) {
    var cell = e.target.closest('.vhs-cell[data-fid]');
    if (!cell) return;
    _emit(cell.getAttribute('data-fid'), 'delegate');
  }, true);

})();
</script>
""")

            click_recv = gr.Textbox(
                value="", label="",
                interactive=True, max_lines=1, visible=False,
                elem_id="vhs-click-recv",
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

    def on_show_loader():
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

    show_loader_btn.click(on_show_loader, outputs=[load_row, load_status_row, show_loader_btn])

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
                [], [], {}, {}, {"fid": -1, "ts": -1}, gr.update(value=[]), "",
                _E_SIG, _E_SIG, _E_SIG, _E_SIG, _E_SCORE,
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))
        if not archive or not ch_title or ch_title.startswith("—"):
            return FAIL
        proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
        mkv   = ARCHIVE_DIR / f"{archive}.mkv"
        video = proxy if proxy.exists() else mkv if mkv.exists() else None
        if not video:
            return FAIL
        # Pass 1: uniform sample for coarse bad-frame detection.
        fids, b64, sigs, err = extract_frames(
            str(video), int(start), int(end), int(n_samp),
            archive, ch_title, progress=progress,
        )
        if err or fids is None:
            F2 = list(FAIL); F2[1] = f"❌  {err or 'Extraction failed'}"; return tuple(F2)

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
                F2 = list(FAIL); F2[1] = f"❌  {err or 'Extraction failed'}"; return tuple(F2)
        # Load overrides from dedicated file, then merge in any recorded in the
        # archive-level frame_quality.tsv manual_override rows (from previous Apply runs).
        overrides = {}
        archive_overrides = load_manual_overrides_from_archive_frame_quality(
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
                fids, b64, sigs, overrides, {"fid": -1, "ts": -1},
                html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

    _LOAD_OUTS = [main_panel, status_md,
                  st_fids, st_b64, st_sigs, st_overrides, st_last_click] + _RB_OUTS + [load_row, load_status_row, show_loader_btn]
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
    def on_click(raw_click, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
                 wc, wn, wt, ww, tm, ik, tv, bp, cols, tw):
        if not raw_click or not raw_click.strip() or not fids:
            return (*[gr.update()] * 7, overrides, "", last_click_event)

        new_ov, new_last_click, srv_dbg = apply_manual_click_override(
            raw_click=raw_click,
            fids=fids,
            sigs=sigs,
            overrides=overrides,
            archive=str(archive or ""),
            wc=wc,
            wn=wn,
            wt=wt,
            ww=ww,
            tm=tm,
            ik=ik,
            tv=tv,
            bp=bp,
            last_click_event=last_click_event,
        )
        if str(srv_dbg).startswith("ignored:"):
            return (*[gr.update()] * 7, overrides, "", new_last_click)

        html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc = _rebuild(
            fids, b64, sigs, new_ov, wc, wn, wt, ww, tm, ik, tv, bp, cols, tw
        )
        return html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, new_ov, "", new_last_click

    click_recv.input(
        on_click,
        [click_recv, st_fids, st_b64, st_sigs, st_overrides, st_last_click,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl],
        [*_RB_OUTS, st_overrides, click_recv, st_last_click],
        show_progress="minimal",
    )

    def on_gallery_select(fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
                          wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, evt: gr.SelectData):
        if evt is None or getattr(evt, "index", None) is None:
            return (*[gr.update()] * 7, overrides, "", last_click_event)
        idx = int(evt.index)
        if idx < 0 or idx >= len(fids):
            return (*[gr.update()] * 7, overrides, "", last_click_event)
        fid = int(fids[idx])
        payload = f"{fid}:{int(time.time() * 1000)}"
        return on_click(
            payload, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
            wc, wn, wt, ww, tm, ik, tv, bp, cols, tw
        )

    grid_gallery.select(
        on_gallery_select,
        [st_fids, st_b64, st_sigs, st_overrides, st_last_click,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl],
        [*_RB_OUTS, st_overrides, click_recv, st_last_click],
        show_progress="minimal",
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
