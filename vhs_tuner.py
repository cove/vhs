#!/usr/bin/env python3
"""
VHS Bad Frame Tuner
===================
Run: streamlit run vhs_tuner.py  (from the project root or scripts/ dir)

Features:
- Chapter dropdown auto-populates frame range
- Per-chapter score caching under metadata/<archive>/tracking_badframe/
- Manual good/bad frame overrides with persistent TSV storage
- Run step_6_make_videos for a single chapter with spinner + log output
- Video preview after render
- Hover-to-enlarge frame grid (popup follows cursor)
"""

import base64
import io
import json
import math
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Project paths ─────────────────────────────────────────────────────────────
# Works whether the tuner lives at project root or in scripts/
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "scripts" else _HERE
ARCHIVE_DIR  = PROJECT_ROOT / "../Archive"
METADATA_DIR = PROJECT_ROOT / "metadata"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
STEP6        = PROJECT_ROOT / "step_6_make_videos.py"

FPS    = 30000 / 1001   # archive cadence
BORDER = 4

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VHS Tuner", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .stApp { background: #111; color: #ddd; }
  section[data-testid="stSidebar"] { background: #181818; }
  h1, h2, h3 { font-family: monospace; }
  .stSlider label, .stRadio label, .stCheckbox label, .stSelectbox label {
    font-family: monospace; font-size: 0.8rem; color: #aaa;
  }
</style>
""", unsafe_allow_html=True)
st.title("📼  VHS Frame Tuner")

# Discrete render button at top
_top_col, _ = st.columns([2, 8])
_render_top = _top_col.button(
    "▶️ Render chapter",
    key="render_top",
    use_container_width=True,
    help="Run step_6_make_videos.py for the selected chapter",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Chapter parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ffmetadata_chapters(path):
    """Parse chapters.ffmetadata → list of dicts with title, start/end sec+frame."""
    chapters, current = [], {}
    default_tb = 1.0 / 1_000_000_000

    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line == "[CHAPTER]":
            if "start_raw" in current:
                chapters.append(current)
            current = {"_tb": default_tb}
            continue
        if "=" not in line or line.startswith(";"):
            continue
        key, _, val = line.partition("=")
        key = key.strip().upper()
        val = val.strip()
        if key == "TIMEBASE":
            try:
                n, d = val.split("/")
                current["_tb"] = float(n) / float(d)
            except Exception:
                pass
        elif key == "START":
            try:
                current["start_raw"] = int(val)
            except Exception:
                pass
        elif key == "END":
            try:
                current["end_raw"] = int(val)
            except Exception:
                pass
        elif key == "TITLE":
            current["title"] = val

    if "start_raw" in current:
        chapters.append(current)

    result = []
    for ch in chapters:
        tb      = ch.get("_tb", default_tb)
        s_sec   = ch.get("start_raw", 0) * tb
        e_sec   = ch.get("end_raw",   0) * tb
        result.append({
            "title":       ch.get("title", "Untitled"),
            "start_sec":   s_sec,
            "end_sec":     e_sec,
            "start_frame": int(round(s_sec * FPS)),
            "end_frame":   int(round(e_sec * FPS)),
            "duration":    e_sec - s_sec,
        })
    return result


def slugify(title):
    return re.sub(r"[^\w]+", "_", str(title).strip()).strip("_").lower()


def _sparkline_svg(values, marker=None, vmin=None, vmax=None, width=120, height=24):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return ""
    lo = float(np.min(vals) if vmin is None else vmin)
    hi = float(np.max(vals) if vmax is None else vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    sorted_vals = np.sort(vals)
    n_points = max(8, int(width))
    points = []
    for i in range(n_points):
        q = i / max(1, n_points - 1)
        idx = int(round(q * (sorted_vals.size - 1)))
        val = float(sorted_vals[idx])
        x = q * (width - 1)
        y = (1.0 - ((val - lo) / max(1e-12, (hi - lo)))) * (height - 1)
        points.append(f"{x:.2f},{y:.2f}")
    poly = " ".join(points)

    marker_svg = ""
    if marker is not None and np.isfinite(float(marker)):
        my = (float(marker) - lo) / max(1e-12, (hi - lo))
        my = min(1.0, max(0.0, my))
        marker_y = (1.0 - my) * (height - 1)
        marker_svg = (
            f"<line x1='0' y1='{marker_y:.2f}' x2='{width - 1}' y2='{marker_y:.2f}' "
            "stroke='#e03030' stroke-width='2'/>"
        )

    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        "style='display:block;background:#141414;border:1px solid #2b2b2b;border-radius:2px'>"
        f"<polyline fill='none' stroke='#7f9dbd' stroke-width='1.6' points='{poly}'/>"
        f"{marker_svg}"
        "</svg>"
    )


def _normalize_to_unit(values):
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return vals
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals, dtype=np.float64)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(vals, dtype=np.float64)
    out = (vals - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Per-chapter score / override paths
# ═══════════════════════════════════════════════════════════════════════════════

def chapter_tracking_dir(archive_name, ch_title):
    d = METADATA_DIR / archive_name / "tracking_badframe"
    d.mkdir(parents=True, exist_ok=True)
    return d

def scores_path(archive_name, ch_title):
    return chapter_tracking_dir(archive_name, ch_title) / f"{slugify(ch_title)}_frame_scores.tsv"

def overrides_path(archive_name, ch_title):
    return chapter_tracking_dir(archive_name, ch_title) / f"{slugify(ch_title)}_overrides.tsv"


def chapter_badframes_path(archive_name, ch_title):
    return chapter_tracking_dir(archive_name, ch_title) / f"{slugify(ch_title)}_badframes.tsv"

def chapter_settings_path(archive_name, ch_title):
    return chapter_tracking_dir(archive_name, ch_title) / f"{slugify(ch_title)}_settings.json"


def load_cached_scores(archive_name, ch_title):
    """Return (fids, signals_dict) from cached TSV or None if not found/stale."""
    p = scores_path(archive_name, ch_title)
    if not p.exists():
        return None, None
    try:
        df = pd.read_csv(p, sep="\t", comment="#")
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"frame", "chroma_loss", "noise_energy", "row_tear", "wave_energy"}
        if not required.issubset(set(df.columns)):
            return None, None
        fids = df["frame"].astype(int).tolist()
        sigs = {
            "chroma": df["chroma_loss"].values.astype(np.float64),
            "noise":  df["noise_energy"].values.astype(np.float64),
            "tear":   df["row_tear"].values.astype(np.float64),
            "wave":   df["wave_energy"].values.astype(np.float64),
        }
        return fids, sigs
    except Exception:
        return None, None


def save_scores(archive_name, ch_title, fids, sigs):
    p = scores_path(archive_name, ch_title)
    with p.open("w", encoding="utf-8") as f:
        f.write("frame\tchroma_loss\tnoise_energy\trow_tear\twave_energy\n")
        for i, fid in enumerate(fids):
            f.write(f"{fid}\t{sigs['chroma'][i]:.8f}\t{sigs['noise'][i]:.8f}"
                    f"\t{sigs['tear'][i]:.8f}\t{sigs['wave'][i]:.8f}\n")


def load_overrides(archive_name, ch_title):
    """Return dict {frame_id: 'good'|'bad'}."""
    p = overrides_path(archive_name, ch_title)
    if not p.exists():
        return {}
    out = {}
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


def save_overrides(archive_name, ch_title, overrides_dict):
    p = overrides_path(archive_name, ch_title)
    lines = ["frame\toverride"]
    for fid in sorted(overrides_dict.keys()):
        lines.append(f"{fid}\t{overrides_dict[fid]}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ranges_from_sorted_frames(frame_ids):
    if not frame_ids:
        return []
    frames = sorted({int(x) for x in frame_ids})
    ranges = []
    start = prev = frames[0]
    for fid in frames[1:]:
        if fid == prev + 1:
            prev = fid
            continue
        ranges.append((start, prev))
        start = prev = fid
    ranges.append((start, prev))
    return ranges


def save_chapter_badframes_tsv(archive_name, ch_title, frame_ids, is_bad_flags):
    p = chapter_badframes_path(archive_name, ch_title)
    bad_frames = [int(fid) for fid, bad in zip(frame_ids, is_bad_flags) if bool(bad)]
    ranges = ranges_from_sorted_frames(bad_frames)
    with p.open("w", encoding="utf-8") as f:
        f.write("start_frame\tend_frame\tnote\n")
        for start, end in ranges:
            f.write(f"{int(start)}\t{int(end)}\tvhs_tuner_chapter\n")
    return p, len(bad_frames), len(ranges)


def load_badframe_ranges_tsv(tsv_path):
    if not tsv_path:
        return []
    p = Path(tsv_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = [x.strip() for x in s.split("\t")]
        if len(parts) < 2:
            continue
        if parts[0].lower().startswith("start"):
            continue
        try:
            a = int(parts[0])
            b = int(parts[1])
        except Exception:
            continue
        if b < a:
            a, b = b, a
        out.append((a, b))
    return out


def save_chapter_settings_json(archive_name, ch_title, payload):
    p = chapter_settings_path(archive_name, ch_title)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Compact CSS — tighter labels, less padding between controls
    st.markdown("""
    <style>
      section[data-testid="stSidebar"] { font-size: 0.78rem; }
      section[data-testid="stSidebar"] .stSlider       { margin-bottom: -10px; }
      section[data-testid="stSidebar"] .stSlider label { font-size: 0.72rem; color: #999; }
      section[data-testid="stSidebar"] .stSelectbox    { margin-bottom: -6px; }
      section[data-testid="stSidebar"] .stTextInput    { margin-bottom: -6px; }
      section[data-testid="stSidebar"] .stRadio        { margin-bottom: -4px; }
      section[data-testid="stSidebar"] .stCheckbox     { margin-bottom: -8px; }
      section[data-testid="stSidebar"] .stNumberInput  { margin-bottom: -6px; }
      section[data-testid="stSidebar"] hr              { margin: 6px 0; }
      section[data-testid="stSidebar"] h3              { font-size: 0.78rem; color: #888;
                                                         text-transform: uppercase;
                                                         letter-spacing: 1px; margin: 4px 0 0 0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Archive + Chapter ──────────────────────────────────────────────────────
    archives = sorted(p.stem for p in ARCHIVE_DIR.glob("*.mkv"))
    if not archives:
        st.error(f"No .mkv files in {ARCHIVE_DIR}")
        st.stop()
    archive_name = st.selectbox("Archive", archives, label_visibility="collapsed")

    chapters_file = METADATA_DIR / archive_name / "chapters.ffmetadata"
    all_chapters  = []
    if chapters_file.exists():
        all_chapters = parse_ffmetadata_chapters(chapters_file)

    selected_chapter = None
    if all_chapters:
        ch_titles      = [ch["title"] for ch in all_chapters]
        selected_title = st.selectbox("Chapter", ch_titles)
        selected_chapter = next(ch for ch in all_chapters if ch["title"] == selected_title)
    else:
        st.caption("⚠️ No chapters.ffmetadata — set range manually.")

    # ── Frame Range ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎞 Range")

    if selected_chapter:
        def_start = selected_chapter["start_frame"]
        def_end   = selected_chapter["end_frame"]
    else:
        def_start, def_end = 6000, 6500

    c1, c2      = st.columns(2)
    start_frame = c1.number_input("Start", value=def_start, step=100)
    end_frame   = c2.number_input("End",   value=def_end,   step=100)
    n_frames    = st.slider("Sample n", 20, 200, 100, 10)

    proxy_candidates = [
        ARCHIVE_DIR / f"{archive_name}_proxy.mp4",
        ARCHIVE_DIR / f"{archive_name}.mkv",
    ]
    default_video = next((str(p) for p in proxy_candidates if p.exists()), "")
    video_path = st.text_input("Video", default_video, label_visibility="collapsed",
                                placeholder="Video path…")
    reload_btn = st.button("🔄 Reload", use_container_width=True)

    # ── Weights (2×2 grid) ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚖️ Weights")
    wc1, wc2 = st.columns(2)
    w_chroma = wc1.slider("chroma",  0.0, 1.0, 0.25, 0.01)
    w_noise  = wc2.slider("noise",   0.0, 1.0, 0.25, 0.01)
    spark_chroma_ph = wc1.empty()
    spark_noise_ph = wc2.empty()
    wc3, wc4 = st.columns(2)
    w_tear   = wc3.slider("tear",    0.0, 1.0, 0.25, 0.01)
    w_wave   = wc4.slider("wave",    0.0, 1.0, 0.25, 0.01)
    spark_tear_ph = wc3.empty()
    spark_wave_ph = wc4.empty()

    # ── Threshold ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎚️ Threshold")
    mode = st.radio("", ["iqr", "value", "quantile"], horizontal=True,
                    label_visibility="collapsed")
    thresh_val, bad_pct, iqr_mult = 1.0, 10, 3.5
    if mode == "iqr":
        iqr_mult   = st.slider("k  (Q3 + k×IQR)", 1.0, 6.0, 3.5, 0.1)
        spark_thresh_ph = st.empty()
    elif mode == "value":
        thresh_val = st.slider("threshold", -5.0, 10.0, 1.0, 0.05)
        spark_thresh_ph = st.empty()
    elif mode == "quantile":
        bad_pct    = st.slider("bad %", 1, 60, 10)
        spark_thresh_ph = st.empty()

    # ── Display ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🖼️ Display")
    dc1, dc2 = st.columns(2)
    cols     = dc1.slider("cols",    4,  20, 10)
    thumb_w  = dc2.slider("px wide", 80, 300, 160, 20)
    ck1, ck2 = st.columns(2)
    show_score = ck1.checkbox("score", True)
    show_fnum  = ck2.checkbox("frame#", True)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💾 Export")
    ec1, ec2 = st.columns([3, 2])
    script_name = ec1.text_input("", "run_tracking_loss.py", label_visibility="collapsed",
                                  placeholder="filename.py")
    export_btn  = ec2.button("📝 Save", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Signal computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals_for_frame(bgr, crop=50):
    h, w = bgr.shape[:2]
    y0 = min(crop, max(0, h - 1));  y1 = max(y0 + 1, h - crop)
    x0 = min(crop, max(0, w - 1));  x1 = max(x0 + 1, w - crop)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = bgr

    # 1. Chroma loss
    s           = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    chroma_loss = 1.0 - float(np.mean(s) / 255.0)

    # 2. Noise energy
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_vars = np.var(gray, axis=1)
    mean_var = float(np.mean(row_vars))
    noise    = float(np.std(row_vars) / mean_var) if mean_var > 1e-6 else 0.0

    # 3. Row tear
    if gray.shape[0] > 1:
        tear = float(np.percentile(np.abs(gray[1:] - gray[:-1]).mean(axis=1), 95))
    else:
        tear = 0.0

    # 4. Wave energy — high-passed per-row horizontal centre-of-mass
    row_sums = gray.sum(axis=1)
    cols_idx = np.arange(gray.shape[1], dtype=np.float32)
    row_com  = (gray @ cols_idx) / np.maximum(row_sums, 1e-6)
    if row_com.shape[0] >= 5:
        trend = np.convolve(row_com, np.ones(5) / 5, mode="same")
        wave  = float(np.std(row_com - trend))
    else:
        wave = float(np.std(row_com))

    return chroma_loss, noise, tear, wave


# ═══════════════════════════════════════════════════════════════════════════════
# Frame extraction — cached in session_state
# ═══════════════════════════════════════════════════════════════════════════════

def extract_frames_and_signals(video_path, start, end, n, archive_name, ch_title):
    """
    Extract n evenly-spaced frames from [start, end].
    Loads raw signal values from per-chapter cache if available;
    otherwise computes and saves them.
    """
    frame_ids = np.linspace(int(start), int(end), int(n), dtype=int).tolist()

    # Try loading cached scores for the exact same frame IDs
    cached_fids, cached_sigs = load_cached_scores(archive_name, ch_title)
    cached_set = set(cached_fids) if cached_fids else set()
    need_video = not cached_fids or not all(f in cached_set for f in frame_ids)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None

    frames_bgr = []
    chroma_s, noise_s, tear_s, wave_s = [], [], [], []

    bar = st.progress(0, text="Loading frames…")
    for i, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        frames_bgr.append(bgr)

        if cached_fids and fid in cached_set:
            idx = cached_fids.index(fid)
            chroma_s.append(float(cached_sigs["chroma"][idx]))
            noise_s.append(float(cached_sigs["noise"][idx]))
            tear_s.append(float(cached_sigs["tear"][idx]))
            wave_s.append(float(cached_sigs["wave"][idx]))
        else:
            ch, no, te, wa = compute_signals_for_frame(bgr)
            chroma_s.append(ch); noise_s.append(no)
            tear_s.append(te);   wave_s.append(wa)

        bar.progress((i + 1) / len(frame_ids), text=f"Frame {fid}…")

    cap.release()
    bar.empty()

    sigs = {
        "chroma": np.array(chroma_s, dtype=np.float64),
        "noise":  np.array(noise_s,  dtype=np.float64),
        "tear":   np.array(tear_s,   dtype=np.float64),
        "wave":   np.array(wave_s,   dtype=np.float64),
    }

    # Save signals to per-chapter cache (always, so partial runs accumulate)
    if ch_title:
        save_scores(archive_name, ch_title, frame_ids, sigs)

    return frame_ids, frames_bgr, sigs


cache_key  = f"{video_path}|{archive_name}|{start_frame}|{end_frame}|{n_frames}"
needs_load = (
    "cache_key" not in st.session_state
    or st.session_state.cache_key != cache_key
    or reload_btn
)

if needs_load:
    p = Path(video_path)
    if not p.exists():
        st.error(f"Video not found: {video_path}")
        st.stop()
    ch_title_for_cache = selected_chapter["title"] if selected_chapter else ""
    fids, frames, signals = extract_frames_and_signals(
        video_path, start_frame, end_frame, n_frames,
        archive_name, ch_title_for_cache
    )
    if fids is None:
        st.error("Could not open video.")
        st.stop()
    st.session_state.cache_key = cache_key
    st.session_state.fids      = fids
    st.session_state.frames    = frames
    st.session_state.signals   = signals

fids    = st.session_state.fids
frames  = st.session_state.frames
signals = st.session_state.signals


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def robust_z(v):
    med   = np.median(v)
    mad   = np.median(np.abs(v - med))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = float(np.std(v)) or 1.0
    return (v - med) / scale


def combined_score(sigs, wc, wn, wt, ww):
    wsum = wc + wn + wt + ww or 1.0
    n = len(sigs["chroma"])
    cz = robust_z(sigs["chroma"]) if wc > 0 else np.zeros(n)
    nz = robust_z(sigs["noise"])  if wn > 0 else np.zeros(n)
    tz = robust_z(sigs["tear"])   if wt > 0 else np.zeros(n)
    wz = robust_z(sigs["wave"])   if ww > 0 else np.zeros(n)
    return (wc * cz + wn * nz + wt * tz + ww * wz) / wsum


def iqr_thresh(scores, k):
    v = scores[np.isfinite(scores)]
    return float(np.percentile(v, 75)) + k * (float(np.percentile(v, 75)) - float(np.percentile(v, 25)))


scores_np = combined_score(signals, w_chroma, w_noise, w_tear, w_wave)

if mode == "iqr":
    threshold = iqr_thresh(scores_np, iqr_mult)
elif mode == "value":
    threshold = thresh_val
else:
    threshold = float(np.quantile(scores_np, 1.0 - bad_pct / 100))

# Apply manual overrides
overrides = {}
if selected_chapter:
    overrides = load_overrides(archive_name, selected_chapter["title"])

is_bad = []
for fid, sc in zip(fids, scores_np):
    if fid in overrides:
        is_bad.append(overrides[fid] == "bad")
    else:
        is_bad.append(bool(sc >= threshold))
is_bad = np.array(is_bad)

n_bad  = int(is_bad.sum())
n_good = len(fids) - n_bad
n_overridden = sum(1 for f in fids if f in overrides)

# Sidebar sparklines: compact distributions + current-value marker.
spark_chroma = _sparkline_svg(_normalize_to_unit(signals["chroma"]), marker=w_chroma, vmin=0.0, vmax=1.0)
spark_noise = _sparkline_svg(_normalize_to_unit(signals["noise"]), marker=w_noise, vmin=0.0, vmax=1.0)
spark_tear = _sparkline_svg(_normalize_to_unit(signals["tear"]), marker=w_tear, vmin=0.0, vmax=1.0)
spark_wave = _sparkline_svg(_normalize_to_unit(signals["wave"]), marker=w_wave, vmin=0.0, vmax=1.0)
if spark_chroma:
    spark_chroma_ph.markdown(spark_chroma, unsafe_allow_html=True)
if spark_noise:
    spark_noise_ph.markdown(spark_noise, unsafe_allow_html=True)
if spark_tear:
    spark_tear_ph.markdown(spark_tear, unsafe_allow_html=True)
if spark_wave:
    spark_wave_ph.markdown(spark_wave, unsafe_allow_html=True)

score_min = float(np.min(scores_np)) if len(scores_np) else 0.0
score_max = float(np.max(scores_np)) if len(scores_np) else 1.0
spark_threshold = _sparkline_svg(scores_np, marker=threshold, vmin=score_min, vmax=score_max)
if spark_threshold:
    spark_thresh_ph.markdown(spark_threshold, unsafe_allow_html=True)

if selected_chapter:
    chapter_badframes_file, chapter_bad_count, chapter_bad_ranges = save_chapter_badframes_tsv(
        archive_name=archive_name,
        ch_title=selected_chapter["title"],
        frame_ids=fids,
        is_bad_flags=is_bad,
    )
else:
    chapter_badframes_file, chapter_bad_count, chapter_bad_ranges = (None, 0, 0)

chapter_settings_file = None
if selected_chapter:
    mode_payload = {
        "mode": mode,
        "iqr_mult": float(iqr_mult) if mode == "iqr" else None,
        "threshold_value": float(thresh_val) if mode == "value" else None,
        "bad_percent": int(bad_pct) if mode == "quantile" else None,
    }
    settings_payload = {
        "archive": archive_name,
        "chapter_title": selected_chapter["title"],
        "video_path": str(video_path),
        "frame_range": {"start": int(start_frame), "end": int(end_frame)},
        "sample_count": int(n_frames),
        "weights": {
            "chroma": float(w_chroma),
            "noise": float(w_noise),
            "tear": float(w_tear),
            "wave": float(w_wave),
        },
        "threshold": mode_payload,
        "display": {
            "cols": int(cols),
            "thumb_width": int(thumb_w),
            "show_score": bool(show_score),
            "show_frame_number": bool(show_fnum),
        },
        "outputs": {
            "frame_scores_tsv": str(scores_path(archive_name, selected_chapter["title"])),
            "overrides_tsv": str(overrides_path(archive_name, selected_chapter["title"])),
            "chapter_badframes_tsv": str(chapter_badframes_file) if chapter_badframes_file else "",
        },
        "sample_results": {
            "sample_frames": int(len(fids)),
            "sample_bad_frames": int(n_bad),
            "sample_good_frames": int(n_good),
            "threshold_value_effective": float(threshold),
        },
    }
    chapter_settings_file = save_chapter_settings_json(
        archive_name=archive_name,
        ch_title=selected_chapter["title"],
        payload=settings_payload,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stats bar
# ═══════════════════════════════════════════════════════════════════════════════

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Frames shown",  len(fids))
s2.metric("🔴 Bad",        f"{n_bad}  ({100*n_bad/max(1,len(fids)):.0f}%)")
s3.metric("🟢 Good",       f"{n_good}  ({100*n_good/max(1,len(fids)):.0f}%)")
s4.metric("Threshold",     f"{threshold:.3f}")
s5.metric("✏️ Overridden", n_overridden)
if chapter_badframes_file:
    st.caption(
        f"Saved chapter badframes: `{chapter_badframes_file.name}` "
        f"({chapter_bad_count} bad frame(s), {chapter_bad_ranges} range(s))"
    )
if chapter_settings_file:
    st.caption(f"Saved chapter settings: `{chapter_settings_file.name}`")


def classify_full_chapter_badframes(
    video_path,
    chapter_start_frame,
    chapter_end_frame,
    weight_chroma,
    weight_noise,
    weight_tear,
    weight_wave,
    threshold_mode,
    threshold_value,
    threshold_bad_pct,
    threshold_iqr_mult,
    manual_overrides,
):
    start_f = int(chapter_start_frame)
    end_f = int(chapter_end_frame)
    if end_f <= start_f:
        return [], np.array([], dtype=np.float64), np.array([], dtype=bool), float("nan")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for full chapter pass: {video_path}")

    frame_ids = list(range(start_f, end_f))
    total = len(frame_ids)
    chroma_s, noise_s, tear_s, wave_s = [], [], [], []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    bar = st.progress(0, text=f"Full chapter scan: 0/{total} frames")
    for i, fid in enumerate(frame_ids):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        ch, no, te, wa = compute_signals_for_frame(bgr)
        chroma_s.append(ch)
        noise_s.append(no)
        tear_s.append(te)
        wave_s.append(wa)
        if i == total - 1 or (i + 1) % 250 == 0:
            bar.progress((i + 1) / total, text=f"Full chapter scan: {i + 1}/{total} frames")
    cap.release()
    bar.empty()

    used = len(chroma_s)
    frame_ids = frame_ids[:used]
    sigs = {
        "chroma": np.array(chroma_s, dtype=np.float64),
        "noise": np.array(noise_s, dtype=np.float64),
        "tear": np.array(tear_s, dtype=np.float64),
        "wave": np.array(wave_s, dtype=np.float64),
    }
    if used == 0:
        return [], np.array([], dtype=np.float64), np.array([], dtype=bool), float("nan")

    scores = combined_score(sigs, weight_chroma, weight_noise, weight_tear, weight_wave)
    if threshold_mode == "iqr":
        threshold = iqr_thresh(scores, threshold_iqr_mult)
    elif threshold_mode == "value":
        threshold = float(threshold_value)
    else:
        threshold = float(np.quantile(scores, 1.0 - threshold_bad_pct / 100))

    is_bad_full = []
    for fid, sc in zip(frame_ids, scores):
        if fid in manual_overrides:
            is_bad_full.append(manual_overrides[fid] == "bad")
        else:
            is_bad_full.append(bool(sc >= threshold))
    is_bad_full = np.array(is_bad_full, dtype=bool)
    return frame_ids, scores, is_bad_full, float(threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# HTML frame grid
# ═══════════════════════════════════════════════════════════════════════════════

def bgr_to_b64(bgr, width=None):
    if width is not None:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (width, int(h * width / max(w, 1))), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_html_grid(frames, fids, scores_np, is_bad, overrides,
                    cols, thumb_w, show_score, show_fnum):
    cells = []
    for bgr, fid, sc, bad in zip(frames, fids, scores_np, is_bad):
        override_label = overrides.get(int(fid))
        if override_label:
            color    = "#e03030" if bad else "#30c870"
            badge    = " ✏️"
        else:
            color    = "#e03030" if bad else "#30c870"
            badge    = ""

        thumb_b64 = bgr_to_b64(bgr, width=thumb_w)
        full_b64  = bgr_to_b64(bgr, width=640)
        parts = []
        if show_fnum:  parts.append(f"#{fid}")
        if show_score: parts.append(f"{sc:.2f}")
        label = "  ".join(parts) + badge

        override_note = f" [override: {override_label}]" if override_label else ""
        popup_label = f"frame {fid} · score {sc:.4f} · <b>{'BAD' if bad else 'GOOD'}</b>{override_note}"

        cells.append(f"""
        <div class="cell">
          <div class="thumb-wrap" style="border-color:{color}"
               onmouseenter="showPopup(this)" onmouseleave="hidePopup(this)">
            <img class="thumb" src="{thumb_b64}" />
            <div class="popup">
              <img src="{full_b64}" style="max-width:640px;display:block;" />
              <div style="font-family:monospace;font-size:12px;color:{color};
                          padding:4px 8px;background:#0e0e0e;">{popup_label}</div>
            </div>
          </div>
          <div class="label" style="color:{color}">{label}</div>
        </div>""")

    html = f"""
    <style>
      .grid {{ display:grid; grid-template-columns:repeat({cols},{thumb_w}px); gap:6px;
               background:#111; padding:8px; }}
      .cell {{ display:flex; flex-direction:column; align-items:center; }}
      .thumb-wrap {{ position:relative; border:{BORDER}px solid; cursor:crosshair; line-height:0; }}
      .thumb {{ display:block; width:{thumb_w}px; }}
      .popup {{ display:none; position:fixed; z-index:9999; background:#0e0e0e;
                border:2px solid #444; box-shadow:0 0 60px rgba(0,0,0,.95); pointer-events:none; }}
      .label {{ font-family:monospace; font-size:10px; margin-top:2px; white-space:nowrap; }}
    </style>
    <div class="grid">{''.join(cells)}</div>
    <script>
      let mx=0, my=0;
      document.addEventListener("mousemove", e => {{
        mx=e.clientX; my=e.clientY;
        const v = document.querySelector(".popup[data-vis='1']");
        if (v) pos(v);
      }});
      function pos(p) {{
        const vw=window.innerWidth, vh=window.innerHeight;
        const pw=p.offsetWidth||640, ph=p.offsetHeight||400, off=16;
        let x=mx+off, y=my+off;
        if (x+pw>vw-8) x=mx-pw-off;
        if (y+ph>vh-8) y=my-ph-off;
        x=Math.max(8,Math.min(x,vw-pw-8));
        y=Math.max(8,Math.min(y,vh-ph-8));
        p.style.left=x+"px"; p.style.top=y+"px";
      }}
      function showPopup(w) {{
        const p=w.querySelector(".popup");
        p.style.display="block"; p.dataset.vis="1"; pos(p);
      }}
      function hidePopup(w) {{
        const p=w.querySelector(".popup");
        p.style.display="none"; p.dataset.vis="0";
      }}
    </script>"""
    return html


grid_html = build_html_grid(
    frames, fids, scores_np, is_bad, overrides,
    cols, thumb_w, show_score, show_fnum
)
h0, w0  = frames[0].shape[:2]
thumb_h = int(thumb_w * h0 / max(w0, 1))
n_rows  = math.ceil(len(frames) / cols)
grid_h  = n_rows * (thumb_h + BORDER * 2 + 20) + 40

st.components.v1.html(grid_html, height=grid_h, scrolling=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Manual override table
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("✏️ Manual Frame Overrides")

if not selected_chapter:
    st.info("Select a chapter to enable per-chapter overrides.")
else:
    # Build display dataframe from current frame sample
    override_rows = []
    for fid, sc, bad in zip(fids, scores_np, is_bad):
        ov = overrides.get(int(fid), "")
        override_rows.append({
            "frame":     int(fid),
            "score":     round(float(sc), 4),
            "computed":  "bad" if bool(sc >= threshold) else "good",
            "override":  ov,
            "effective": overrides.get(int(fid), "bad" if bad else "good"),
        })

    ov_df = pd.DataFrame(override_rows)

    st.caption(
        "Set **override** to `bad` or `good` to force a frame's label regardless of score. "
        "Leave blank to use the computed score. Hit **Save Overrides** to persist."
    )

    edited = st.data_editor(
        ov_df,
        column_config={
            "frame":     st.column_config.NumberColumn("Frame", disabled=True),
            "score":     st.column_config.NumberColumn("Score", format="%.4f", disabled=True),
            "computed":  st.column_config.TextColumn("Computed", disabled=True),
            "override":  st.column_config.SelectboxColumn(
                "Override",
                options=["", "good", "bad"],
                help="Force this frame's label. Leave blank for auto.",
            ),
            "effective": st.column_config.TextColumn("Effective", disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        key="override_editor",
    )

    if st.button("💾  Save Overrides", type="primary"):
        new_overrides = {}
        edited_frame_ids = []
        edited_is_bad = []
        for _, row in edited.iterrows():
            fid = int(row["frame"])
            ov = str(row["override"]).strip().lower()
            if ov in ("good", "bad"):
                new_overrides[fid] = ov
            computed = str(row.get("computed", "")).strip().lower()
            effective = ov if ov in ("good", "bad") else computed
            edited_frame_ids.append(fid)
            edited_is_bad.append(effective == "bad")
        save_overrides(archive_name, selected_chapter["title"], new_overrides)
        chapter_badframes_file, chapter_bad_count, chapter_bad_ranges = save_chapter_badframes_tsv(
            archive_name=archive_name,
            ch_title=selected_chapter["title"],
            frame_ids=edited_frame_ids,
            is_bad_flags=edited_is_bad,
        )
        st.success(f"Saved {len(new_overrides)} override(s) → "
                   f"`{overrides_path(archive_name, selected_chapter['title']).name}`")
        st.caption(
            f"Saved chapter badframes: `{chapter_badframes_file.name}` "
            f"({chapter_bad_count} bad frame(s), {chapter_bad_ranges} range(s))"
        )
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Render chapter with step_6_make_videos.py
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("🎬 Render Chapter")

# ── Custom HTML5 video player with frame + score HUD ─────────────────────────

def make_preview_video(video_path, scale=0.5):
    """
    Transcode to a half-size (or any scale) MP4 in a temp file.
    Returns the temp path — caller is responsible for cleanup.
    Much smaller base64 payload for the browser (~4x fewer pixels → ~3-5x smaller file).
    """
    import tempfile, shutil
    try:
        from common import FFMPEG_BIN
    except Exception:
        FFMPEG_BIN = "ffmpeg"

    tmp = tempfile.NamedTemporaryFile(suffix="_preview.mp4", delete=False)
    tmp.close()
    cmd = [
        FFMPEG_BIN, "-nostdin", "-v", "error",
        "-i", str(video_path),
        "-vf", f"scale=iw*{scale}:ih*{scale}",
        "-c:v", "libx264", "-crf", "26", "-preset", "fast",
        "-c:a", "copy",
        "-y", tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(tmp.name).exists():
        # Transcode failed — fall back to original
        shutil.copy(str(video_path), tmp.name)
    return tmp.name


def build_video_player(
    video_path,
    fids,
    scores_np,
    is_bad,
    chapter_start_frame=0,
    fps=FPS,
    bad_ranges_global=None,
):
    """
    fids         : global frame numbers from the tuner sample
    scores_np    : combined scores matching fids
    is_bad       : boolean array matching fids — the ACTUAL classification
                   from the tuner (weights + threshold mode + manual overrides)
    chapter_start_frame : global frame number where the chapter starts;
                   used to convert chapter-local video time → global frame for lookup
    """
    import os
    preview_path = make_preview_video(video_path, scale=0.5)
    try:
        with open(preview_path, "rb") as vf:
            video_b64 = base64.b64encode(vf.read()).decode()
    finally:
        try:
            os.unlink(preview_path)
        except Exception:
            pass
    video_uri = "data:video/mp4;base64," + video_b64

    # Build parallel JS arrays: global frame → (score, is_bad)
    # JS will convert local video frame to global via chapter_start_frame offset
    fids_js   = "[" + ",".join(str(int(f))          for f in fids)    + "]"
    scores_js = "[" + ",".join(f"{float(s):.4f}"    for s in scores_np) + "]"
    bad_js    = "[" + ",".join("true" if b else "false" for b in is_bad) + "]"
    bad_ranges_global = bad_ranges_global or []
    bad_ranges_js = (
        "[" + ",".join(f"[{int(a)},{int(b)}]" for (a, b) in bad_ranges_global) + "]"
        if bad_ranges_global
        else "[]"
    )

    return f"""
    <style>
      .vhs-player {{background:#0e0e0e;padding:10px;border-radius:6px;
                    border:1px solid #2a2a2a;display:inline-block;width:100%;}}
      .vhs-video-wrap {{position:relative; width:100%;}}
      .vhs-video  {{width:100%;display:block;}}
      .vhs-overlay {{
        position:absolute; top:8px; left:8px; z-index:5;
        font-family:monospace; font-size:12px; color:#fff;
        background:rgba(0,0,0,0.65); border:1px solid #444;
        padding:4px 8px; border-radius:4px;
      }}
      .vhs-hud    {{font-family:monospace;font-size:13px;color:#aaa;
                    padding:5px 8px;background:#161616;border-top:1px solid #222;
                    display:flex;gap:24px;align-items:center;}}
      .vhs-hud .val  {{color:#eee;font-weight:bold;}}
      .vhs-hud .bad  {{color:#e03030;font-weight:bold;}}
      .vhs-hud .good {{color:#30c870;font-weight:bold;}}
    </style>
    <div class="vhs-player">
      <div class="vhs-video-wrap">
        <video id="vhsvid" class="vhs-video" controls preload="metadata">
          <source src="{video_uri}" type="video/mp4"/>
        </video>
        <div id="hud-overlay" class="vhs-overlay">Frame -</div>
      </div>
      <div class="vhs-hud">
        <span>Frame: <span class="val" id="hud-frame">—</span></span>
        <span>Score: <span id="hud-score">—</span></span>
        <span id="hud-label">—</span>
      </div>
    </div>
    <script>
    (function(){{
      const fps={fps:.6f};
      const chapterStartFrame={int(chapter_start_frame)};
      const fids={fids_js}, scores={scores_js}, bads={bad_js};
      const badRanges={bad_ranges_js};

      // Map global frame → {{score, bad}} for nearest-sample lookup
      const frameMap=new Map();
      for(let i=0;i<fids.length;i++) frameMap.set(fids[i], {{score:scores[i], bad:bads[i]}});

      function nearestEntry(globalFrame){{
        if(frameMap.has(globalFrame)) return frameMap.get(globalFrame);
        let best=null, bestDist=Infinity;
        for(const [f,e] of frameMap){{
          const d=Math.abs(f-globalFrame);
          if(d<bestDist){{bestDist=d; best=e;}}
        }}
        return best;
      }}

      function inBadRanges(globalFrame){{
        for(let i=0;i<badRanges.length;i++){{
          const r=badRanges[i];
          if(globalFrame>=r[0] && globalFrame<=r[1]) return true;
        }}
        return false;
      }}

      const vid=document.getElementById("vhsvid");
      const hudFrame=document.getElementById("hud-frame");
      const hudScore=document.getElementById("hud-score");
      const hudLabel=document.getElementById("hud-label");
      const hudOverlay=document.getElementById("hud-overlay");

      function refreshHud(){{
        // Video time is chapter-local; convert to global frame for lookup
        const localFrame=Math.round(vid.currentTime*fps);
        const globalFrame=localFrame+chapterStartFrame;
        const entry=nearestEntry(globalFrame);
        const inBad=inBadRanges(globalFrame);
        hudFrame.textContent=globalFrame.toLocaleString()+" (local "+localFrame+")";
        hudOverlay.textContent="Frame "+globalFrame.toLocaleString()+" (local "+localFrame+")";
        if(entry!==null){{
          hudScore.textContent=entry.score.toFixed(4);
          hudScore.className=inBad?"bad":"good";
          hudLabel.textContent=inBad?"🔴 BAD":"🟢 GOOD";
          hudLabel.className=inBad?"bad":"good";
        }}
      }}
      vid.addEventListener("timeupdate", refreshHud);
      vid.addEventListener("seeked", refreshHud);
      vid.addEventListener("loadedmetadata", refreshHud);
    }})();
    </script>"""


def do_render(ch_title):
    chapter_badframes_for_render = None
    if selected_chapter:
        full_fids, full_scores, full_is_bad, full_threshold = classify_full_chapter_badframes(
            video_path=video_path,
            chapter_start_frame=selected_chapter["start_frame"],
            chapter_end_frame=selected_chapter["end_frame"],
            weight_chroma=w_chroma,
            weight_noise=w_noise,
            weight_tear=w_tear,
            weight_wave=w_wave,
            threshold_mode=mode,
            threshold_value=thresh_val,
            threshold_bad_pct=bad_pct,
            threshold_iqr_mult=iqr_mult,
            manual_overrides=overrides,
        )
        chapter_badframes_for_render, full_bad_count, full_bad_ranges = save_chapter_badframes_tsv(
            archive_name=archive_name,
            ch_title=selected_chapter["title"],
            frame_ids=full_fids,
            is_bad_flags=full_is_bad,
        )
        existing_settings = {}
        settings_file = chapter_settings_path(archive_name, selected_chapter["title"])
        if settings_file.exists():
            try:
                existing_settings = json.loads(settings_file.read_text(encoding="utf-8"))
            except Exception:
                existing_settings = {}
        existing_settings["render_full_chapter"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "chapter_frame_range": {
                "start": int(selected_chapter["start_frame"]),
                "end": int(selected_chapter["end_frame"]),
            },
            "frames_processed": int(len(full_fids)),
            "threshold_mode": str(mode),
            "threshold_value_effective": float(full_threshold) if np.isfinite(full_threshold) else None,
            "bad_frames": int(full_bad_count),
            "bad_ranges": int(full_bad_ranges),
            "badframes_tsv": str(chapter_badframes_for_render),
        }
        save_chapter_settings_json(
            archive_name=archive_name,
            ch_title=selected_chapter["title"],
            payload=existing_settings,
        )
        st.caption(
            f"Full chapter pass complete: {len(full_fids)} frame(s), "
            f"{full_bad_count} bad frame(s), {full_bad_ranges} range(s)"
        )
    cmd = [
        sys.executable,
        str(STEP6),
        "--archive",
        archive_name,
        "--title",
        ch_title,
        "--title-exact",
    ]
    if chapter_badframes_for_render:
        cmd += [
            "--badframes-tsv",
            str(chapter_badframes_for_render),
            "--badframes-archive",
            archive_name,
        ]
    with st.spinner(f"Rendering '{ch_title}'… this may take several minutes."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    full_log = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        st.success("Render completed.")
    else:
        st.error(f"step_6 exited with code {result.returncode}")
    with st.expander("📋 Log output", expanded=(result.returncode != 0)):
        st.code(full_log or "(no output)", language="text")
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from common import VIDEOS_DIR, CLIPS_DIR, safe
        rendered = None
        if result.returncode == 0:
            for search_dir in [VIDEOS_DIR, CLIPS_DIR]:
                candidate = search_dir / f"{safe(ch_title)}.mp4"
                if candidate.exists() and candidate.stat().st_size > 100_000:
                    rendered = candidate
                    break
        if rendered and result.returncode == 0:
            st.markdown(f"**Output:** `{rendered}`")
            ch_start = selected_chapter["start_frame"] if selected_chapter else 0
            bad_ranges_global = []
            if chapter_badframes_file:
                bad_ranges_global = load_badframe_ranges_tsv(chapter_badframes_file)
            player_html = build_video_player(
                str(rendered), fids, scores_np, is_bad,
                chapter_start_frame=ch_start,
                bad_ranges_global=bad_ranges_global,
            )
            st.components.v1.html(player_html, height=540)
        elif result.returncode == 0:
            st.warning("Render completed but output file not found — check the log above.")
        else:
            st.warning("Render failed; preview not shown to avoid using stale output.")
    except Exception as e:
        st.warning(f"Could not locate rendered file (common.py import failed: {e}). "
                   "Check the log above for the output path.")


if not selected_chapter:
    st.info("Select a chapter to enable rendering.")
elif not STEP6.exists():
    st.warning(f"step_6_make_videos.py not found at {STEP6}")
else:
    ch_title = selected_chapter["title"]
    st.markdown(f"**Chapter:** {ch_title}")
    _, run_col, _ = st.columns([3, 1, 3])
    run_btn_section = run_col.button("▶️ Render", key="render_section",
                                     type="primary", use_container_width=True)
    if run_btn_section or _render_top:
        do_render(ch_title)

# ═══════════════════════════════════════════════════════════════════════════════
# Export script
# ═══════════════════════════════════════════════════════════════════════════════

def generate_script(video_path, w_chroma, w_noise, w_tear, w_wave,
                    mode, thresh_val, bad_pct, iqr_mult):
    if mode == "iqr":
        thresh_comment = f"# Threshold: Q3 + {iqr_mult} × IQR per chapter-aligned window"
    elif mode == "value":
        thresh_comment = f"# Threshold: hard value {thresh_val:.4f}"
    else:
        thresh_comment = f"# Threshold: top {bad_pct}% flagged"

    return textwrap.dedent(f"""\
        #!/usr/bin/env python3.11
        \"\"\"
        Auto-generated by vhs_tuner.py
        {thresh_comment}
        \"\"\"

        from pathlib import Path
        import sys

        PROJECT_ROOT = Path(__file__).parent.parent.resolve()
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from tracking_loss import TrackingLossConfig, run_tracking_loss_classification

        config = TrackingLossConfig(
            archive               = "{archive_name}",
            video                 = r"{video_path}",
            weight_chroma         = {w_chroma},
            weight_noise          = {w_noise},
            weight_tear           = {w_tear},
            weight_wave           = {w_wave},
            iqr_mult              = {iqr_mult},
            threshold_window_size = 1000,
            crop_top              = 50,
            crop_bottom           = 50,
            crop_left             = 50,
            crop_right            = 50,
            export_review_png_count = 500,
        )

        if __name__ == "__main__":
            result = run_tracking_loss_classification(config=config)
            print("\\nDone.")
            print(f"  Bad frame ranges : {{result['badframes_path']}}")
            print(f"  Frame scores     : {{result['frame_scores_path']}}")
            print(f"  Summary          : {{result['summary_path']}}")
    """)


if export_btn:
    text = generate_script(
        video_path, w_chroma, w_noise, w_tear, w_wave,
        mode, thresh_val, bad_pct, iqr_mult
    )
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SCRIPTS_DIR / script_name
    out.write_text(text, encoding="utf-8")
    st.success(f"Saved → {out}")
    with st.expander("Preview"):
        st.code(text, language="python")
