#!/usr/bin/env python3
"""
VHS Bad Frame Detector — Interactive Tuning UI
Run with: streamlit run vhs_tuner.py

Loads a pre-computed frame_scores.tsv (edge_energy, row_instability, field_mismatch),
lets you tune weights + threshold with sliders, and shows a live thumbnail grid
colored red/green so you can see instantly what changed.
"""

import io
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VHS Frame Tuner",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Dark utilitarian look — feels right for archive work */
  body, .stApp { background: #111; color: #ddd; }
  section[data-testid="stSidebar"] { background: #1a1a1a; }
  .stSlider label { color: #aaa; font-size: 0.8rem; font-family: monospace; }
  .metric-box {
    background: #1e1e1e; border: 1px solid #333; border-radius: 4px;
    padding: 10px 14px; text-align: center; font-family: monospace;
  }
  .metric-box .val { font-size: 1.4rem; font-weight: bold; }
  .metric-box .lbl { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
  .bad  { color: #ff4d4d; }
  .good { color: #44cc88; }
  h1, h2, h3 { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("📼 VHS Bad Frame Tuner")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (pulled from your existing script)
# ─────────────────────────────────────────────────────────────────────────────

def robust_zscore(values):
    vals = np.asarray(values, dtype=np.float64)
    center = float(np.median(vals))
    mad = float(np.median(np.abs(vals - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        std = float(np.std(vals))
        scale = std if std > 1e-12 else 1.0
    return (vals - center) / scale


def combine_signals(edge, row, field, w_edge, w_row, w_field):
    weight_sum = w_edge + w_row + w_field
    if weight_sum <= 0:
        weight_sum = 1.0
    ez = robust_zscore(edge) if w_edge > 0 else np.zeros(len(edge))
    rz = robust_zscore(row)  if w_row  > 0 else np.zeros(len(row))
    fz = robust_zscore(field) if w_field > 0 else np.zeros(len(field))
    return (w_edge * ez + w_row * rz + w_field * fz) / weight_sum


def otsu_threshold(values, bins=256):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        return float(vmin)
    hist, edges = np.histogram(vals, bins=bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-12)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    between = w1[:-1] * w2[1:] * np.square(m1[:-1] - m2[1:])
    return float(centers[int(np.argmax(between))])


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — file inputs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Data Sources")
    tsv_path_str = st.text_input(
        "frame_scores.tsv path",
        placeholder="metadata/callahan_01_archive/badframes.tsv",
        help="Output from your tracking loss script. Must have edge_energy, row_instability, field_mismatch columns."
    )
    video_path_str = st.text_input(
        "Video file path (optional)",
        placeholder="../Archive/callahan_01_archive/callahan_01_proxy.mp4",
        help="Used to extract thumbnails. Leave blank to show frame numbers only."
    )
    n_thumbs = st.slider("Thumbnails to display", 100, 2000, 500, step=100)
    thumb_size = st.select_slider("Thumbnail size (px)", [60, 80, 100, 120, 160], value=80)

    st.divider()
    st.header("⚖️ Signal Weights")
    w_edge  = st.slider("weight_edge  (Sobel Y / horizontal tearing)", 0.0, 1.0, 0.45, 0.05)
    w_row   = st.slider("weight_row   (luma instability across rows)",   0.0, 1.0, 0.25, 0.05)
    w_field = st.slider("weight_field (even/odd field mismatch)",        0.0, 1.0, 0.30, 0.05)

    st.divider()
    st.header("🎚️ Threshold")
    thresh_mode = st.radio("Mode", ["value", "otsu", "quantile"], horizontal=True)
    if thresh_mode == "value":
        thresh_val = st.slider("Threshold value", -5.0, 10.0, 1.0, 0.05)
    elif thresh_mode == "quantile":
        bad_rate = st.slider("Expected bad %", 1, 50, 10)
    else:
        st.caption("Otsu: computed automatically from score distribution")

    st.divider()
    st.header("✂️ Signal Crop (for re-scoring)")
    st.caption("These only apply if you re-score from video. If loading TSV, crop was already applied.")

# ─────────────────────────────────────────────────────────────────────────────
# Load TSV
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_tsv(path_str):
    df = pd.read_csv(path_str, sep="\t", comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"frame", "edge_energy", "row_instability", "field_mismatch"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"TSV missing columns: {missing}")
        st.stop()
    df = df.sort_values("frame").reset_index(drop=True)
    return df


if not tsv_path_str:
    st.info("👈 Enter your **frame_scores.tsv** path in the sidebar to begin.")
    st.stop()

tsv_path = Path(tsv_path_str)
if not tsv_path.exists():
    st.error(f"File not found: {tsv_path}")
    st.stop()

df = load_tsv(tsv_path_str)
total_frames = len(df)

# ─────────────────────────────────────────────────────────────────────────────
# Re-compute scores from raw signals with current slider weights
# ─────────────────────────────────────────────────────────────────────────────

scores = combine_signals(
    df["edge_energy"].values,
    df["row_instability"].values,
    df["field_mismatch"].values,
    w_edge, w_row, w_field
)

if thresh_mode == "value":
    threshold = thresh_val
elif thresh_mode == "quantile":
    threshold = float(np.quantile(scores, 1.0 - bad_rate / 100))
else:
    threshold = otsu_threshold(scores)

labels = scores >= threshold
n_bad  = int(labels.sum())
n_good = total_frames - n_bad

# ─────────────────────────────────────────────────────────────────────────────
# Stats row
# ─────────────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
def metric(col, label, val, cls=""):
    col.markdown(f'<div class="metric-box"><div class="val {cls}">{val}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)

metric(c1, "Total frames", f"{total_frames:,}")
metric(c2, "Bad frames",   f"{n_bad:,}",  "bad")
metric(c3, "Good frames",  f"{n_good:,}", "good")
metric(c4, "Bad %",        f"{100*n_bad/max(1,total_frames):.1f}%")
metric(c5, "Threshold",    f"{threshold:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Score distribution histogram
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📊 Score distribution", expanded=False):
    import altair as alt
    hist_df = pd.DataFrame({"score": scores})
    chart = (
        alt.Chart(hist_df)
        .mark_bar(color="#555", opacity=0.8)
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=80), title="Combined Score"),
            y=alt.Y("count()", title="Frames"),
        )
    )
    rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="#ff4d4d", strokeWidth=2).encode(x="threshold:Q")
    st.altair_chart((chart + rule).properties(height=220), use_container_width=True)
    st.caption(f"Red line = threshold ({threshold:.4f}). Frames to the right → BAD.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail grid
# ─────────────────────────────────────────────────────────────────────────────

st.subheader(f"🎞 Frame Grid  ({n_thumbs} samples)")

filter_opt = st.radio("Show", ["All", "Bad only", "Good only"], horizontal=True)

# Pick which frame indices to show
frame_indices = df["frame"].values.tolist()
score_vals    = scores.tolist()
label_vals    = labels.tolist()

if filter_opt == "Bad only":
    pool = [(i, f, s) for i, (f, s, l) in enumerate(zip(frame_indices, score_vals, label_vals)) if l]
elif filter_opt == "Good only":
    pool = [(i, f, s) for i, (f, s, l) in enumerate(zip(frame_indices, score_vals, label_vals)) if not l]
else:
    pool = list(enumerate(zip(frame_indices, score_vals, label_vals)))
    pool = [(i, f, s) for i, (f, s) in [(i, (f, s)) for i, (f, s, l) in pool]]

# Evenly spaced sample
if len(pool) > n_thumbs:
    idxs = np.linspace(0, len(pool) - 1, n_thumbs, dtype=int)
    pool = [pool[i] for i in idxs]

# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail extraction
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Extracting thumbnails…")
def extract_thumbnails(video_path_str, frame_ids, size):
    """Extract thumbnails from video. Returns dict frame_id -> PIL Image."""
    thumbs = {}
    if not video_path_str:
        return thumbs
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return thumbs

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        thumbs[int(fid)] = Image.fromarray(rgb)
    cap.release()
    return thumbs


def make_placeholder(size, bad):
    img = Image.new("RGB", (size, size), color=(40, 20, 20) if bad else (20, 40, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size-1, size-1], outline=(180, 40, 40) if bad else (40, 160, 90), width=2)
    return img


def add_border(img, bad, score, size):
    """Add colored border + score text to thumbnail."""
    border = 3
    color = (220, 50, 50) if bad else (50, 200, 100)
    w, h = img.size
    canvas = Image.new("RGB", (w + border*2, h + border*2 + 14), color=(15, 15, 15))
    canvas.paste(img, (border, border))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width-1, canvas.height-1], outline=color, width=border)
    # Score text at bottom
    score_text = f"{score:.2f}"
    draw.text((border + 1, h + border + 1), score_text, fill=color)
    return canvas


# Collect frame IDs needed
needed_frame_ids = [f for (_, f, _) in pool] if pool and isinstance(pool[0], tuple) and len(pool[0]) == 3 else []

# Re-derive pool properly
pool_proper = []
fid_set_pool = set()
if filter_opt == "Bad only":
    for row_i, (fid, sc, lb) in enumerate(zip(frame_indices, score_vals, label_vals)):
        if lb:
            pool_proper.append((row_i, fid, sc, lb))
elif filter_opt == "Good only":
    for row_i, (fid, sc, lb) in enumerate(zip(frame_indices, score_vals, label_vals)):
        if not lb:
            pool_proper.append((row_i, fid, sc, lb))
else:
    for row_i, (fid, sc, lb) in enumerate(zip(frame_indices, score_vals, label_vals)):
        pool_proper.append((row_i, fid, sc, lb))

if len(pool_proper) > n_thumbs:
    idxs = np.linspace(0, len(pool_proper) - 1, n_thumbs, dtype=int)
    pool_proper = [pool_proper[i] for i in idxs]

needed_fids = [fid for (_, fid, _, _) in pool_proper]

thumbs = {}
if video_path_str:
    video_path = Path(video_path_str)
    if video_path.exists():
        thumbs = extract_thumbnails(video_path_str, tuple(needed_fids), thumb_size)
    else:
        st.warning(f"Video not found: {video_path_str} — showing placeholders")

# ─────────────────────────────────────────────────────────────────────────────
# Render grid
# ─────────────────────────────────────────────────────────────────────────────

CELL = thumb_size + 6 + 14  # thumb + border + score text row
COLS = max(1, min(40, st.session_state.get("grid_cols", 20)))

# Compute grid columns based on page width (approximate)
cols_slider = st.slider("Grid columns", 5, 40, 20, 1)
COLS = cols_slider

if not pool_proper:
    st.warning("No frames match the current filter.")
else:
    # Build one big image for performance
    n_cells = len(pool_proper)
    n_rows  = math.ceil(n_cells / COLS)
    grid_w  = COLS * (CELL)
    grid_h  = n_rows * (CELL)

    grid = Image.new("RGB", (grid_w, grid_h), color=(15, 15, 15))

    for cell_i, (_, fid, sc, lb) in enumerate(pool_proper):
        col_i = cell_i % COLS
        row_i = cell_i // COLS
        x = col_i * CELL
        y = row_i * CELL

        if fid in thumbs:
            thumb = thumbs[fid].copy()
            # Resize to thumb_size x thumb_size
            thumb = thumb.resize((thumb_size, int(thumb_size * thumb.height / max(thumb.width, 1))), Image.LANCZOS)
        else:
            thumb = make_placeholder(thumb_size, lb)

        bordered = add_border(thumb, lb, sc, thumb_size)
        grid.paste(bordered, (x, y))

    st.image(grid, use_container_width=True)

    # Summary below
    st.caption(
        f"Showing {len(pool_proper)} of {n_bad if filter_opt=='Bad only' else (n_good if filter_opt=='Good only' else total_frames)} frames. "
        f"🔴 = bad (score ≥ {threshold:.3f})  🟢 = good"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Export current config
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
with st.expander("💾 Export current settings as CLI flags"):
    flags = f"""--weight-edge {w_edge} \\
--weight-row {w_row} \\
--weight-field {w_field} \\
--threshold-mode {thresh_mode}"""
    if thresh_mode == "value":
        flags += f" \\\n--threshold-value {threshold:.4f}"
    elif thresh_mode == "quantile":
        flags += f" \\\n--bad-rate {bad_rate/100:.3f}"
    st.code(flags, language="bash")
    st.caption("Paste these flags into your existing tracking loss script to run the full batch.")
