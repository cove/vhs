# Core logic extracted from vhs_tuner.py

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# -- Project paths -------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name in {"scripts", "libs"} else _HERE
sys.path.insert(0, str(PROJECT_ROOT))

ARCHIVE_DIR  = PROJECT_ROOT / "../Archive"
METADATA_DIR = PROJECT_ROOT / "metadata"
FPS          = 30000 / 1001
BORDER       = 3
TUNER_CACHE_ROOT = Path(os.environ.get("VHS_TUNER_CACHE_DIR") or (Path(tempfile.gettempdir()) / "vhs_tuner_cache"))
TUNER_EXTRACT_DIR = TUNER_CACHE_ROOT / "extracts"
TUNER_FRAME_CACHE_DIR = TUNER_CACHE_ROOT / "frame_samples"
TUNER_DEBUG_EXTRACT_ENV = "VHS_TUNER_DEBUG_EXTRACT_FRAMES"
RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV = "RENDER_DEBUG_EXTRACT_FRAME_NUMBERS"
TUNER_FRAME_CACHE_VERSION = 1
_CACHE_SIGNAL_KEYS = ("chroma", "noise", "tear", "wave")
_LAST_CACHE_CLEANUP_TS = 0.0

try:
    from tracking_loss import TrackingLossConfig, run_tracking_loss_classification
    _HAS_TRACKING = True
except ImportError:
    TrackingLossConfig = None            # type: ignore
    run_tracking_loss_classification = None  # type: ignore
    _HAS_TRACKING = False

from common import (
    chapter_frame_bounds,
    combined_score,
    compute_threshold,
    get_bad_frames_for_chapter,
    make_extract_chapter,
    parse_chapters,
    update_chapter_bad_frames_in_render_settings,
)

# ===============================================================================
# Chapter / metadata helpers
# ===============================================================================

def load_archive_chapters(path: Path) -> list[dict]:
    # Keep chapter frame mapping identical to the render pipeline.
    path = Path(path)
    archive = str(path.parent.name)
    _ffm, chapters = parse_chapters(path)
    result = []
    for ch in chapters:
        title = str(ch.get("title", "Untitled"))
        start_sec = float(ch.get("start", 0.0))
        end_sec = float(ch.get("end", 0.0))
        start_frame, end_frame = chapter_frame_bounds(ch, fps_num=30000, fps_den=1001)
        result.append(
            {
                "title": title,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "bad_frames": get_bad_frames_for_chapter(archive, title),
            }
        )
    return result

# Backward-compatible alias for older callers.
parse_ffmetadata_chapters = load_archive_chapters


def _normalize_frame_span(ch_start: int, ch_end: int) -> tuple[int, int]:
    # Half-open chapter range: [start, end)
    start = int(ch_start)
    end = int(ch_end)
    if end < start:
        start, end = end, start
    if end == start:
        end = start + 1
    return start, end

def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}

def _env_int(name: str, default: int, minimum: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return max(minimum, int(default))
    try:
        val = int(raw)
    except Exception:
        return max(minimum, int(default))
    return max(minimum, val)

def _source_signature_token(source: str | Path | None) -> str:
    if source is None:
        return "nosrc"
    p = Path(source)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    try:
        st = p.stat()
        marker = f"{resolved}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
    except Exception:
        marker = f"{resolved}|missing"
    return hashlib.blake2b(marker.encode("utf-8"), digest_size=8).hexdigest()

def _cleanup_tuner_cache(force: bool = False) -> None:
    global _LAST_CACHE_CLEANUP_TS
    now = time.time()
    interval_sec = _env_int("VHS_TUNER_CACHE_CLEANUP_INTERVAL_SEC", default=300, minimum=30)
    if not force and (now - _LAST_CACHE_CLEANUP_TS) < float(interval_sec):
        return
    _LAST_CACHE_CLEANUP_TS = now

    root = TUNER_CACHE_ROOT
    if not root.exists():
        return

    ttl_days = _env_int("VHS_TUNER_CACHE_TTL_DAYS", default=14, minimum=1)
    max_bytes = _env_int("VHS_TUNER_CACHE_MAX_BYTES", default=2 * 1024 * 1024 * 1024, minimum=64 * 1024 * 1024)
    cutoff = now - (float(ttl_days) * 86400.0)

    files: list[tuple[float, int, Path]] = []
    total_bytes = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except Exception:
            continue
        mtime = float(st.st_mtime)
        size = int(max(0, st.st_size))
        if mtime < cutoff:
            try:
                p.unlink()
            except Exception:
                pass
            continue
        files.append((mtime, size, p))
        total_bytes += size

    if total_bytes > max_bytes:
        files.sort(key=lambda x: x[0])
        for _mtime, size, p in files:
            if total_bytes <= max_bytes:
                break
            try:
                p.unlink()
                total_bytes -= size
            except Exception:
                continue

    dirs = [p for p in root.rglob("*") if p.is_dir()]
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        try:
            d.rmdir()
        except Exception:
            continue

def _chapter_extract_cache_path(
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    debug_overlay: bool,
    source_video: str | Path | None = None,
) -> Path:
    start_i, end_i = _normalize_frame_span(ch_start, ch_end)
    mode = "debug" if bool(debug_overlay) else "clean"
    source_sig = _source_signature_token(source_video)
    stem = f"{archive}__{slugify(chapter_title)}__{start_i}_{end_i}__{mode}__{source_sig}"
    return TUNER_EXTRACT_DIR / stem / "extracted.mkv"

def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    try:
        return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()

def _ensure_render_chapter_extract(
    *,
    source_video: Path,
    archive: str,
    chapter_title: str,
    ch_start: int,
    ch_end: int,
    debug_overlay: bool,
) -> tuple[Path | None, str]:
    _cleanup_tuner_cache()
    start_i, end_i = _normalize_frame_span(ch_start, ch_end)
    expected_frames = max(1, end_i - start_i)
    out_path = _chapter_extract_cache_path(
        archive=archive,
        chapter_title=chapter_title,
        ch_start=start_i,
        ch_end=end_i,
        debug_overlay=debug_overlay,
        source_video=source_video,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and _video_frame_count(out_path) == expected_frames:
        return out_path, ""

    start_sec = float(start_i) * 1001.0 / 30000.0
    end_sec = float(end_i) * 1001.0 / 30000.0
    cmd = make_extract_chapter(
        source_video,
        start_sec,
        end_sec,
        out_path,
        start_frame=start_i,
        end_frame=end_i,
        debug_frame_numbers=bool(debug_overlay),
    )
    proc = subprocess.run(
        [str(x) for x in cmd],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or "ffmpeg extraction failed").strip()
    if _video_frame_count(out_path) != expected_frames:
        return None, (
            f"Extracted chapter frame count mismatch for {out_path.name}: "
            f"expected {expected_frames}"
        )
    return out_path, ""

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
    # Seed per-session overrides from persisted chapter BAD_FRAMES so a chapter
    # reload reflects previously saved decisions in the sampled frame view.
    start, end = _normalize_frame_span(ch_start, ch_end)
    bad_frames = get_bad_frames_for_chapter(str(archive or ""), str(chapter_title or ""))
    out: dict[int, str] = {}
    for fid in (bad_frames or []):
        try:
            fi = int(fid)
        except Exception:
            continue
        if start <= fi < end:
            out[fi] = "bad"
    return out


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
    chapters = load_archive_chapters(cf)
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
    out_path = update_chapter_bad_frames_in_render_settings(
        str(archive or ""),
        {str(chapter_title): out_global},
    )
    return out_path, len(out_global)


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

def _signals_cache_path(
    archive: str,
    ch_title: str,
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    frame_read_offset: int = 0,
) -> Path:
    s, e = _normalize_frame_span(start_frame, end_frame)
    source_sig = _source_signature_token(video_path)
    key_raw = "|".join(
        [
            str(int(TUNER_FRAME_CACHE_VERSION)),
            str(archive or "").strip(),
            str(ch_title or "").strip(),
            str(int(s)),
            str(int(e)),
            str(int(frame_read_offset)),
            str(source_sig),
        ]
    )
    key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:24]
    stem = f"{slugify(archive) or 'archive'}__{slugify(ch_title) or 'chapter'}__{key}"
    return TUNER_FRAME_CACHE_DIR / f"{stem}.json.gz"

def load_cached_signals(
    archive: str,
    ch_title: str,
    *,
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    frame_read_offset: int = 0,
) -> tuple[list[int] | None, dict | None, dict[int, str] | None]:
    path = _signals_cache_path(
        archive=archive,
        ch_title=ch_title,
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_read_offset=frame_read_offset,
    )
    if not path.exists():
        return None, None, None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return None, None, None

    try:
        version = int(payload.get("version", -1))
    except Exception:
        return None, None, None
    if version != int(TUNER_FRAME_CACHE_VERSION):
        return None, None, None

    fids_raw = payload.get("fids", [])
    sigs_raw = payload.get("signals", {})
    thumbs_raw = payload.get("thumbs", {})

    if not isinstance(fids_raw, list) or not isinstance(sigs_raw, dict):
        return None, None, None
    try:
        fids = [int(x) for x in fids_raw]
    except Exception:
        return None, None, None

    out_sigs: dict[str, np.ndarray] = {}
    for key in _CACHE_SIGNAL_KEYS:
        vals = sigs_raw.get(key)
        if not isinstance(vals, list) or len(vals) != len(fids):
            return None, None, None
        try:
            out_sigs[key] = np.asarray([float(v) for v in vals], dtype=np.float64)
        except Exception:
            return None, None, None

    thumbs: dict[int, str] = {}
    if isinstance(thumbs_raw, dict):
        for raw_fid, raw_b64 in thumbs_raw.items():
            if not isinstance(raw_b64, str):
                continue
            try:
                fid_i = int(raw_fid)
            except Exception:
                continue
            thumbs[fid_i] = raw_b64

    try:
        path.touch()
    except Exception:
        pass
    return fids, out_sigs, thumbs

def save_cached_signals(
    archive: str,
    ch_title: str,
    *,
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    frame_read_offset: int = 0,
    fids: list[int],
    sigs: dict,
    thumbs_by_fid: dict[int, str] | None = None,
) -> None:
    if not fids or not sigs:
        return

    fids_out = [int(x) for x in fids]
    n = len(fids_out)
    sigs_out: dict[str, list[float]] = {}
    for key in _CACHE_SIGNAL_KEYS:
        arr = np.asarray(sigs.get(key, []), dtype=np.float64)
        if int(arr.shape[0]) != n:
            return
        sigs_out[key] = [float(v) for v in arr.tolist()]

    fid_set = set(fids_out)
    thumbs_out: dict[str, str] = {}
    for raw_fid, raw_b64 in dict(thumbs_by_fid or {}).items():
        try:
            fid_i = int(raw_fid)
        except Exception:
            continue
        if fid_i not in fid_set:
            continue
        b64 = str(raw_b64 or "")
        if b64:
            thumbs_out[str(fid_i)] = b64

    s, e = _normalize_frame_span(start_frame, end_frame)
    path = _signals_cache_path(
        archive=archive,
        ch_title=ch_title,
        video_path=video_path,
        start_frame=s,
        end_frame=e,
        frame_read_offset=frame_read_offset,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": int(TUNER_FRAME_CACHE_VERSION),
        "archive": str(archive or ""),
        "chapter": str(ch_title or ""),
        "start_frame": int(s),
        "end_frame": int(e),
        "frame_read_offset": int(frame_read_offset),
        "source_sig": _source_signature_token(video_path),
        "updated_at": float(time.time()),
        "fids": fids_out,
        "signals": sigs_out,
        "thumbs": thumbs_out,
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        return
    _cleanup_tuner_cache()

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
    frame_read_offset: int = 0,
    progress=None,
    should_cancel=None,
    frame_callback=None,
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

    cached_fids, cached_sigs, cached_thumbs = load_cached_signals(
        archive,
        ch_title,
        video_path=video_path,
        start_frame=start_i,
        end_frame=end_i,
        frame_read_offset=frame_read_offset,
    )
    cached_lookup: dict[int, dict[str, float]] = {}
    if cached_fids and cached_sigs:
        for i, fid in enumerate(cached_fids):
            cached_lookup[fid] = {k: float(v[i]) for k, v in cached_sigs.items()}
    thumb_lookup: dict[int, str] = dict(cached_thumbs or {})

    cap: cv2.VideoCapture | None = None

    frames_b64: list[str] = []
    chroma_s, noise_s, tear_s, wave_s = [], [], [], []

    read_offset = int(frame_read_offset)
    n_total = max(1, len(frame_ids))
    for idx, fid in enumerate(frame_ids):
        if callable(should_cancel) and bool(should_cancel()):
            if cap is not None:
                cap.release()
            return None, None, None, "Load cancelled."
        read_fid = int(fid) - read_offset
        if progress is not None:
            progress(idx / n_total, desc=f"Frame {fid}...")

        c = cached_lookup.get(int(fid))
        cached_thumb = thumb_lookup.get(int(fid), "")
        need_decode = c is None or (include_thumbs and not cached_thumb)

        bgr = None
        if need_decode:
            if read_fid < 0:
                bgr = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                if cap is None:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        return None, None, None, f"Cannot open video: {video_path}"
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(read_fid))
                ok, read_bgr = cap.read()
                if not ok or read_bgr is None:
                    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
                else:
                    bgr = read_bgr

        if include_thumbs:
            if cached_thumb:
                frame_thumb = cached_thumb
            else:
                frame_thumb = _bgr_to_jpeg_b64(bgr if bgr is not None else np.zeros((240, 320, 3), dtype=np.uint8))
                thumb_lookup[int(fid)] = frame_thumb
            frames_b64.append(frame_thumb)
        else:
            frame_thumb = ""

        if c is not None:
            ch = float(c["chroma"])
            no = float(c["noise"])
            te = float(c["tear"])
            wa = float(c["wave"])
            chroma_s.append(ch); noise_s.append(no)
            tear_s.append(te);   wave_s.append(wa)
        else:
            compute_bgr = bgr if bgr is not None else np.zeros((240, 320, 3), dtype=np.uint8)
            ch, no, te, wa = _compute_signals(compute_bgr)
            chroma_s.append(ch); noise_s.append(no)
            tear_s.append(te);   wave_s.append(wa)
        if callable(frame_callback):
            frame_callback(
                int(fid),
                frame_thumb,
                float(ch),
                float(no),
                float(te),
                float(wa),
                idx + 1,
                len(frame_ids),
            )

    if cap is not None:
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
    save_cached_signals(
        archive,
        ch_title,
        video_path=video_path,
        start_frame=start_i,
        end_frame=end_i,
        frame_read_offset=frame_read_offset,
        fids=sorted_fids,
        sigs=sorted_sigs,
        thumbs_by_fid=thumb_lookup if include_thumbs else None,
    )

    return frame_ids, frames_b64, sigs, ""

# ===============================================================================
# Scoring / thresholding (shared in common.py)
# ===============================================================================

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
    Keeps a compact sparkline width while still shrinking to fit narrow layouts.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    SVG_W = 200  # viewBox width - browser scales to container

    if v.size == 0:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {height}" '
            f'style="background:#0a0a0a;display:block;width:220px;max-width:100%;border-radius:2px;margin-bottom:3px">'
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
        f'style="background:#0a0a0a;display:block;width:220px;max-width:100%;border-radius:2px;margin-bottom:3px">'
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
        height=24,
        line_color=_col(wc),
    )
    sc_noise = _sparkline_svg(
        _unit(sigs.get("noise", np.array([]))),
        wn,
        f"noise   w={wn:.2f}",
        height=24,
        line_color=_col(wn),
    )
    sc_tear = _sparkline_svg(
        _unit(sigs.get("tear", np.array([]))),
        wt,
        f"tear    w={wt:.2f}",
        height=24,
        line_color=_col(wt),
    )
    sc_wave = _sparkline_svg(
        _unit(sigs.get("wave", np.array([]))),
        ww,
        f"wave    w={ww:.2f}",
        height=24,
        line_color=_col(ww),
    )
    sc_score  = _sparkline_svg(scores, threshold, "composite score",
                                height=32, line_color="#5599dd")

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
    show_frame_labels: bool = False,
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
        local_fid = int(fid) - int(chapter_start_frame)

        if bool(show_frame_labels):
            # Optional burn-in for quick global/local visual verification.
            overlay_lines = [f"G:{int(fid)}", f"L:{local_fid}"]
            draw = ImageDraw.Draw(img)
            pad_x = 3
            pad_y = 2
            line_gap = 1
            line_sizes = []
            for line in overlay_lines:
                if hasattr(draw, "textbbox"):
                    x0, y0, x1, y1 = draw.textbbox((0, 0), line)
                    line_sizes.append((x1 - x0, y1 - y0))
                else:
                    line_sizes.append(draw.textsize(line))
            box_w = max((w for w, _ in line_sizes), default=0) + (2 * pad_x)
            box_h = sum((h for _, h in line_sizes)) + (line_gap * (len(overlay_lines) - 1)) + (2 * pad_y)
            draw.rectangle((0, 0, box_w, box_h), fill=(0, 0, 0))
            y = pad_y
            for line, (_, h) in zip(overlay_lines, line_sizes):
                draw.text((pad_x, y), line, fill=(255, 255, 255))
                y += h + line_gap

        # Restore fast visual scanning: colored border per frame state.
        styled = ImageOps.expand(img, border=BORDER, fill=color)
        items.append((styled, f"G:{int(fid)}  L:{local_fid}  s={sc:.2f}  {state_short}"))
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
# Apply: run tracking_loss and write BAD_FRAMES into render_settings.json
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
        logs.append("tracking_loss wrote BAD_FRAMES into render_settings.json")
        logs.append(f"Updated chapter blocks: {int(result.get('updated_chapters', 0))}")
    except Exception as exc:
        logs.append(f"tracking_loss failed: {exc}")
        return "\n".join(logs)

    logs.append("render pipeline reads BAD_FRAMES from render_settings.json")
    return "\n".join(logs)

# Chapter list helpers
def _get_archives() -> list[str]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(p.stem for p in ARCHIVE_DIR.glob("*.mkv"))


def _resolve_archive_video(archive: str) -> Path | None:
    proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
    mkv = ARCHIVE_DIR / f"{archive}.mkv"
    return proxy if proxy.exists() else mkv if mkv.exists() else None

CHAPTER_SELECT_LABEL = "-- select chapter --"
CHAPTER_MISSING_LABEL = "-- no chapters file found --"

def _get_chapter_titles(archive: str) -> list[str]:
    if not archive:
        return [CHAPTER_SELECT_LABEL]
    cf = METADATA_DIR / archive / "chapters.ffmetadata"
    if not cf.exists():
        return [CHAPTER_MISSING_LABEL]
    chapters = load_archive_chapters(cf)
    return [CHAPTER_SELECT_LABEL] + [ch["title"] for ch in chapters]

def _find_chapter(chapters: list[dict], title: str) -> dict | None:
    return next((c for c in chapters if c["title"] == title), None)

def _fmt_hms(total_sec: float) -> str:
    sec_i = max(0, int(round(float(total_sec))))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _chapter_frame_count(ch: dict) -> int:
    start_i, end_i = _normalize_frame_span(
        int(ch.get("start_frame", 0)),
        int(ch.get("end_frame", 0)),
    )
    return max(0, end_i - start_i)

def _chapter_bad_count(ch: dict) -> int:
    start_i, end_i = _normalize_frame_span(
        int(ch.get("start_frame", 0)),
        int(ch.get("end_frame", 0)),
    )
    bad = [
        int(x)
        for x in (ch.get("bad_frames", []) or [])
        if start_i <= int(x) < end_i
    ]
    return len(bad)

def _chapter_rows(chapters: list[dict]) -> tuple[list[list], list[list]]:
    rows: list[list] = []
    compact_rows: list[list] = []
    for idx, ch in enumerate(chapters):
        title = str(ch.get("title", "Untitled"))
        dur = max(0.0, float(ch.get("end_sec", 0.0)) - float(ch.get("start_sec", 0.0)))
        frame_count = _chapter_frame_count(ch)
        bad_count = _chapter_bad_count(ch)
        rows.append([idx + 1, title, _fmt_hms(dur), frame_count, bad_count])
        compact_rows.append([idx + 1, bad_count])
    return rows, compact_rows

def _chapter_details_md(ch: dict | None) -> str:
    if not ch:
        return "`Select a chapter from the left rail.`"
    title = str(ch.get("title", "Untitled"))
    dur = max(0.0, float(ch.get("end_sec", 0.0)) - float(ch.get("start_sec", 0.0)))
    frame_count = _chapter_frame_count(ch)
    bad_count = _chapter_bad_count(ch)
    start_f = int(ch.get("start_frame", 0))
    end_f = int(ch.get("end_frame", 0))
    return (
        f"**{title}**\n\n"
        f"Duration: `{_fmt_hms(dur)}`  |  "
        f"Frames: `{frame_count}`  |  "
        f"BAD already: `{bad_count}`\n\n"
        f"Frame span: `{start_f}` - `{end_f}` (end exclusive)"
    )


def build_archive_state(
    archive: str,
    selected_title: str | None = None,
) -> dict:
    titles = _get_chapter_titles(archive)
    cf = METADATA_DIR / archive / "chapters.ffmetadata" if archive else None
    chapters = load_archive_chapters(cf) if cf and cf.exists() else []
    chapter_titles = [str(ch.get("title", "")) for ch in chapters if str(ch.get("title", ""))]
    chapter_rows, compact_rows = _chapter_rows(chapters)

    if selected_title and str(selected_title) in chapter_titles:
        chapter_value = str(selected_title)
    else:
        chapter_value = chapter_titles[0] if chapter_titles else (titles[0] if titles else CHAPTER_SELECT_LABEL)

    picked = _find_chapter(chapters, chapter_value)
    start_frame = int(picked["start_frame"]) if picked else None
    end_frame = int(picked["end_frame"]) if picked else None
    details = _chapter_details_md(picked)

    if chapter_titles:
        status = ""
    elif titles and titles[0] == CHAPTER_MISSING_LABEL:
        status = "`No chapters.ffmetadata found for selected archive.`"
    else:
        status = "`No chapters available for selected archive.`"

    return {
        "titles": titles,
        "chapters": chapters,
        "chapter_titles": chapter_titles,
        "chapter_rows": chapter_rows,
        "compact_rows": compact_rows,
        "chapter_value": chapter_value,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "details": details,
        "status": status,
    }


def _frame_is_bad(
    fid: int,
    score: float,
    threshold: float,
    overrides: dict[int, str] | None,
) -> bool:
    ov = (overrides or {}).get(int(fid))
    if ov == "bad":
        return True
    if ov == "good":
        return False
    return bool(float(score) >= float(threshold))


def _select_visible_indices(
    fids: list[int],
    bad_fids: list[int],
    context: int,
) -> list[int]:
    if not fids:
        return []
    if not bad_fids:
        # Fallback: show sampled frames when detector finds no bad frames.
        return list(range(len(fids)))
    ctx = max(0, int(context))
    if ctx <= 0:
        bad_set = {int(f) for f in bad_fids}
        return [i for i, fid in enumerate(fids) if int(fid) in bad_set]
    spans = sorted((int(fid) - ctx, int(fid) + ctx) for fid in bad_fids)
    merged: list[list[int]] = []
    for lo, hi in spans:
        if not merged or lo > merged[-1][1] + 1:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    vis: list[int] = []
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


def sample_count_from_stride(start_frame: int, end_frame: int, sample_stride: int) -> int:
    span = max(1, int(end_frame) - int(start_frame))
    stride = max(1, int(sample_stride))
    return max(1, min(span, int(np.ceil(span / float(stride)))))


def build_review_data(
    *,
    fids: list[int],
    b64: list[str],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    t_mode: str,
    iqr_k: float,
    tval: float,
    bpct: float,
    context: int,
    chapter_start_frame: int,
    show_image_ids: bool,
) -> tuple[list[tuple[Image.Image, str]], str, str, str, str, str, str, list[int]]:
    if not fids or not b64:
        return [], "*(no frames loaded)*", _sparkline_svg(np.array([]), None, "", height=24), _sparkline_svg(np.array([]), None, "", height=24), _sparkline_svg(np.array([]), None, "", height=24), _sparkline_svg(np.array([]), None, "", height=24), _sparkline_svg(np.array([]), None, "", height=32), []
    sc = combined_score(sigs, wc, wn, wt, ww)
    thr = compute_threshold(sc, t_mode, iqr_k, tval, bpct)
    bad_fids = [
        int(fid)
        for fid, s in zip(fids, sc)
        if _frame_is_bad(int(fid), float(s), float(thr), overrides)
    ]
    vis_idx = _select_visible_indices(fids, bad_fids, int(context))
    vis_fids = [int(fids[i]) for i in vis_idx]
    vis_b64 = [b64[i] for i in vis_idx]
    vis_sc = np.array([sc[i] for i in vis_idx], dtype=np.float64)
    gallery_items = build_gallery_items(
        vis_b64,
        vis_fids,
        vis_sc,
        overrides,
        thr,
        chapter_start_frame=int(chapter_start_frame),
        show_frame_labels=bool(show_image_ids),
    )
    n_bad = sum(
        _frame_is_bad(int(f), float(s), float(thr), overrides)
        for f, s in zip(fids, sc)
    )
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
    return gallery_items, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids


def build_finalize_summary(
    *,
    chapter_title: str,
    chapter: dict | None,
    ch_start: int,
    ch_end: int,
    fids: list[int],
    sigs: dict[str, np.ndarray],
    overrides: dict[int, str],
    vis_fids: list[int],
    sample_stride: int,
    context: int,
    wc: float,
    wn: float,
    wt: float,
    ww: float,
    tm: str,
    ik: float,
    tv: float,
    bp: float,
) -> tuple[str, str]:
    if not fids or not sigs:
        return (
            "`No sampled frames loaded yet.`",
            "ERROR:  Load a chapter before finalizing.",
        )
    scores = combined_score(sigs, wc, wn, wt, ww)
    thr = compute_threshold(scores, tm, ik, tv, bp)
    n_bad = sum(
        1 for fid, sc in zip(fids, scores)
        if _frame_is_bad(int(fid), float(sc), float(thr), overrides or {})
    )
    n_ov = sum(1 for fid in fids if int(fid) in (overrides or {}))
    chapter_frames = _chapter_frame_count(chapter) if chapter else max(1, int(ch_end) - int(ch_start))
    bad_before = _chapter_bad_count(chapter) if chapter else 0
    summary = (
        f"### Final Stats\n"
        f"- Chapter: **{chapter_title}**\n"
        f"- Chapter total frames: `{chapter_frames}`\n"
        f"- Frames sampled: `{len(fids)}`\n"
        f"- Frames currently shown: `{len(vis_fids or [])}`\n"
        f"- BAD already in metadata: `{bad_before}`\n"
        f"- BAD in sampled set now: `{n_bad}`\n"
        f"- Manual overrides: `{n_ov}`\n"
        f"- IQR k: `{float(ik):.2f}` (threshold `{float(thr):.3f}`)\n"
        f"- Sample rate: `1/{max(1, int(sample_stride))}`  |  "
        f"Bad batch proximity: `{max(0, int(context))}`"
    )
    return summary, "`Review complete. Click Save and Return to Chapters.`"
