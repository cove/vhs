#!/usr/bin/env python3.11
"""
Label Studio bridge for VHS archive annotation workflows.

This script covers:
1) Bad-frame QA sync to chapters.ffmetadata (BAD_FRAMES local offsets)
2) Chapter timeline export/import to chapters.ffmetadata
3) People timeline export/import to metadata/<archive>/people.tsv

It also includes task scaffolding and a tracking_loss-backed regenerate command.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import cv2
import numpy as np

from common import (
    ARCHIVE_DIR,
    METADATA_DIR,
    parse_bad_frames_csv,
    update_chapter_bad_frames_in_ffmetadata,
)

try:
    from tracking_loss import TrackingLossConfig, run_tracking_loss_classification

    _HAS_TRACKING = True
except Exception:
    TrackingLossConfig = None  # type: ignore[assignment]
    run_tracking_loss_classification = None  # type: ignore[assignment]
    _HAS_TRACKING = False


FPS_NUM = 30000
FPS_DEN = 1001
FPS = FPS_NUM / FPS_DEN
DEFAULT_TIMEBASE = "1001/30000"
DEFAULT_LABEL_STUDIO_HOST = "0.0.0.0"
DEFAULT_LABEL_STUDIO_PORT = 8080
DEFAULT_HOOK_HOST = "0.0.0.0"
DEFAULT_HOOK_PORT = 9091
DEFAULT_VIDEO_URL_PREFIX = "/data/local-files/?d="

LABEL_STUDIO_CONFIG_XML = """<View>
  <Video name="video" value="$video"/>

  <!-- Use Case 1: Bad Frame QA (per-frame toggle) -->
  <Choices name="frame_quality" toName="video" perFrame="true">
    <Choice value="BAD"/>
  </Choices>

  <!-- Use Case 2: Chapter Boundaries (timeline segments) -->
  <Labels name="chapters" toName="video">
    <Label value="Chapter" background="#4A90D9"/>
  </Labels>
  <TextArea name="chapter_title" toName="video"
            perRegion="true" placeholder="Chapter title"/>

  <!-- Use Case 3: People on Screen (timeline segments) -->
  <Labels name="people" toName="video">
    <Label value="Person" background="#27AE60"/>
  </Labels>
  <TextArea name="person_name" toName="video"
            perRegion="true" placeholder="Person name"/>
</View>
"""


@dataclass(frozen=True)
class ChapterSpan:
    title: str
    start_raw: int
    end_raw: int
    timebase_num: int
    timebase_den: int
    start_frame: int
    end_frame: int
    bad_frames: list[int]


def _norm_title(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _timebase_to_frame(raw_value: int, tb_num: int, tb_den: int) -> int:
    num = int(raw_value) * int(tb_num) * FPS_NUM
    den = int(tb_den) * FPS_DEN
    if den == 0:
        return 0
    return int(round(num / den))


def seconds_to_frame(seconds: float) -> int:
    return int(round(float(seconds) * FPS))


def frame_to_seconds(frame_id: int) -> float:
    return float(frame_id) * FPS_DEN / FPS_NUM


def format_timecode(seconds: float, decimals: int = 4) -> str:
    secs = max(0.0, float(seconds))
    hours = int(secs // 3600)
    minutes = int((secs % 3600) // 60)
    rem = secs - (hours * 3600) - (minutes * 60)
    width = decimals + 3
    return f"{hours:02d}:{minutes:02d}:{rem:0{width}.{decimals}f}"


def _coerce_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    return []


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _coerce_str(value: Any) -> str:
    return str(value or "").strip()


def _find_chapters_file(archive: str) -> Path:
    return METADATA_DIR / archive / "chapters.ffmetadata"


def _build_video_url(video_path: Path, video_url_prefix: str) -> str:
    p = Path(video_path).resolve()
    p_text = str(p).replace("\\", "/")
    prefix = _coerce_str(video_url_prefix)
    if not prefix:
        return p_text
    if prefix == DEFAULT_VIDEO_URL_PREFIX:
        doc_root_text = _coerce_str(
            os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT")
            or os.getenv("LOCAL_FILES_DOCUMENT_ROOT")
        )
        doc_root = Path(doc_root_text).resolve() if doc_root_text else ARCHIVE_DIR.resolve()
        try:
            rel = p.relative_to(doc_root)
        except Exception as exc:
            raise ValueError(
                f"Video path is outside LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT: {p} (root: {doc_root})"
            ) from exc
        rel_text = str(rel).replace("\\", "/").lstrip("/")
        return f"{prefix}{urllib_parse.quote(rel_text, safe='/')}"
    return f"{prefix}{p_text}"


def resolve_archive_video(archive: str) -> Path:
    proxy = ARCHIVE_DIR / f"{archive}_proxy.mp4"
    mkv = ARCHIVE_DIR / f"{archive}.mkv"
    if proxy.exists():
        return proxy
    if mkv.exists():
        return mkv
    raise FileNotFoundError(
        f"No video found for archive '{archive}'. Checked: {proxy}, {mkv}"
    )


def list_archives() -> list[str]:
    names: set[str] = set()
    for mkv in ARCHIVE_DIR.glob("*.mkv"):
        names.add(mkv.stem)
    for proxy in ARCHIVE_DIR.glob("*_proxy.mp4"):
        stem = proxy.stem
        if stem.endswith("_proxy"):
            names.add(stem[:-6])
    return sorted(x for x in names if x)


def list_chapter_titles(archive: str) -> list[str]:
    archive_name = _coerce_str(archive)
    if not archive_name:
        return []
    chapters_file = _find_chapters_file(archive_name)
    if not chapters_file.exists():
        return []
    return [ch.title for ch in parse_chapters_ffmetadata(chapters_file)]


def parse_chapters_ffmetadata(path: Path) -> list[ChapterSpan]:
    chapters: list[ChapterSpan] = []
    current: dict[str, Any] = {}
    default_num, default_den = 1, 1_000_000_000

    def _finish(ch: dict[str, Any]) -> None:
        if "start_raw" not in ch or "end_raw" not in ch:
            return
        start_raw = int(ch["start_raw"])
        end_raw = int(ch["end_raw"])
        tb_num = int(ch.get("timebase_num", default_num))
        tb_den = int(ch.get("timebase_den", default_den))
        title = _coerce_str(ch.get("title", "Untitled")) or "Untitled"
        start_frame = _timebase_to_frame(start_raw, tb_num, tb_den)
        end_frame = _timebase_to_frame(end_raw, tb_num, tb_den)
        bad = parse_bad_frames_csv(ch.get("bad_frames", ""))
        chapters.append(
            ChapterSpan(
                title=title,
                start_raw=start_raw,
                end_raw=end_raw,
                timebase_num=tb_num,
                timebase_den=tb_den,
                start_frame=start_frame,
                end_frame=end_frame,
                bad_frames=bad,
            )
        )

    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line == "[CHAPTER]":
            _finish(current)
            current = {"timebase_num": default_num, "timebase_den": default_den}
            continue
        if "=" not in line or line.startswith(";"):
            continue
        key, _, value = line.partition("=")
        key_u = key.strip().upper()
        val = value.strip()
        if key_u == "TIMEBASE":
            try:
                n, d = val.split("/", 1)
                current["timebase_num"] = int(n)
                current["timebase_den"] = int(d)
            except Exception:
                pass
        elif key_u == "START":
            try:
                current["start_raw"] = int(val)
            except Exception:
                pass
        elif key_u == "END":
            try:
                current["end_raw"] = int(val)
            except Exception:
                pass
        elif key_u == "TITLE":
            current["title"] = val
        elif key_u == "BAD_FRAMES":
            current["bad_frames"] = val

    _finish(current)
    return chapters


def _split_ffmetadata_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "[CHAPTER]":
            return lines[:idx]
    return lines


def _default_ffmetadata_header(archive: str) -> list[str]:
    pretty = archive.replace("_", " ").strip()
    return [";FFMETADATA1", f"title={pretty}"]


def _normalize_ffmetadata_header(header: list[str], archive: str) -> list[str]:
    lines = [str(x).rstrip("\n") for x in (header or [])]
    if not lines:
        lines = _default_ffmetadata_header(archive)
    if not any(line.strip() == ";FFMETADATA1" for line in lines):
        lines.insert(0, ";FFMETADATA1")
    return lines


def _render_chapter_blocks(chapters: list[tuple[int, int, str]]) -> list[str]:
    out: list[str] = []
    for start, end, title in chapters:
        out.extend(
            [
                "[CHAPTER]",
                f"TIMEBASE={DEFAULT_TIMEBASE}",
                f"START={int(start)}",
                f"END={int(end)}",
                f"title={_coerce_str(title)}",
                "BAD_FRAMES=",
                "TRANSCRIPT=off",
                "",
            ]
        )
    return out


def write_chapters_ffmetadata(
    *,
    archive: str,
    chapters_file: Path,
    chapter_ranges: list[tuple[int, int, str]],
    replace_existing: bool,
) -> int:
    existing = parse_chapters_ffmetadata(chapters_file) if chapters_file.exists() else []
    if existing and not replace_existing:
        raise ValueError(
            f"{chapters_file} already has chapters. Pass --replace-chapters to overwrite."
        )
    cleaned: list[tuple[int, int, str]] = []
    for start, end, title in chapter_ranges:
        s = int(start)
        e = int(end)
        t = _coerce_str(title)
        if not t or e <= s:
            continue
        cleaned.append((s, e, t))
    cleaned.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    header = _normalize_ffmetadata_header(_split_ffmetadata_header(chapters_file), archive)
    output_lines = list(header)
    if output_lines and output_lines[-1].strip():
        output_lines.append("")
    output_lines.extend(_render_chapter_blocks(cleaned))
    chapters_file.parent.mkdir(parents=True, exist_ok=True)
    chapters_file.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
    return len(cleaned)


def find_chapter_by_title(chapters: list[ChapterSpan], title: str) -> ChapterSpan | None:
    wanted = _norm_title(title)
    for ch in chapters:
        if _norm_title(ch.title) == wanted:
            return ch
    return None


def find_chapter_for_frame(chapters: list[ChapterSpan], frame_id: int) -> ChapterSpan | None:
    candidates = [
        ch
        for ch in chapters
        if int(ch.start_frame) <= int(frame_id) < int(ch.end_frame)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda ch: ((ch.end_frame - ch.start_frame), ch.start_frame))
    return candidates[0]


def compute_frame_signals(frame_bgr: np.ndarray, crop: int = 50) -> tuple[float, float, float, float]:
    h, w = frame_bgr.shape[:2]
    y0 = min(int(crop), max(0, h - 1))
    y1 = max(y0 + 1, h - int(crop))
    x0 = min(int(crop), max(0, w - 1))
    x1 = max(x0 + 1, w - int(crop))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        roi = frame_bgr

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    chroma_loss = 1.0 - float(np.mean(sat) / 255.0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_vars = np.var(gray, axis=1)
    mean_var = float(np.mean(row_vars))
    noise = float(np.std(row_vars) / mean_var) if mean_var > 1e-6 else 0.0

    if gray.shape[0] > 1:
        row_diffs = np.abs(gray[1:] - gray[:-1]).mean(axis=1)
        tear = float(np.percentile(row_diffs, 95))
    else:
        tear = 0.0

    row_sums = gray.sum(axis=1)
    cols = np.arange(gray.shape[1], dtype=np.float32)
    row_com = (gray @ cols) / np.maximum(row_sums, 1e-6)
    if row_com.shape[0] >= 5:
        trend = np.convolve(row_com, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        wave = float(np.std(row_com - trend))
    else:
        wave = float(np.std(row_com))
    return chroma_loss, noise, tear, wave


def robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        std = float(np.std(arr))
        scale = std if std > 1e-12 else 1.0
    return (arr - median) / scale


def combined_score(
    sigs: dict[str, np.ndarray],
    weight_chroma: float,
    weight_noise: float,
    weight_tear: float,
    weight_wave: float,
) -> np.ndarray:
    total_w = (
        float(weight_chroma) + float(weight_noise) + float(weight_tear) + float(weight_wave)
    )
    if total_w <= 0:
        raise ValueError("At least one signal weight must be > 0.")
    return (
        robust_zscore(sigs["chroma"]) * float(weight_chroma)
        + robust_zscore(sigs["noise"]) * float(weight_noise)
        + robust_zscore(sigs["tear"]) * float(weight_tear)
        + robust_zscore(sigs["wave"]) * float(weight_wave)
    ) / total_w


def compute_threshold(
    scores: np.ndarray,
    *,
    mode: str,
    iqr_mult: float,
    threshold_value: float,
    bad_percentile: float,
) -> float:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    mode_norm = _coerce_str(mode).lower()
    if mode_norm == "iqr":
        q1 = float(np.percentile(finite, 25))
        q3 = float(np.percentile(finite, 75))
        return q3 + float(iqr_mult) * (q3 - q1)
    if mode_norm == "value":
        return float(threshold_value)
    if mode_norm in ("percentile", "quantile"):
        pct = max(0.0, min(100.0, float(bad_percentile)))
        return float(np.quantile(finite, 1.0 - (pct / 100.0)))
    raise ValueError(f"Unsupported threshold mode: {mode}")


def score_video_range(
    video_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    crop: int = 50,
) -> tuple[list[int], dict[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read frame count: {video_path}")

    start = max(0, int(start_frame))
    end = min(total, int(end_frame))
    step = max(1, int(frame_step))
    if end <= start:
        cap.release()
        raise ValueError(f"Invalid frame range [{start}, {end}).")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fids: list[int] = []
    chroma: list[float] = []
    noise: list[float] = []
    tear: list[float] = []
    wave: list[float] = []

    frame_id = start
    while frame_id < end:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if ((frame_id - start) % step) == 0:
            c, n, t, w = compute_frame_signals(bgr, crop=crop)
            fids.append(frame_id)
            chroma.append(c)
            noise.append(n)
            tear.append(t)
            wave.append(w)
        frame_id += 1
    cap.release()

    if not fids:
        raise RuntimeError("No sampled frames were scored.")
    sigs = {
        "chroma": np.asarray(chroma, dtype=np.float64),
        "noise": np.asarray(noise, dtype=np.float64),
        "tear": np.asarray(tear, dtype=np.float64),
        "wave": np.asarray(wave, dtype=np.float64),
    }
    return fids, sigs


def score_specific_frames(
    video_path: Path,
    frame_ids: list[int],
    *,
    crop: int = 50,
) -> tuple[list[int], dict[str, np.ndarray]]:
    target = sorted({int(x) for x in (frame_ids or []) if int(x) >= 0})
    if not target:
        return [], {
            "chroma": np.array([], dtype=np.float64),
            "noise": np.array([], dtype=np.float64),
            "tear": np.array([], dtype=np.float64),
            "wave": np.array([], dtype=np.float64),
        }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    chroma: list[float] = []
    noise: list[float] = []
    tear: list[float] = []
    wave: list[float] = []
    kept: list[int] = []
    for fid in target:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        c, n, t, w = compute_frame_signals(bgr, crop=crop)
        kept.append(int(fid))
        chroma.append(c)
        noise.append(n)
        tear.append(t)
        wave.append(w)
    cap.release()
    sigs = {
        "chroma": np.asarray(chroma, dtype=np.float64),
        "noise": np.asarray(noise, dtype=np.float64),
        "tear": np.asarray(tear, dtype=np.float64),
        "wave": np.asarray(wave, dtype=np.float64),
    }
    return kept, sigs


def select_focus_frame_ids(
    *,
    start_frame: int,
    end_frame: int,
    max_frames: int,
    coarse_fids: list[int],
    coarse_scores: np.ndarray,
    threshold: float,
    burst_radius: int = 4,
) -> list[int]:
    budget = max(1, int(max_frames))
    start = int(start_frame)
    end_excl = int(end_frame)
    if end_excl <= start:
        return [start]
    radius = max(1, int(burst_radius))
    end_incl = end_excl - 1

    bad_candidates: list[tuple[int, float]] = []
    for fid, score in zip(coarse_fids, coarse_scores):
        if float(score) >= float(threshold):
            bad_candidates.append((int(fid), float(score)))
    bad_candidates.sort(key=lambda row: row[1], reverse=True)

    selected: set[int] = set()
    for fid, _score in bad_candidates:
        lo = max(start, fid - radius)
        hi = min(end_incl, fid + radius)
        window = [f for f in range(lo, hi + 1) if f not in selected]
        if len(selected) + len(window) > budget:
            continue
        selected.update(window)
        if len(selected) >= budget:
            break

    if len(selected) < budget:
        fill_n = budget - len(selected)
        baseline = np.linspace(start, end_incl, fill_n, dtype=int).tolist()
        for fid in baseline:
            selected.add(int(fid))
            if len(selected) >= budget:
                break

    ordered = sorted(selected)
    if len(ordered) > budget:
        ordered = ordered[:budget]
    return ordered


def build_task_payload(
    *,
    archive: str,
    chapter_title: str,
    video_url_prefix: str = DEFAULT_VIDEO_URL_PREFIX,
    sample_budget: int,
    coarse_step: int,
    burst_radius: int,
    weight_chroma: float,
    weight_noise: float,
    weight_tear: float,
    weight_wave: float,
    threshold_mode: str,
    iqr_mult: float,
    threshold_value: float,
    bad_percentile: float,
) -> dict[str, Any]:
    chapters_file = _find_chapters_file(archive)
    if not chapters_file.exists():
        raise FileNotFoundError(f"chapters.ffmetadata not found: {chapters_file}")
    chapters = parse_chapters_ffmetadata(chapters_file)
    chapter = find_chapter_by_title(chapters, chapter_title)
    if chapter is None:
        raise ValueError(f"Chapter not found in {chapters_file}: {chapter_title}")
    if chapter.end_frame <= chapter.start_frame:
        raise ValueError(
            f"Chapter has invalid bounds: [{chapter.start_frame}, {chapter.end_frame})"
        )

    video_path = resolve_archive_video(archive)
    video_path_text = str(video_path.resolve()).replace("\\", "/")
    video_url = _build_video_url(video_path, video_url_prefix)

    coarse_fids, coarse_sigs = score_video_range(
        video_path,
        start_frame=chapter.start_frame,
        end_frame=chapter.end_frame,
        frame_step=max(1, int(coarse_step)),
        crop=50,
    )
    coarse_scores = combined_score(
        coarse_sigs,
        weight_chroma=weight_chroma,
        weight_noise=weight_noise,
        weight_tear=weight_tear,
        weight_wave=weight_wave,
    )
    threshold = compute_threshold(
        coarse_scores,
        mode=threshold_mode,
        iqr_mult=iqr_mult,
        threshold_value=threshold_value,
        bad_percentile=bad_percentile,
    )

    focus_ids = select_focus_frame_ids(
        start_frame=chapter.start_frame,
        end_frame=chapter.end_frame,
        max_frames=sample_budget,
        coarse_fids=coarse_fids,
        coarse_scores=coarse_scores,
        threshold=threshold,
        burst_radius=burst_radius,
    )

    focus_ids, focus_sigs = score_specific_frames(video_path, focus_ids, crop=50)
    focus_scores = combined_score(
        focus_sigs,
        weight_chroma=weight_chroma,
        weight_noise=weight_noise,
        weight_tear=weight_tear,
        weight_wave=weight_wave,
    )
    auto_bad = {
        int(fid)
        for fid, score in zip(focus_ids, focus_scores)
        if float(score) >= float(threshold)
    }
    listed_bad = {
        chapter.start_frame + int(local)
        for local in chapter.bad_frames
        if 0 <= int(local) < (chapter.end_frame - chapter.start_frame)
    }
    seeded_bad = sorted(auto_bad | listed_bad)

    predictions = [
        {
            "id": f"bad_{int(fid)}",
            "from_name": "frame_quality",
            "to_name": "video",
            "type": "choices",
            "value": {"choices": ["BAD"], "frame": int(fid)},
        }
        for fid in seeded_bad
    ]

    task: dict[str, Any] = {
        "data": {
            "video": video_url,
            "video_url": video_url,
            "archive": archive,
            "chapter_title": chapter.title,
            "chapter_start_frame": int(chapter.start_frame),
            "chapter_end_frame": int(chapter.end_frame),
            "sample_frame_ids": [int(x) for x in focus_ids],
            "existing_bad_frame_ids": sorted(int(x) for x in listed_bad),
            "auto_bad_frame_ids": sorted(int(x) for x in auto_bad),
            "threshold_mode": _coerce_str(threshold_mode),
            "threshold_value": float(threshold),
            "weights": {
                "chroma": float(weight_chroma),
                "noise": float(weight_noise),
                "tear": float(weight_tear),
                "wave": float(weight_wave),
            },
        },
        "predictions": [{"result": predictions}],
    }

    summary = {
        "archive": archive,
        "chapter_title": chapter.title,
        "video_path": video_path_text,
        "chapter_start_frame": int(chapter.start_frame),
        "chapter_end_frame": int(chapter.end_frame),
        "coarse_scored_frames": int(len(coarse_fids)),
        "focus_sample_frames": int(len(focus_ids)),
        "threshold_mode": _coerce_str(threshold_mode),
        "threshold_value": float(threshold),
        "predicted_bad_frames": int(len(auto_bad)),
        "listed_bad_frames": int(len(listed_bad)),
        "seeded_bad_frames": int(len(seeded_bad)),
    }
    return {"task": task, "summary": summary}


def write_label_studio_config(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(LABEL_STUDIO_CONFIG_XML.strip() + "\n", encoding="utf-8")


def regenerate_chapter_bad_frames(
    *,
    archive: str,
    chapter_title: str,
    frame_step: int,
    weight_chroma: float,
    weight_noise: float,
    weight_tear: float,
    weight_wave: float,
    iqr_mult: float,
) -> dict[str, Any]:
    if not _HAS_TRACKING:
        raise RuntimeError("tracking_loss module is not available.")

    chapters_file = _find_chapters_file(archive)
    if not chapters_file.exists():
        raise FileNotFoundError(f"chapters.ffmetadata not found: {chapters_file}")
    chapters = parse_chapters_ffmetadata(chapters_file)
    chapter = find_chapter_by_title(chapters, chapter_title)
    if chapter is None:
        raise ValueError(f"Chapter not found in {chapters_file}: {chapter_title}")

    video_path = resolve_archive_video(archive)
    config = TrackingLossConfig(  # type: ignore[call-arg]
        archive=archive,
        video=str(video_path),
        chapters_file=str(chapters_file),
        start_frame=int(chapter.start_frame),
        max_frame=int(chapter.end_frame),
        frame_step=max(1, int(frame_step)),
        weight_chroma=float(weight_chroma),
        weight_noise=float(weight_noise),
        weight_tear=float(weight_tear),
        weight_wave=float(weight_wave),
        iqr_mult=float(iqr_mult),
        threshold_window_size=1000,
    )
    result = run_tracking_loss_classification(config=config)  # type: ignore[misc]
    return {
        "archive": archive,
        "chapter_title": chapter.title,
        "video_path": str(video_path),
        "chapters_file": str(chapters_file),
        "tracking_result": result,
    }


def _iter_tasks_from_export(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("Unsupported export JSON format.")
    if isinstance(payload.get("tasks"), list):
        return [item for item in payload["tasks"] if isinstance(item, dict)]
    if isinstance(payload.get("data"), dict) or isinstance(payload.get("annotations"), list):
        return [payload]
    if isinstance(payload.get("result"), list):
        return [payload]
    raise ValueError("Unsupported export JSON format.")


def _latest_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    ann = task.get("annotations")
    if not isinstance(ann, list) or not ann:
        comp = task.get("completions")
        if isinstance(comp, list) and comp:
            ann = comp
        elif isinstance(task.get("result"), list):
            return task
        else:
            return None

    def _key(item: dict[str, Any]) -> tuple[str, str, int]:
        updated = _coerce_str(item.get("updated_at"))
        created = _coerce_str(item.get("created_at"))
        try:
            ident = int(item.get("id", 0))
        except Exception:
            ident = 0
        return (updated, created, ident)

    typed = [item for item in ann if isinstance(item, dict)]
    if not typed:
        return None
    typed.sort(key=_key)
    return typed[-1]


def _annotation_results(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(annotation.get("result"), list):
        return [r for r in annotation["result"] if isinstance(r, dict)]
    if isinstance(annotation.get("results"), list):
        return [r for r in annotation["results"] if isinstance(r, dict)]
    return []


def _extract_region_texts(results: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    text_map: dict[str, dict[str, str]] = {}
    for row in results:
        from_name = _coerce_str(row.get("from_name"))
        if not from_name:
            continue
        value = row.get("value")
        if not isinstance(value, dict):
            continue
        text_value = value.get("text")
        if isinstance(text_value, list):
            text = " ".join(_coerce_str(x) for x in text_value).strip()
        elif isinstance(text_value, str):
            text = _coerce_str(text_value)
        else:
            text = ""
        if not text:
            continue
        rid = _coerce_str(row.get("id"))
        if not rid:
            continue
        by_name = text_map.setdefault(rid, {})
        by_name[from_name] = text
    return text_map


def _extract_time_range(value: dict[str, Any]) -> tuple[float, float] | None:
    start = _coerce_float(
        value.get("start", value.get("startOffset", value.get("start_offset")))
    )
    end = _coerce_float(value.get("end", value.get("endOffset", value.get("end_offset"))))
    if start is None or end is None:
        return None
    if end <= start:
        return None
    return float(start), float(end)


def _result_has_bad_choice(value: dict[str, Any]) -> bool:
    choices = value.get("choices")
    if isinstance(choices, list):
        for item in choices:
            if _coerce_str(item).upper() == "BAD":
                return True
    labels = value.get("labels")
    if isinstance(labels, list):
        for item in labels:
            if _coerce_str(item).upper() == "BAD":
                return True
    return False


def _extract_frames_from_value(value: dict[str, Any], sampled_frames: list[int]) -> list[int]:
    out: set[int] = set()

    if "frame" in value:
        try:
            out.add(int(value["frame"]))
        except Exception:
            pass

    if isinstance(value.get("frames"), list):
        for item in value["frames"]:
            try:
                out.add(int(item))
            except Exception:
                continue

    if isinstance(value.get("sequence"), list):
        for item in value["sequence"]:
            if not isinstance(item, dict):
                continue
            if "frame" in item:
                try:
                    out.add(int(item["frame"]))
                    continue
                except Exception:
                    pass
            start = _coerce_float(item.get("start"))
            end = _coerce_float(item.get("end"))
            if start is None and end is None:
                continue
            f0 = seconds_to_frame(start or 0.0)
            f1 = seconds_to_frame(end if end is not None else start or 0.0)
            lo, hi = sorted((f0, f1))
            if (hi - lo) <= 1000:
                out.update(range(lo, hi + 1))
            else:
                out.add(lo)
                out.add(hi)

    idx = value.get("index")
    if idx is not None and sampled_frames:
        try:
            i = int(idx)
            if 0 <= i < len(sampled_frames):
                out.add(int(sampled_frames[i]))
        except Exception:
            pass

    if not out:
        time_range = _extract_time_range(value)
        if time_range is not None:
            f0 = seconds_to_frame(time_range[0])
            f1 = seconds_to_frame(time_range[1])
            lo, hi = sorted((f0, f1))
            if (hi - lo) <= 1000:
                out.update(range(lo, hi + 1))
            else:
                out.add(lo)
                out.add(hi)

    return sorted(int(x) for x in out if int(x) >= 0)


def _format_person_label(text: str) -> str:
    cleaned = _coerce_str(text)
    if not cleaned:
        return ""
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned
    return f"[{cleaned}]"


def write_people_tsv(path: Path, rows: list[tuple[float, float, str]]) -> int:
    dedup: set[tuple[str, str, str]] = set()
    lines: list[str] = []
    ordered = sorted(rows, key=lambda item: (float(item[0]), float(item[1]), item[2].lower()))
    for start, end, text in ordered:
        if float(end) <= float(start):
            continue
        label = _format_person_label(text)
        if not label:
            continue
        start_tc = format_timecode(start, decimals=4)
        end_tc = format_timecode(end, decimals=4)
        key = (start_tc, end_tc, label)
        if key in dedup:
            continue
        dedup.add(key)
        lines.append(f"{start_tc}\t{end_tc}\t{label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)


def _parse_timecode_seconds(text: str) -> float:
    val = _coerce_str(text).replace(",", ".")
    if not val:
        return 0.0
    parts = val.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])


def _load_people_rows(path: Path) -> list[tuple[float, float, str]]:
    if not path.exists():
        return []
    rows: list[tuple[float, float, str]] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("start"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) < 3:
            continue
        start = _parse_timecode_seconds(parts[0])
        end = _parse_timecode_seconds(parts[1])
        label = ",".join(parts[2:]).strip()
        if end <= start or not label:
            continue
        rows.append((start, end, label))
    return rows


def build_seed_results_for_archive(archive: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    chapters_file = _find_chapters_file(archive)
    if chapters_file.exists():
        chapters = parse_chapters_ffmetadata(chapters_file)
    else:
        chapters = []

    for idx, ch in enumerate(chapters):
        start_sec = frame_to_seconds(int(ch.start_frame))
        end_sec = frame_to_seconds(int(ch.end_frame))
        rid = f"ch_{idx:04d}"
        results.append(
            {
                "id": rid,
                "from_name": "chapters",
                "to_name": "video",
                "type": "labels",
                "value": {"start": float(start_sec), "end": float(end_sec), "labels": ["Chapter"]},
            }
        )
        results.append(
            {
                "id": rid,
                "from_name": "chapter_title",
                "to_name": "video",
                "type": "textarea",
                "value": {"text": [ch.title]},
            }
        )
        span = max(0, int(ch.end_frame) - int(ch.start_frame))
        for local in ch.bad_frames:
            lf = int(local)
            if lf < 0 or lf >= span:
                continue
            gfid = int(ch.start_frame) + lf
            results.append(
                {
                    "id": f"bad_{gfid}",
                    "from_name": "frame_quality",
                    "to_name": "video",
                    "type": "choices",
                    "value": {"choices": ["BAD"], "frame": int(gfid)},
                }
            )

    people_file = METADATA_DIR / archive / "people.tsv"
    for idx, (start_sec, end_sec, person) in enumerate(_load_people_rows(people_file)):
        rid = f"person_{idx:04d}"
        results.append(
            {
                "id": rid,
                "from_name": "people",
                "to_name": "video",
                "type": "labels",
                "value": {"start": float(start_sec), "end": float(end_sec), "labels": ["Person"]},
            }
        )
        results.append(
            {
                "id": rid,
                "from_name": "person_name",
                "to_name": "video",
                "type": "textarea",
                "value": {"text": [_coerce_str(person)]},
            }
        )
    return results


def build_archive_task(
    archive: str,
    video_url_prefix: str = DEFAULT_VIDEO_URL_PREFIX,
) -> dict[str, Any]:
    archive_name = _coerce_str(archive)
    if not archive_name:
        raise ValueError("archive cannot be empty")
    video_path = resolve_archive_video(archive_name)
    video_url = _build_video_url(video_path, video_url_prefix)
    seed_results = build_seed_results_for_archive(archive_name)
    task: dict[str, Any] = {
        "data": {"video": video_url, "video_url": video_url, "archive": archive_name},
    }
    if seed_results:
        task["predictions"] = [{"model_version": "ffmetadata_seed_v1", "result": seed_results}]
    return task


def build_tasks_for_archives(
    archives: list[str] | None = None,
    *,
    video_url_prefix: str = DEFAULT_VIDEO_URL_PREFIX,
) -> list[dict[str, Any]]:
    chosen = archives or list_archives()
    tasks: list[dict[str, Any]] = []
    for archive in chosen:
        tasks.append(build_archive_task(archive, video_url_prefix=video_url_prefix))
    return tasks


def write_project_bundle(
    *,
    config_path: Path,
    tasks_path: Path,
    archives: list[str] | None = None,
    video_url_prefix: str = DEFAULT_VIDEO_URL_PREFIX,
) -> dict[str, Any]:
    write_label_studio_config(config_path)
    tasks = build_tasks_for_archives(archives, video_url_prefix=video_url_prefix)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return {
        "config_path": str(config_path),
        "tasks_path": str(tasks_path),
        "task_count": len(tasks),
    }


def _ls_http_json(
    *,
    base_url: str,
    api_key: str,
    method: str,
    api_path: str,
    payload: dict[str, Any] | list[Any] | None = None,
) -> Any:
    base = _coerce_str(base_url).rstrip("/")
    if not base:
        raise ValueError("base_url cannot be empty")
    url = f"{base}{api_path}"
    body = None
    headers = {"Authorization": f"Token {_coerce_str(api_key)}"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url=url, method=method.upper(), headers=headers, data=body)
    try:
        with urllib_request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Label Studio API {method} {api_path} failed: {exc.code} {detail}")


def _is_ls_stream_reuse_error(exc: Exception) -> bool:
    return "You cannot access body after reading from request's data stream" in str(exc)


def setup_label_studio_project(
    *,
    base_url: str,
    api_key: str,
    title: str,
    video_url_prefix: str = DEFAULT_VIDEO_URL_PREFIX,
    archives: list[str] | None = None,
    project_id: int | None = None,
) -> dict[str, Any]:
    project_title = _coerce_str(title) or "VHS Archive Annotation Tool"
    if project_id is None:
        created = _ls_http_json(
            base_url=base_url,
            api_key=api_key,
            method="POST",
            api_path="/api/projects",
            payload={"title": project_title, "label_config": LABEL_STUDIO_CONFIG_XML},
        )
        project_id = int(created.get("id"))
    else:
        _ls_http_json(
            base_url=base_url,
            api_key=api_key,
            method="PATCH",
            api_path=f"/api/projects/{int(project_id)}",
            payload={"title": project_title, "label_config": LABEL_STUDIO_CONFIG_XML},
        )

    tasks = build_tasks_for_archives(archives, video_url_prefix=video_url_prefix)
    try:
        import_resp = _ls_http_json(
            base_url=base_url,
            api_key=api_key,
            method="POST",
            api_path=f"/api/projects/{int(project_id)}/import",
            payload=tasks,
        )
    except Exception as exc:
        if not _is_ls_stream_reuse_error(exc):
            raise
        created_ids: list[int] = []
        for task in tasks:
            payload = dict(task)
            payload["project"] = int(project_id)
            created = _ls_http_json(
                base_url=base_url,
                api_key=api_key,
                method="POST",
                api_path="/api/tasks/",
                payload=payload,
            )
            if isinstance(created, dict):
                try:
                    created_ids.append(int(created.get("id")))
                except Exception:
                    pass
        import_resp = {
            "fallback": "/api/tasks/",
            "stream_import_error": str(exc),
            "task_count": len(tasks),
            "task_ids": created_ids,
        }
    return {
        "project_id": int(project_id),
        "project_title": project_title,
        "tasks_sent": len(tasks),
        "import_response": import_resp,
    }


def start_label_studio(
    *,
    host: str = DEFAULT_LABEL_STUDIO_HOST,
    port: int = DEFAULT_LABEL_STUDIO_PORT,
    data_dir: str = "",
) -> int:
    exe = shutil.which("label-studio")
    if not exe:
        raise FileNotFoundError(
            "label-studio executable was not found on PATH. "
            "Install with: pip install label-studio"
        )
    cmd = [exe, "start", "--host", _coerce_str(host) or DEFAULT_LABEL_STUDIO_HOST, "--port", str(int(port))]
    if _coerce_str(data_dir):
        cmd.extend(["--data-dir", _coerce_str(data_dir)])
    env = dict(os.environ)
    env.setdefault("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", "true")
    env.setdefault("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", str(ARCHIVE_DIR.resolve()))
    # Keep legacy env names for compatibility with older Label Studio builds.
    env.setdefault("LOCAL_FILES_SERVING_ENABLED", env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"])
    env.setdefault("LOCAL_FILES_DOCUMENT_ROOT", env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"])
    return int(subprocess.call(cmd, env=env))


def apply_annotation_event(
    *,
    archive: str,
    task: dict[str, Any],
    annotation: dict[str, Any],
    chapters_file: Path | None = None,
    write_bad_frames: bool = True,
    write_people: bool = True,
    write_chapters: bool = False,
    replace_chapters: bool = False,
) -> dict[str, Any]:
    event_payload = [{"data": task.get("data", {}), "annotations": [annotation]}]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "event.json"
        p.write_text(json.dumps(event_payload), encoding="utf-8")
        return apply_label_studio_export(
            archive=archive,
            export_path=p,
            chapters_file=chapters_file,
            write_bad_frames=write_bad_frames,
            write_people=write_people,
            write_chapters=write_chapters,
            replace_chapters=replace_chapters,
        )


def serve_label_studio_hooks(
    *,
    host: str = DEFAULT_HOOK_HOST,
    port: int = DEFAULT_HOOK_PORT,
    archive: str = "",
    write_bad_frames: bool = True,
    write_people: bool = True,
    write_chapters: bool = False,
) -> None:
    archive_default = _coerce_str(archive)

    def _predict_for_task(task: dict[str, Any]) -> list[dict[str, Any]]:
        data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
        archive_name = _coerce_str(data.get("archive")) or archive_default
        if not archive_name:
            return []
        chapter_title = _coerce_str(data.get("chapter_title"))
        if chapter_title:
            try:
                payload = build_task_payload(
                    archive=archive_name,
                    chapter_title=chapter_title,
                    video_url_prefix="",
                    sample_budget=240,
                    coarse_step=5,
                    burst_radius=4,
                    weight_chroma=0.25,
                    weight_noise=0.25,
                    weight_tear=0.25,
                    weight_wave=0.25,
                    threshold_mode="iqr",
                    iqr_mult=3.5,
                    threshold_value=1.0,
                    bad_percentile=10.0,
                )
                preds = payload.get("task", {}).get("predictions", [])
                if preds and isinstance(preds, list):
                    first = preds[0]
                    if isinstance(first, dict):
                        result = first.get("result")
                        if isinstance(result, list):
                            return [x for x in result if isinstance(x, dict)]
            except Exception:
                return []
        return build_seed_results_for_archive(archive_name)

    class HookHandler(BaseHTTPRequestHandler):
        server_version = "VHSLabelStudioHooks/0.2"

        def _json(self, code: int, obj: dict[str, Any]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("", "/health"):
                self._json(200, {"ok": True})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except Exception:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                self._json(400, {"error": "invalid json"})
                return

            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/predict", "/ml/predict"):
                tasks = payload.get("tasks") if isinstance(payload, dict) else None
                if not isinstance(tasks, list):
                    tasks = []
                results = []
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    results.append(
                        {
                            "model_version": "vhs_label_studio_v0.2",
                            "score": 1.0,
                            "result": _predict_for_task(task),
                        }
                    )
                self._json(200, {"results": results, "model_version": "vhs_label_studio_v0.2"})
                return

            if path in ("/webhook", "/label-studio/webhook"):
                try:
                    task = payload.get("task") if isinstance(payload, dict) else None
                    annotation = payload.get("annotation") if isinstance(payload, dict) else None
                    if not isinstance(task, dict):
                        task = payload if isinstance(payload, dict) else {}
                    if not isinstance(annotation, dict):
                        ann = task.get("annotations") if isinstance(task, dict) else None
                        if isinstance(ann, list) and ann:
                            annotation = ann[-1] if isinstance(ann[-1], dict) else None
                    if not isinstance(annotation, dict):
                        annotation = {"result": payload.get("result", []) if isinstance(payload, dict) else []}
                    data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
                    archive_name = _coerce_str(data.get("archive")) or archive_default
                    if not archive_name:
                        self._json(400, {"error": "archive not provided in payload or server defaults"})
                        return
                    summary = apply_annotation_event(
                        archive=archive_name,
                        task=task,
                        annotation=annotation,
                        chapters_file=None,
                        write_bad_frames=write_bad_frames,
                        write_people=write_people,
                        write_chapters=write_chapters,
                        replace_chapters=False,
                    )
                    self._json(200, {"ok": True, "summary": summary})
                    return
                except Exception as exc:
                    self._json(500, {"error": str(exc)})
                    return

            self._json(404, {"error": "not found"})

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((_coerce_str(host) or DEFAULT_HOOK_HOST, int(port)), HookHandler)
    print(f"Hook server listening on http://{host}:{port}")
    server.serve_forever()


def apply_label_studio_export(
    *,
    archive: str,
    export_path: Path,
    chapters_file: Path | None = None,
    write_bad_frames: bool,
    write_people: bool,
    write_chapters: bool,
    replace_chapters: bool,
) -> dict[str, Any]:
    chapters_path = chapters_file or _find_chapters_file(archive)
    chapters = parse_chapters_ffmetadata(chapters_path) if chapters_path.exists() else []
    chapter_by_norm = {_norm_title(ch.title): ch for ch in chapters}

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    tasks = _iter_tasks_from_export(payload)

    bad_updates: dict[str, set[int]] = {}
    touched_chapters: set[str] = set()
    people_rows: list[tuple[float, float, str]] = []
    chapter_rows: list[tuple[float, float, str]] = []

    for task in tasks:
        data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
        task_archive = _coerce_str(data.get("archive"))
        if task_archive and task_archive != archive:
            continue
        sampled_frames = _coerce_int_list(
            data.get("sample_frame_ids", data.get("sampled_frame_ids", []))
        )

        task_chapter_raw = _coerce_str(data.get("chapter_title"))
        task_chapter = chapter_by_norm.get(_norm_title(task_chapter_raw))
        if task_chapter is not None:
            touched_chapters.add(task_chapter.title)
            bad_updates.setdefault(task_chapter.title, set())

        annotation = _latest_annotation(task)
        if annotation is None:
            continue
        results = _annotation_results(annotation)
        text_map = _extract_region_texts(results)

        for result in results:
            from_name = _coerce_str(result.get("from_name")).lower()
            value = result.get("value")
            if not isinstance(value, dict):
                continue
            rid = _coerce_str(result.get("id"))

            if from_name == "frame_quality":
                if not _result_has_bad_choice(value):
                    continue
                frame_ids = _extract_frames_from_value(value, sampled_frames)
                for fid in frame_ids:
                    chapter = task_chapter if task_chapter is not None else find_chapter_for_frame(
                        chapters, fid
                    )
                    if chapter is None:
                        continue
                    local = int(fid) - int(chapter.start_frame)
                    if local < 0 or local >= (chapter.end_frame - chapter.start_frame):
                        continue
                    touched_chapters.add(chapter.title)
                    bad_updates.setdefault(chapter.title, set()).add(local)

            elif from_name == "people":
                time_range = _extract_time_range(value)
                if time_range is None:
                    continue
                person_name = text_map.get(rid, {}).get("person_name", "")
                if not person_name:
                    continue
                people_rows.append((time_range[0], time_range[1], person_name))

            elif from_name == "chapters":
                time_range = _extract_time_range(value)
                if time_range is None:
                    continue
                chapter_title = text_map.get(rid, {}).get("chapter_title", "")
                chapter_title = chapter_title or f"Chapter {len(chapter_rows) + 1:02d}"
                chapter_rows.append((time_range[0], time_range[1], chapter_title))

    summary: dict[str, Any] = {
        "archive": archive,
        "tasks_read": len(tasks),
        "bad_frame_chapters_requested": 0,
        "bad_frame_chapters_updated": 0,
        "bad_frames_written": 0,
        "chapters_written": 0,
        "people_rows_written": 0,
        "chapters_file": str(chapters_path),
        "people_file": str(METADATA_DIR / archive / "people.tsv"),
    }

    if write_bad_frames:
        if not chapters_path.exists():
            raise FileNotFoundError(f"chapters.ffmetadata not found: {chapters_path}")
        for title in touched_chapters:
            bad_updates.setdefault(title, set())
        update_payload = {
            title: sorted(int(x) for x in vals)
            for title, vals in bad_updates.items()
        }
        touched = update_chapter_bad_frames_in_ffmetadata(chapters_path, update_payload)
        summary["bad_frame_chapters_requested"] = int(len(update_payload))
        summary["bad_frame_chapters_updated"] = int(touched)
        summary["bad_frames_written"] = int(sum(len(v) for v in update_payload.values()))

    if write_chapters:
        frame_ranges: list[tuple[int, int, str]] = []
        for start_sec, end_sec, title in chapter_rows:
            sf = seconds_to_frame(start_sec)
            ef = seconds_to_frame(end_sec)
            if ef <= sf:
                continue
            frame_ranges.append((sf, ef, _coerce_str(title)))
        written = write_chapters_ffmetadata(
            archive=archive,
            chapters_file=chapters_path,
            chapter_ranges=frame_ranges,
            replace_existing=replace_chapters,
        )
        summary["chapters_written"] = int(written)

    if write_people:
        people_path = METADATA_DIR / archive / "people.tsv"
        written = write_people_tsv(people_path, people_rows)
        summary["people_rows_written"] = int(written)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label Studio bridge for VHS archive annotation metadata sync."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser(
        "start-label-studio",
        help="Launch Label Studio server (the primary annotation UI runtime).",
    )
    p_start.add_argument("--host", default=DEFAULT_LABEL_STUDIO_HOST)
    p_start.add_argument("--port", type=int, default=DEFAULT_LABEL_STUDIO_PORT)
    p_start.add_argument("--data-dir", default="")

    p_hooks = sub.add_parser(
        "serve-hooks",
        help="Serve webhook + ML prediction endpoints for Label Studio integration.",
    )
    p_hooks.add_argument("--host", default=DEFAULT_HOOK_HOST)
    p_hooks.add_argument("--port", type=int, default=DEFAULT_HOOK_PORT)
    p_hooks.add_argument("--archive", default="")
    p_hooks.add_argument("--skip-bad-frames", action="store_true")
    p_hooks.add_argument("--skip-people", action="store_true")
    p_hooks.add_argument("--write-chapters", action="store_true")

    p_emit = sub.add_parser(
        "emit-project-bundle",
        help="Write Label Studio config XML + seeded tasks JSON for one-time import.",
    )
    p_emit.add_argument("--config-output", required=True)
    p_emit.add_argument("--tasks-output", required=True)
    p_emit.add_argument(
        "--video-url-prefix",
        default=DEFAULT_VIDEO_URL_PREFIX,
        help=(
            "Prefix for data.video/data.video_url "
            f"(default: {DEFAULT_VIDEO_URL_PREFIX!r})."
        ),
    )
    p_emit.add_argument("--archives", nargs="*", default=[])

    p_setup = sub.add_parser(
        "setup-project",
        help="Create/update Label Studio project and import seeded tasks via API.",
    )
    p_setup.add_argument("--url", default="http://127.0.0.1:8080")
    p_setup.add_argument("--api-key", required=True)
    p_setup.add_argument("--title", default="VHS Archive Annotation Tool")
    p_setup.add_argument("--project-id", type=int, default=0)
    p_setup.add_argument(
        "--video-url-prefix",
        default=DEFAULT_VIDEO_URL_PREFIX,
        help=(
            "Prefix for data.video/data.video_url "
            f"(default: {DEFAULT_VIDEO_URL_PREFIX!r})."
        ),
    )
    p_setup.add_argument("--archives", nargs="*", default=[])

    p_cfg = sub.add_parser(
        "write-config", help="Write the Label Studio labeling config XML."
    )
    p_cfg.add_argument("--output", required=True, help="Output XML path.")

    p_task = sub.add_parser(
        "build-task",
        help="Build a Label Studio task JSON scaffold for one archive chapter.",
    )
    p_task.add_argument("--archive", required=True)
    p_task.add_argument("--chapter", required=True, help="Chapter title.")
    p_task.add_argument("--output", required=True, help="Output JSON path.")
    p_task.add_argument(
        "--video-url-prefix",
        default=DEFAULT_VIDEO_URL_PREFIX,
        help=(
            "Prefix for data.video/data.video_url "
            f"(default: {DEFAULT_VIDEO_URL_PREFIX!r})."
        ),
    )
    p_task.add_argument("--sample-budget", type=int, default=240)
    p_task.add_argument("--coarse-step", type=int, default=5)
    p_task.add_argument("--burst-radius", type=int, default=4)
    p_task.add_argument("--weight-chroma", type=float, default=0.25)
    p_task.add_argument("--weight-noise", type=float, default=0.25)
    p_task.add_argument("--weight-tear", type=float, default=0.25)
    p_task.add_argument("--weight-wave", type=float, default=0.25)
    p_task.add_argument(
        "--threshold-mode", choices=["iqr", "value", "percentile"], default="iqr"
    )
    p_task.add_argument("--iqr-mult", type=float, default=3.5)
    p_task.add_argument("--threshold-value", type=float, default=1.0)
    p_task.add_argument("--bad-percentile", type=float, default=10.0)

    p_apply = sub.add_parser(
        "apply-export",
        help="Apply Label Studio export JSON to chapters.ffmetadata / people.tsv.",
    )
    p_apply.add_argument("--archive", required=True)
    p_apply.add_argument("--export", required=True, help="Label Studio export JSON.")
    p_apply.add_argument(
        "--chapters-file",
        default="",
        help="Optional explicit chapters.ffmetadata path.",
    )
    p_apply.add_argument(
        "--skip-bad-frames",
        action="store_true",
        help="Do not write BAD_FRAMES back to chapters.ffmetadata.",
    )
    p_apply.add_argument(
        "--skip-people",
        action="store_true",
        help="Do not write metadata/<archive>/people.tsv.",
    )
    p_apply.add_argument(
        "--write-chapters",
        action="store_true",
        help="Import chapter regions and write chapter blocks.",
    )
    p_apply.add_argument(
        "--replace-chapters",
        action="store_true",
        help="Required when writing chapters into a file that already has chapters.",
    )

    p_regen = sub.add_parser(
        "regenerate",
        help="Run tracking_loss full chapter BAD_FRAMES regeneration.",
    )
    p_regen.add_argument("--archive", required=True)
    p_regen.add_argument("--chapter", required=True, help="Chapter title.")
    p_regen.add_argument("--frame-step", type=int, default=1)
    p_regen.add_argument("--weight-chroma", type=float, default=0.25)
    p_regen.add_argument("--weight-noise", type=float, default=0.25)
    p_regen.add_argument("--weight-tear", type=float, default=0.25)
    p_regen.add_argument("--weight-wave", type=float, default=0.25)
    p_regen.add_argument("--iqr-mult", type=float, default=3.5)

    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return start_label_studio()

    parser = _build_parser()
    args = parser.parse_args(args_list)

    if args.command == "start-label-studio":
        return start_label_studio(
            host=_coerce_str(args.host) or DEFAULT_LABEL_STUDIO_HOST,
            port=int(args.port),
            data_dir=_coerce_str(args.data_dir),
        )

    if args.command == "serve-hooks":
        serve_label_studio_hooks(
            host=_coerce_str(args.host) or DEFAULT_HOOK_HOST,
            port=int(args.port),
            archive=_coerce_str(args.archive),
            write_bad_frames=not bool(args.skip_bad_frames),
            write_people=not bool(args.skip_people),
            write_chapters=bool(args.write_chapters),
        )
        return 0

    if args.command == "emit-project-bundle":
        archives = [_coerce_str(a) for a in list(args.archives or []) if _coerce_str(a)]
        summary = write_project_bundle(
            config_path=Path(args.config_output),
            tasks_path=Path(args.tasks_output),
            archives=archives if archives else None,
            video_url_prefix=_coerce_str(args.video_url_prefix),
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "setup-project":
        archives = [_coerce_str(a) for a in list(args.archives or []) if _coerce_str(a)]
        summary = setup_label_studio_project(
            base_url=_coerce_str(args.url),
            api_key=_coerce_str(args.api_key),
            title=_coerce_str(args.title),
            video_url_prefix=_coerce_str(args.video_url_prefix),
            archives=archives if archives else None,
            project_id=(int(args.project_id) if int(args.project_id) > 0 else None),
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "write-config":
        out = Path(args.output)
        write_label_studio_config(out)
        print(f"Wrote Label Studio config: {out}")
        return 0

    if args.command == "build-task":
        payload = build_task_payload(
            archive=_coerce_str(args.archive),
            chapter_title=_coerce_str(args.chapter),
            video_url_prefix=_coerce_str(args.video_url_prefix),
            sample_budget=int(args.sample_budget),
            coarse_step=int(args.coarse_step),
            burst_radius=int(args.burst_radius),
            weight_chroma=float(args.weight_chroma),
            weight_noise=float(args.weight_noise),
            weight_tear=float(args.weight_tear),
            weight_wave=float(args.weight_wave),
            threshold_mode=_coerce_str(args.threshold_mode),
            iqr_mult=float(args.iqr_mult),
            threshold_value=float(args.threshold_value),
            bad_percentile=float(args.bad_percentile),
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([payload["task"]], indent=2), encoding="utf-8")
        print(json.dumps(payload["summary"], indent=2))
        print(f"Wrote task JSON: {out}")
        return 0

    if args.command == "apply-export":
        chapters_file = (
            Path(args.chapters_file)
            if _coerce_str(args.chapters_file)
            else _find_chapters_file(_coerce_str(args.archive))
        )
        summary = apply_label_studio_export(
            archive=_coerce_str(args.archive),
            export_path=Path(args.export),
            chapters_file=chapters_file,
            write_bad_frames=not bool(args.skip_bad_frames),
            write_people=not bool(args.skip_people),
            write_chapters=bool(args.write_chapters),
            replace_chapters=bool(args.replace_chapters),
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "regenerate":
        result = regenerate_chapter_bad_frames(
            archive=_coerce_str(args.archive),
            chapter_title=_coerce_str(args.chapter),
            frame_step=int(args.frame_step),
            weight_chroma=float(args.weight_chroma),
            weight_noise=float(args.weight_noise),
            weight_tear=float(args.weight_tear),
            weight_wave=float(args.weight_wave),
            iqr_mult=float(args.iqr_mult),
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
