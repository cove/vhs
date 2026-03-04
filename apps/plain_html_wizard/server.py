#!/usr/bin/env python3.11
from __future__ import annotations

import json
import html
import re
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import (
    ARCHIVE_DIR,
    FFMPEG_BIN,
    METADATA_DIR,
    combined_score,
    compute_threshold,
    get_gamma_profile_for_chapter,
    update_chapter_gamma_in_render_settings,
)
from libs.vhs_tuner_core import (
    _chapter_bad_overrides,
    _chapter_extract_cache_path,
    _ensure_render_chapter_extract,
    _env_truthy,
    _find_chapter,
    _get_archives,
    _normalize_frame_span,
    _resolve_archive_video,
    build_archive_state,
    extract_frames,
    persist_bad_frames_for_chapter,
    sample_count_from_stride,
    select_focus_frame_ids,
    slugify,
    RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV,
    TUNER_DEBUG_EXTRACT_ENV,
)

STATIC_DIR = _HERE / "static"
INDEX_HTML = STATIC_DIR / "index.html"
SESSION_COOKIE = "vhs_plain_wizard_sid"
FPS_NUM = 30000
FPS_DEN = 1001
PEOPLE_TSV_HEADER = "start\tend\tpeople"


@dataclass
class SessionState:
    archive: str = ""
    chapter: str = ""
    chapters: list[dict[str, Any]] = field(default_factory=list)
    chapter_rows: list[list[Any]] = field(default_factory=list)

    start_frame: int = 0
    end_frame: int = 1
    sample_stride: int = 1
    context: int = 10
    strict_sampling: bool = True
    exact_extract: bool = True
    debug_extract: bool = False

    wc: float = 0.25
    wn: float = 0.25
    wt: float = 0.25
    ww: float = 0.25
    t_mode: str = "iqr"
    iqr_k: float = 3.5
    tval: float = 1.0
    bpct: float = 10.0

    fids: list[int] = field(default_factory=list)
    b64: list[str] = field(default_factory=list)
    sigs: dict[str, np.ndarray] = field(default_factory=dict)
    overrides: dict[int, str] = field(default_factory=dict)
    threshold: float = 0.0
    load_running: bool = False
    load_progress: float = 0.0
    load_message: str = ""
    load_sample_done: int = 0
    load_sample_total: int = 0
    load_cancel_requested: bool = False
    preview_running: bool = False
    preview_progress: float = 0.0
    preview_message: str = ""
    preview_frame_done: int = 0
    preview_frame_total: int = 0
    preview_video_path: str = ""
    gamma_default: float = 1.0
    gamma_ranges: list[dict[str, Any]] = field(default_factory=list)
    people_entries: list[dict[str, Any]] = field(default_factory=list)
    partial_fids: list[int] = field(default_factory=list)
    partial_b64: list[str] = field(default_factory=list)
    partial_sigs: dict[str, list[float]] = field(
        default_factory=lambda: {"chroma": [], "noise": [], "tear": [], "wave": []}
    )


_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, SessionState] = {}


def _set_load_progress(
    session: SessionState,
    *,
    running: bool | None = None,
    progress: float | None = None,
    message: str | None = None,
    sample_done: int | None = None,
    sample_total: int | None = None,
) -> None:
    if running is not None:
        session.load_running = bool(running)
    if progress is not None:
        session.load_progress = max(0.0, min(100.0, float(progress)))
    if message is not None:
        session.load_message = str(message)
    if sample_done is not None:
        session.load_sample_done = max(0, int(sample_done))
    if sample_total is not None:
        session.load_sample_total = max(0, int(sample_total))


def _set_preview_progress(
    session: SessionState,
    *,
    running: bool | None = None,
    progress: float | None = None,
    message: str | None = None,
    frame_done: int | None = None,
    frame_total: int | None = None,
) -> None:
    if running is not None:
        session.preview_running = bool(running)
    if progress is not None:
        session.preview_progress = max(0.0, min(100.0, float(progress)))
    if message is not None:
        session.preview_message = str(message)
    if frame_done is not None:
        session.preview_frame_done = max(0, int(frame_done))
    if frame_total is not None:
        session.preview_frame_total = max(0, int(frame_total))


def _normalize_iqr_k(raw: Any, default: float = 3.5) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(0.0, min(12.0, float(value)))

def _normalize_gamma_value(raw: Any, default: float = 1.0) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    if not (value == value):
        value = float(default)
    return max(0.05, min(8.0, float(value)))

def _normalize_gamma_ranges_payload(
    raw_ranges: Any,
    *,
    ch_start: int | None = None,
    ch_end: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[tuple[int, int, float, int]] = []
    for idx, item in enumerate(list(raw_ranges or [])):
        start = end = gamma = None
        if isinstance(item, dict):
            start = item.get("start_frame")
            end = item.get("end_frame")
            gamma = item.get("gamma")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, gamma = item[0], item[1], item[2]
        try:
            a = int(start)
            b = int(end)
        except Exception:
            continue
        if b <= a:
            continue
        if ch_start is not None and ch_end is not None:
            a = max(int(ch_start), a)
            b = min(int(ch_end), b)
            if b <= a:
                continue
        g = _normalize_gamma_value(gamma, default=1.0)
        rows.append((a, b, g, idx))
    if not rows:
        return []

    boundaries = set()
    for a, b, _g, _idx in rows:
        boundaries.add(int(a))
        boundaries.add(int(b))
    cuts = sorted(boundaries)
    out: list[tuple[int, int, float]] = []
    for i in range(len(cuts) - 1):
        seg_a = int(cuts[i])
        seg_b = int(cuts[i + 1])
        if seg_b <= seg_a:
            continue
        winner_idx = -1
        winner_gamma = None
        for a, b, g, idx in rows:
            if a <= seg_a and seg_b <= b and idx >= winner_idx:
                winner_idx = idx
                winner_gamma = float(g)
        if winner_gamma is None:
            continue
        if out and out[-1][1] == seg_a and abs(float(out[-1][2]) - float(winner_gamma)) < 1e-6:
            prev_a, _prev_b, prev_g = out[-1]
            out[-1] = (prev_a, seg_b, prev_g)
        else:
            out.append((seg_a, seg_b, float(winner_gamma)))
    return [
        {"start_frame": int(a), "end_frame": int(b), "gamma": round(float(g), 4)}
        for a, b, g in out
        if int(b) > int(a)
    ]


def _frame_to_seconds(frame_id: int) -> float:
    return float(int(frame_id) * FPS_DEN) / float(FPS_NUM)


def _seconds_to_timestamp(seconds: float) -> str:
    secs = max(0.0, float(seconds))
    total_ms = int(round(secs * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    whole_seconds, millis = divmod(rem, 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(whole_seconds):02d}.{int(millis):03d}"


def _parse_timestamp_seconds(raw: Any) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 1:
            value = float(parts[0])
        elif len(parts) == 2:
            mins = int(parts[0])
            secs = float(parts[1])
            value = float((mins * 60) + secs)
        elif len(parts) == 3:
            hours = int(parts[0])
            mins = int(parts[1])
            secs = float(parts[2])
            value = float((hours * 3600) + (mins * 60) + secs)
        else:
            return None
    except Exception:
        return None
    if not (value == value):
        return None
    return max(0.0, float(value))


def _normalize_people_entries_payload(
    raw_entries: Any,
    *,
    chapter_duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[tuple[float, float, str, int]] = []
    duration = None
    if chapter_duration_seconds is not None:
        try:
            duration = max(0.0, float(chapter_duration_seconds))
        except Exception:
            duration = None

    for idx, item in enumerate(list(raw_entries or [])):
        start_raw = end_raw = people_raw = None
        if isinstance(item, dict):
            start_raw = item.get("start_seconds", item.get("start"))
            end_raw = item.get("end_seconds", item.get("end"))
            people_raw = item.get("people")
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start_raw, end_raw, people_raw = item[0], item[1], item[2]
        start = _parse_timestamp_seconds(start_raw)
        end = _parse_timestamp_seconds(end_raw)
        if start is None or end is None or end <= start:
            continue
        if duration is not None:
            start = max(0.0, min(duration, float(start)))
            end = max(0.0, min(duration, float(end)))
            if end <= start:
                continue
        people = re.sub(r"\s+", " ", str(people_raw or "")).strip()
        if not people:
            continue
        rows.append((float(start), float(end), people, int(idx)))

    if not rows:
        return []

    rows.sort(key=lambda item: (item[0], item[1], item[3]))
    out: list[dict[str, Any]] = []
    for start, end, people, _idx in rows:
        start_s = round(float(start), 3)
        end_s = round(float(end), 3)
        if out:
            prev = out[-1]
            if (
                str(prev["people"]) == people
                and float(prev["end_seconds"]) + 0.001 >= start_s
            ):
                prev["end_seconds"] = max(float(prev["end_seconds"]), end_s)
                prev["end"] = _seconds_to_timestamp(float(prev["end_seconds"]))
                continue
        out.append(
            {
                "start_seconds": start_s,
                "end_seconds": end_s,
                "start": _seconds_to_timestamp(start_s),
                "end": _seconds_to_timestamp(end_s),
                "people": people,
            }
        )
    return out


def _read_people_tsv_rows(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for raw in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("start\t") or lower.startswith("start,end"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(",")
        if len(parts) < 3:
            continue
        start = _parse_timestamp_seconds(parts[0])
        end = _parse_timestamp_seconds(parts[1])
        people = re.sub(r"\s+", " ", ",".join(parts[2:]).strip())
        if start is None or end is None or end <= start or not people:
            continue
        rows.append((float(start), float(end), str(people)))
    return rows


def _canonicalize_people_tsv_rows(
    rows: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    items = []
    for start, end, people in list(rows or []):
        try:
            a = float(start)
            b = float(end)
        except Exception:
            continue
        if not (a == a and b == b):
            continue
        if b <= a:
            continue
        text = re.sub(r"\s+", " ", str(people or "")).strip()
        if not text:
            continue
        items.append((max(0.0, a), max(0.0, b), text))
    if not items:
        return []
    items.sort(key=lambda item: (item[0], item[1], item[2].lower()))
    out: list[tuple[float, float, str]] = []
    for start, end, people in items:
        if out:
            prev_start, prev_end, prev_people = out[-1]
            if prev_people == people and prev_end + 0.001 >= start:
                out[-1] = (prev_start, max(prev_end, end), prev_people)
                continue
        out.append((start, end, people))
    return out


def _write_people_tsv_rows(path: Path, rows: list[tuple[float, float, str]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [PEOPLE_TSV_HEADER]
    for start, end, people in list(rows or []):
        lines.append(
            f"{_seconds_to_timestamp(float(start))}\t{_seconds_to_timestamp(float(end))}\t{str(people)}"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_people_entries_for_chapter(archive: str, ch_start: int, ch_end: int) -> list[dict[str, Any]]:
    archive_name = str(archive or "").strip()
    if not archive_name:
        return []
    path = METADATA_DIR / archive_name / "people.tsv"
    if not path.exists():
        return []
    chapter_start = _frame_to_seconds(ch_start)
    chapter_end = _frame_to_seconds(ch_end)
    if chapter_end <= chapter_start:
        return []
    local_entries = []
    for start, end, people in _read_people_tsv_rows(path):
        lo = max(float(start), float(chapter_start))
        hi = min(float(end), float(chapter_end))
        if hi <= lo:
            continue
        local_entries.append(
            {
                "start_seconds": max(0.0, lo - chapter_start),
                "end_seconds": max(0.0, hi - chapter_start),
                "people": people,
            }
        )
    return _normalize_people_entries_payload(
        local_entries,
        chapter_duration_seconds=max(0.0, chapter_end - chapter_start),
    )


def _save_people_entries_for_chapter(
    archive: str,
    ch_start: int,
    ch_end: int,
    local_entries: list[dict[str, Any]],
) -> tuple[Path, int]:
    archive_name = str(archive or "").strip()
    path = METADATA_DIR / archive_name / "people.tsv"
    chapter_start = _frame_to_seconds(ch_start)
    chapter_end = _frame_to_seconds(ch_end)
    if chapter_end <= chapter_start:
        _write_people_tsv_rows(path, _canonicalize_people_tsv_rows(_read_people_tsv_rows(path)))
        return path, 0

    existing = _read_people_tsv_rows(path)
    kept: list[tuple[float, float, str]] = []
    for start, end, people in existing:
        if end <= chapter_start or start >= chapter_end:
            kept.append((float(start), float(end), str(people)))
            continue
        if start < chapter_start:
            kept.append((float(start), float(chapter_start), str(people)))
        if end > chapter_end:
            kept.append((float(chapter_end), float(end), str(people)))

    normalized_local = _normalize_people_entries_payload(
        local_entries,
        chapter_duration_seconds=max(0.0, chapter_end - chapter_start),
    )
    chapter_rows: list[tuple[float, float, str]] = []
    for item in normalized_local:
        start_local = _parse_timestamp_seconds(item.get("start_seconds", item.get("start")))
        end_local = _parse_timestamp_seconds(item.get("end_seconds", item.get("end")))
        if start_local is None or end_local is None or end_local <= start_local:
            continue
        people = re.sub(r"\s+", " ", str(item.get("people", "")).strip())
        if not people:
            continue
        chapter_rows.append(
            (
                float(chapter_start + start_local),
                float(chapter_start + end_local),
                str(people),
            )
        )

    merged = _canonicalize_people_tsv_rows([*kept, *chapter_rows])
    _write_people_tsv_rows(path, merged)
    return path, len(chapter_rows)


def _apply_profiles_from_payload(session: SessionState, payload: dict[str, Any] | None) -> None:
    payload = payload or {}
    raw_gamma_profile = payload.get("gamma_profile")
    if raw_gamma_profile is None:
        raw_gamma_profile = payload.get("gamma")
    if isinstance(raw_gamma_profile, dict):
        session.gamma_default = _normalize_gamma_value(
            raw_gamma_profile.get("default_gamma", session.gamma_default),
            default=session.gamma_default,
        )
        session.gamma_ranges = _normalize_gamma_ranges_payload(
            raw_gamma_profile.get("ranges", session.gamma_ranges),
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )

    chapter_duration = max(0.0, _frame_to_seconds(session.end_frame) - _frame_to_seconds(session.start_frame))
    raw_people_profile = payload.get("people_profile")
    if raw_people_profile is None:
        raw_people_profile = payload.get("people")
    if isinstance(raw_people_profile, dict):
        session.people_entries = _normalize_people_entries_payload(
            raw_people_profile.get("entries", session.people_entries),
            chapter_duration_seconds=chapter_duration,
        )
    elif isinstance(raw_people_profile, list):
        session.people_entries = _normalize_people_entries_payload(
            raw_people_profile,
            chapter_duration_seconds=chapter_duration,
        )


def _persist_session_progress(session: SessionState) -> tuple[Path | None, Path, int, int, int]:
    out_path, count, analyzed, err = persist_bad_frames_for_chapter(
        archive=session.archive,
        chapter_title=session.chapter,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        fids=session.fids,
        sigs=session.sigs,
        overrides=session.overrides,
        wc=session.wc,
        wn=session.wn,
        wt=session.wt,
        ww=session.ww,
        tm=session.t_mode,
        ik=session.iqr_k,
        tv=session.tval,
        bp=session.bpct,
        progress=None,
    )
    if err:
        raise RuntimeError(str(err))

    gamma_path = update_chapter_gamma_in_render_settings(
        archive=session.archive,
        chapter_title=session.chapter,
        gamma_ranges=session.gamma_ranges,
        default_gamma=session.gamma_default,
    )
    people_path, people_count = _save_people_entries_for_chapter(
        archive=session.archive,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
        local_entries=session.people_entries,
    )
    return out_path, gamma_path, int(count), int(analyzed), int(people_count)


def _details_text(chapter_row: dict[str, Any] | None) -> str:
    if not chapter_row:
        return "Select a chapter."
    return (
        f"{chapter_row['title']}\n"
        f"Duration: {chapter_row['time']} | Frames: {chapter_row['frames']} | BAD already: {chapter_row['bad']}\n"
        f"Frame span: {chapter_row['start_frame']} - {chapter_row['end_frame']} (end exclusive)"
    )


def _chapter_rows_payload(chapters: list[dict[str, Any]], chapter_rows: list[list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(chapter_rows):
        ch = chapters[i] if i < len(chapters) else {}
        out.append(
            {
                "index": int(row[0]),
                "title": str(row[1]),
                "time": str(row[2]),
                "frames": int(row[3]),
                "bad": int(row[4]),
                "start_frame": int(ch.get("start_frame", 0)),
                "end_frame": int(ch.get("end_frame", 1)),
                "bad_frames": [int(x) for x in (ch.get("bad_frames", []) or [])],
            }
        )
    return out


def _archive_state(session: SessionState, archive: str, selected_title: str | None = None) -> dict[str, Any]:
    data = build_archive_state(str(archive or ""), selected_title=selected_title)
    session.archive = str(archive or "")
    session.chapter = str(data["chapter_value"])
    session.chapters = list(data["chapters"])
    session.chapter_rows = list(data["chapter_rows"])
    session.start_frame = int(data["start_frame"] if data["start_frame"] is not None else 0)
    session.end_frame = int(data["end_frame"] if data["end_frame"] is not None else 1)

    rows = _chapter_rows_payload(session.chapters, session.chapter_rows)
    selected = next((r for r in rows if r["title"] == session.chapter), None)
    return {
        "archive": session.archive,
        "chapter": session.chapter,
        "status": str(data["status"]),
        "details": _details_text(selected),
        "start_frame": int(session.start_frame),
        "end_frame": int(session.end_frame),
        "chapters": rows,
    }


def _frame_status(session: SessionState, fid: int, score: float, thr: float) -> tuple[str, str]:
    ov = session.overrides.get(int(fid))
    if ov == "bad":
        return "bad", "MB"
    if ov == "good":
        return "good", "MG"
    if float(score) >= float(thr):
        return "bad", "AB"
    return "good", "AG"


def _build_review_payload(session: SessionState, include_images: bool) -> dict[str, Any]:
    if not session.fids or not session.sigs:
        return {
            "threshold": 0.0,
            "stats": {"total": 0, "bad": 0, "good": 0, "shown": 0, "overrides": 0},
            "frames": [],
        }

    scores = combined_score(session.sigs, session.wc, session.wn, session.wt, session.ww)
    thr = float(compute_threshold(scores, session.t_mode, session.iqr_k, session.tval, session.bpct))
    session.threshold = thr

    frames: list[dict[str, Any]] = []
    bad = 0
    for i, fid in enumerate(session.fids):
        score = float(scores[i])
        status, source = _frame_status(session, int(fid), score, thr)
        if status == "bad":
            bad += 1
        frame_item: dict[str, Any] = {
            "fid": int(fid),
            "status": status,
            "source": source,
            "score": round(score, 4),
            "label": f"G:{int(fid)}  L:{max(0, int(fid) - int(session.start_frame))}  s={score:.2f}  {source}",
        }
        if include_images:
            frame_item["image"] = session.b64[i]
        frames.append(frame_item)

    total = len(session.fids)
    return {
        "threshold": round(thr, 4),
        "stats": {
            "total": total,
            "bad": int(bad),
            "good": int(total - bad),
            "shown": total,
            "overrides": int(len(session.overrides)),
        },
        "frames": frames,
    }


def _build_partial_review_payload(session: SessionState, include_images: bool) -> dict[str, Any]:
    total = min(
        len(session.partial_fids),
        len(session.partial_b64),
        len(session.partial_sigs.get("chroma", [])),
        len(session.partial_sigs.get("noise", [])),
        len(session.partial_sigs.get("tear", [])),
        len(session.partial_sigs.get("wave", [])),
    )
    if total <= 0:
        return {
            "threshold": 0.0,
            "stats": {"total": 0, "bad": 0, "good": 0, "shown": 0, "overrides": 0},
            "frames": [],
        }
    tmp = SessionState()
    tmp.start_frame = int(session.start_frame)
    tmp.fids = [int(x) for x in session.partial_fids[:total]]
    tmp.b64 = list(session.partial_b64[:total])
    tmp.sigs = {
        "chroma": np.asarray(session.partial_sigs["chroma"][:total], dtype=np.float64),
        "noise": np.asarray(session.partial_sigs["noise"][:total], dtype=np.float64),
        "tear": np.asarray(session.partial_sigs["tear"][:total], dtype=np.float64),
        "wave": np.asarray(session.partial_sigs["wave"][:total], dtype=np.float64),
    }
    tmp.overrides = dict(session.overrides)
    tmp.wc = float(session.wc)
    tmp.wn = float(session.wn)
    tmp.wt = float(session.wt)
    tmp.ww = float(session.ww)
    tmp.t_mode = str(session.t_mode)
    tmp.iqr_k = float(session.iqr_k)
    tmp.tval = float(session.tval)
    tmp.bpct = float(session.bpct)
    return _build_review_payload(tmp, include_images=include_images)


def _selected_bad_frame_ids(session: SessionState) -> list[int]:
    if not session.fids or not session.sigs:
        return []
    scores = combined_score(session.sigs, session.wc, session.wn, session.wt, session.ww)
    thr = float(compute_threshold(scores, session.t_mode, session.iqr_k, session.tval, session.bpct))
    out: list[int] = []
    for fid, score in zip(session.fids, scores):
        status, _src = _frame_status(session, int(fid), float(score), thr)
        if status == "bad":
            out.append(int(fid))
    return out


def _summary_payload(session: SessionState) -> dict[str, Any]:
    review = _build_review_payload(session, include_images=False)
    bad_ids = [str(f["fid"]) for f in review["frames"] if f["status"] == "bad"]
    preview = ", ".join(bad_ids)
    gamma_ranges = _normalize_gamma_ranges_payload(
        session.gamma_ranges,
        ch_start=session.start_frame,
        ch_end=session.end_frame,
    )
    gamma_lines = []
    if gamma_ranges:
        gamma_lines.append("Gamma ranges:")
        for item in gamma_ranges:
            gamma_lines.append(
                f"- {int(item['start_frame'])}-{int(item['end_frame'])} (end exclusive): gamma {float(item['gamma']):.3f}"
            )
    else:
        gamma_lines.append("Gamma ranges: (none)")
    gamma_text = "\n".join(gamma_lines)

    people_entries = _normalize_people_entries_payload(
        session.people_entries,
        chapter_duration_seconds=max(0.0, _frame_to_seconds(session.end_frame) - _frame_to_seconds(session.start_frame)),
    )
    people_lines = []
    if people_entries:
        people_lines.append(f"People subtitle entries: {len(people_entries)}")
        for item in people_entries:
            people_lines.append(
                f"- {item['start']} - {item['end']}: {item['people']}"
            )
    else:
        people_lines.append("People subtitle entries: (none)")
    people_text = "\n".join(people_lines)

    summary_text = (
        f"Archive: {session.archive}\n"
        f"Chapter: {session.chapter}\n"
        f"Frame span: {session.start_frame} - {session.end_frame} (end exclusive)\n"
        f"Sample stride: 1/{session.sample_stride}\n"
        f"Bad batch proximity: {session.context}\n"
        f"IQR k: {session.iqr_k:.2f}\n"
        f"Threshold: {review['threshold']:.4f}\n"
        f"Analyzed samples: {review['stats']['total']}\n"
        f"Marked bad: {review['stats']['bad']}\n"
        f"Marked good: {review['stats']['good']}\n"
        f"Manual overrides: {review['stats']['overrides']}\n"
        f"Gamma default: {float(session.gamma_default):.3f}\n"
        f"{gamma_text}\n"
        f"{people_text}\n\n"
        f"BAD frame IDs (sampled set):\n{preview or '(none)'}"
    )
    return {"summary": summary_text, "review": review}


class WizardHandler(BaseHTTPRequestHandler):
    server_version = "VHSTuner/1.0"

    def _ensure_session(self) -> SessionState:
        self._set_cookie: str | None = None
        cookies = SimpleCookie(self.headers.get("Cookie", ""))
        sid = cookies.get(SESSION_COOKIE)
        sid_val = sid.value if sid else ""

        with _SESSION_LOCK:
            if sid_val and sid_val in _SESSIONS:
                return _SESSIONS[sid_val]

            sid_val = uuid.uuid4().hex
            sess = SessionState()
            _SESSIONS[sid_val] = sess
            self._set_cookie = f"{SESSION_COOKIE}={sid_val}; Path=/; HttpOnly; SameSite=Lax"
            return sess

    def _send_json(self, payload: dict[str, Any], code: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self._set_cookie:
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, code: int = HTTPStatus.OK, content_type: str = "text/plain; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self._set_cookie:
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()
        self.wfile.write(data)

    def _send_video_file(self, path: Path) -> None:
        p = Path(path)
        if not p.exists() or not p.is_file():
            self._send_error_json("Preview video is not available.", code=HTTPStatus.NOT_FOUND)
            return

        total_size = int(p.stat().st_size)
        if total_size <= 0:
            self._send_error_json("Preview video is empty.", code=HTTPStatus.NOT_FOUND)
            return

        start = 0
        end = total_size - 1
        status = HTTPStatus.OK
        content_range = None

        range_header = str(self.headers.get("Range", "") or "").strip()
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)$", range_header)
            if m:
                g_start, g_end = m.groups()
                if g_start:
                    start = int(g_start)
                if g_end:
                    end = int(g_end)
                if not g_end:
                    end = total_size - 1
                if g_start and not g_end:
                    end = total_size - 1
                if start < 0:
                    start = 0
                if end >= total_size:
                    end = total_size - 1
                if start > end or start >= total_size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{total_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    if self._set_cookie:
                        self.send_header("Set-Cookie", self._set_cookie)
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
                content_range = f"bytes {start}-{end}/{total_size}"

        length = (end - start) + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if content_range:
            self.send_header("Content-Range", content_range)
        if self._set_cookie:
            self.send_header("Set-Cookie", self._set_cookie)
        self.end_headers()

        with p.open("rb") as fh:
            fh.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                chunk = fh.read(min(chunk_size, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _preview_page_html(self, session: SessionState) -> str:
        title_text = html.escape(str(session.chapter or "Preview"))
        return (
            "<!doctype html>\n"
            "<html><head><meta charset=\"utf-8\"><title>VHS Preview</title>"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<style>"
            "html,body{width:100%;height:100%;overflow:hidden;}"
            "body{margin:0;background:#111;color:#eee;font-family:Segoe UI,Arial,sans-serif;}"
            ".wrap{display:grid;grid-template-rows:auto minmax(0,1fr);height:100vh;gap:8px;padding:10px;box-sizing:border-box;}"
            ".meta{font-size:13px;opacity:.9;}"
            "video{display:block;width:100%;height:100%;max-width:100%;max-height:100%;box-sizing:border-box;object-fit:contain;background:#000;border:1px solid #333;border-radius:8px;}"
            "</style></head><body><div class=\"wrap\">"
            f"<div class=\"meta\">Preview: {title_text}</div>"
            "<video controls autoplay preload=\"auto\" src=\"/api/preview_video\"></video>"
            "</div></body></html>"
        )

    def _read_json(self) -> dict[str, Any]:
        raw_len = int(self.headers.get("Content-Length", "0") or "0")
        if raw_len <= 0:
            return {}
        raw = self.rfile.read(raw_len)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_error_json(self, message: str, code: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"ok": False, "error": message}, code=code)

    def do_GET(self) -> None:
        session = self._ensure_session()
        parsed = urlparse(self.path)

        if parsed.path == "/":
            if not INDEX_HTML.exists():
                self._send_text("Missing index.html", code=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_text(INDEX_HTML.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
            return

        if parsed.path == "/preview":
            preview_raw = str(session.preview_video_path or "").strip()
            preview_path = Path(preview_raw) if preview_raw else None
            if not preview_path or not preview_path.exists() or not preview_path.is_file():
                self._send_text(
                    "Preview render is not ready yet. Run Preview Render from Step 2 first.",
                    code=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_text(self._preview_page_html(session), content_type="text/html; charset=utf-8")
            return

        if parsed.path == "/api/preview_video":
            preview_raw = str(session.preview_video_path or "").strip()
            preview_path = Path(preview_raw) if preview_raw else None
            if not preview_path or not preview_path.exists() or not preview_path.is_file():
                self._send_error_json("Preview render is not ready yet.", code=HTTPStatus.NOT_FOUND)
                return
            self._send_video_file(preview_path)
            return

        if parsed.path == "/api/archives":
            archives = _get_archives()
            selected = session.archive if session.archive in archives else (archives[0] if archives else "")
            self._send_json({"ok": True, "archives": archives, "selected": selected})
            return

        if parsed.path == "/api/archive_state":
            params = parse_qs(parsed.query)
            archive = str((params.get("archive", [""])[0] or "").strip())
            chapter = str((params.get("chapter", [""])[0] or "").strip())
            if not archive:
                archives = _get_archives()
                archive = archives[0] if archives else ""
            state = _archive_state(session, archive, selected_title=(chapter or None))
            self._send_json({"ok": True, "archive_state": state})
            return

        if parsed.path == "/api/summary":
            if not session.fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            self._send_json({"ok": True, **_summary_payload(session)})
            return

        if parsed.path == "/api/load_progress":
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.load_running),
                    "progress": float(session.load_progress),
                    "message": str(session.load_message or ""),
                    "sample_done": int(session.load_sample_done),
                    "sample_total": int(session.load_sample_total),
                }
            )
            return

        if parsed.path == "/api/preview_progress":
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.preview_running),
                    "progress": float(session.preview_progress),
                    "message": str(session.preview_message or ""),
                    "frame_done": int(session.preview_frame_done),
                    "frame_total": int(session.preview_frame_total),
                }
            )
            return

        if parsed.path == "/api/load_review":
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.load_running),
                    "review": _build_partial_review_payload(session, include_images=True),
                }
            )
            return

        self._send_error_json("Not found", code=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        session = self._ensure_session()
        parsed = urlparse(self.path)

        try:
            payload = self._read_json()
        except Exception:
            self._send_error_json("Invalid JSON body")
            return

        if parsed.path == "/api/load_chapter":
            self._handle_load_chapter(session, payload)
            return

        if parsed.path == "/api/cancel_load":
            session.load_cancel_requested = True
            if session.load_running:
                _set_load_progress(
                    session,
                    message="Cancel requested... stopping after current frame.",
                )
            self._send_json(
                {
                    "ok": True,
                    "running": bool(session.load_running),
                    "message": (
                        "Cancel requested... stopping after current frame."
                        if session.load_running
                        else "No active load to cancel."
                    ),
                }
            )
            return

        if parsed.path == "/api/apply_iqr":
            if not session.fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            session.iqr_k = _normalize_iqr_k(payload.get("iqr_k", session.iqr_k), default=session.iqr_k)
            review = _build_review_payload(session, include_images=False)
            self._send_json({"ok": True, "review": review})
            return

        if parsed.path == "/api/toggle_frame":
            if not session.fids and not session.partial_fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            try:
                fid = int(payload.get("fid"))
            except Exception:
                self._send_error_json("Missing or invalid frame id.")
                return
            self._handle_toggle_frame(session, fid)
            return

        if parsed.path == "/api/set_frame_range":
            if not session.fids and not session.partial_fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            try:
                start_fid = int(payload.get("start_fid"))
                end_fid = int(payload.get("end_fid"))
            except Exception:
                self._send_error_json("Missing or invalid range frame ids.")
                return
            status_raw = str(payload.get("status", "bad") or "bad").strip().lower()
            status = "good" if status_raw == "good" else "bad"
            self._handle_set_frame_range(session, start_fid, end_fid, status)
            return

        if parsed.path == "/api/preview_render":
            self._handle_preview_render(session, payload)
            return

        if parsed.path == "/api/save":
            self._handle_save(session, payload)
            return

        if parsed.path == "/api/save_progress":
            self._handle_save_progress(session, payload)
            return

        self._send_error_json("Not found", code=HTTPStatus.NOT_FOUND)

    def _handle_load_chapter(self, session: SessionState, payload: dict[str, Any]) -> None:
        def fail(message: str) -> None:
            _set_load_progress(
                session,
                running=False,
                progress=0.0,
                message=str(message),
            )
            session.load_cancel_requested = False
            self._send_error_json(message)

        def cancelled() -> bool:
            if not bool(session.load_cancel_requested):
                return False
            fail("Load cancelled.")
            return True

        session.load_cancel_requested = False
        _set_load_progress(
            session,
            running=True,
            progress=1.0,
            message="Preparing chapter load...",
            sample_done=0,
            sample_total=0,
        )

        archive = str(payload.get("archive", session.archive) or "").strip()
        chapter = str(payload.get("chapter", session.chapter) or "").strip()
        if not archive or not chapter:
            fail("Archive and chapter are required.")
            return

        if cancelled():
            return
        _archive_state(session, archive, selected_title=chapter)
        chapter_obj = _find_chapter(session.chapters, chapter)
        if not chapter_obj:
            fail("Selected chapter was not found.")
            return

        default_start = int(chapter_obj.get("start_frame", 0))
        default_end = int(chapter_obj.get("end_frame", default_start + 1))

        try:
            start_raw = int(payload.get("start_frame", default_start))
            end_raw = int(payload.get("end_frame", default_end))
            sample_stride = max(1, int(payload.get("sample_stride", session.sample_stride)))
            context = max(0, int(payload.get("context", session.context)))
            iqr_k = _normalize_iqr_k(payload.get("iqr_k", session.iqr_k), default=session.iqr_k)
        except Exception:
            fail("Invalid numeric load settings.")
            return

        session.chapter = chapter
        session.start_frame, session.end_frame = _normalize_frame_span(start_raw, end_raw)
        session.sample_stride = sample_stride
        session.context = context
        session.strict_sampling = bool(payload.get("strict_sampling", session.strict_sampling))
        session.exact_extract = bool(payload.get("exact_extract", session.exact_extract))
        session.debug_extract = bool(payload.get("debug_extract", session.debug_extract))
        session.iqr_k = iqr_k
        session.preview_video_path = ""
        session.fids = []
        session.b64 = []
        session.sigs = {}
        session.overrides = _chapter_bad_overrides(
            archive=session.archive,
            chapter_title=session.chapter,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.gamma_default = 1.0
        session.gamma_ranges = []
        session.people_entries = []
        session.partial_fids = []
        session.partial_b64 = []
        session.partial_sigs = {"chroma": [], "noise": [], "tear": [], "wave": []}

        video = _resolve_archive_video(session.archive)
        if not video:
            fail(f"No archive video found for '{session.archive}'.")
            return

        if cancelled():
            return
        n_samp = sample_count_from_stride(session.start_frame, session.end_frame, session.sample_stride)
        _set_load_progress(
            session,
            progress=4.0,
            message=f"Target sampled frames: {int(n_samp)}",
            sample_done=0,
            sample_total=int(n_samp),
        )
        read_video = video
        frame_read_offset = 0

        debug_overlay = bool(session.debug_extract) or _env_truthy(TUNER_DEBUG_EXTRACT_ENV) or _env_truthy(
            RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV
        )

        if bool(session.exact_extract):
            _set_load_progress(session, progress=8.0, message="Extracting chapter segment...")
            if cancelled():
                return
            try:
                read_video_p, ex_err = _ensure_render_chapter_extract(
                    source_video=video,
                    archive=session.archive,
                    chapter_title=session.chapter,
                    ch_start=session.start_frame,
                    ch_end=session.end_frame,
                    debug_overlay=debug_overlay,
                )
            except Exception as exc:
                fail(f"Render extract failed: {type(exc).__name__}: {exc}")
                return
            if ex_err or read_video_p is None:
                fail(ex_err or "Render extract failed")
                return
            read_video = read_video_p
            frame_read_offset = session.start_frame
            _set_load_progress(session, progress=28.0, message="Chapter extract ready; sampling frames...")
        else:
            _set_load_progress(session, progress=12.0, message="Sampling source video frames...")

        if cancelled():
            return
        sample_target = max(1, int(n_samp))
        stage_start = 30.0 if bool(session.exact_extract) else 14.0
        stage_end = 92.0

        def _sample_progress(frac: float, desc: str | None = None) -> None:
            _ = desc
            try:
                f = float(frac)
            except Exception:
                f = 0.0
            f = max(0.0, min(1.0, f))
            done = max(0, min(sample_target, int(round(f * sample_target))))
            p = stage_start + f * (stage_end - stage_start)
            _set_load_progress(
                session,
                progress=p,
                message=f"Sampling frames {done}/{sample_target}",
                sample_done=done,
                sample_total=sample_target,
            )

        def _sample_frame(
            fid: int,
            frame_b64: str,
            chroma: float,
            noise: float,
            tear: float,
            wave: float,
            _done: int,
            _total: int,
        ) -> None:
            session.partial_fids.append(int(fid))
            session.partial_b64.append(str(frame_b64 or ""))
            session.partial_sigs["chroma"].append(float(chroma))
            session.partial_sigs["noise"].append(float(noise))
            session.partial_sigs["tear"].append(float(tear))
            session.partial_sigs["wave"].append(float(wave))

        fids, b64, sigs, err = extract_frames(
            str(read_video),
            session.start_frame,
            session.end_frame,
            int(n_samp),
            session.archive,
            session.chapter,
            frame_read_offset=frame_read_offset,
            progress=_sample_progress,
            should_cancel=lambda: bool(session.load_cancel_requested),
            frame_callback=_sample_frame,
        )
        if err or fids is None or b64 is None or sigs is None:
            fail(err or "Failed to extract frames.")
            return
        _set_load_progress(
            session,
            progress=95.0,
            message=f"Processing sampled frames {sample_target}/{sample_target}",
            sample_done=sample_target,
            sample_total=sample_target,
        )

        if not bool(session.strict_sampling):
            sc0 = combined_score(sigs, session.wc, session.wn, session.wt, session.ww)
            thr0 = compute_threshold(sc0, session.t_mode, session.iqr_k, session.tval, session.bpct)
            focus_fids = select_focus_frame_ids(
                start=session.start_frame,
                end=session.end_frame,
                max_frames=int(n_samp),
                coarse_fids=fids,
                coarse_scores=sc0,
                threshold=thr0,
                burst_radius=4,
            )
            if focus_fids != fids:
                session.partial_fids = []
                session.partial_b64 = []
                session.partial_sigs = {"chroma": [], "noise": [], "tear": [], "wave": []}
                fids, b64, sigs, err = extract_frames(
                    str(read_video),
                    session.start_frame,
                    session.end_frame,
                    int(n_samp),
                    session.archive,
                    session.chapter,
                    frame_ids=focus_fids,
                    frame_read_offset=frame_read_offset,
                    progress=_sample_progress,
                    should_cancel=lambda: bool(session.load_cancel_requested),
                    frame_callback=_sample_frame,
                )
                if err or fids is None or b64 is None or sigs is None:
                    fail(err or "Failed to extract focus frames.")
                    return

        session.fids = [int(x) for x in fids]
        session.b64 = list(b64)
        session.sigs = dict(sigs)
        gamma_profile = get_gamma_profile_for_chapter(
            archive=session.archive,
            chapter_title=session.chapter,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.gamma_default = _normalize_gamma_value(gamma_profile.get("default_gamma", 1.0), default=1.0)
        session.gamma_ranges = _normalize_gamma_ranges_payload(
            gamma_profile.get("ranges", []),
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )
        session.people_entries = _load_people_entries_for_chapter(
            archive=session.archive,
            ch_start=session.start_frame,
            ch_end=session.end_frame,
        )

        details = {
            "archive": session.archive,
            "chapter": session.chapter,
            "start_frame": session.start_frame,
            "end_frame": session.end_frame,
            "chapter_frame_count": int(session.end_frame - session.start_frame),
            "sampled_count": int(len(session.fids)),
            "sample_stride": session.sample_stride,
            "context": session.context,
            "exact_extract": bool(session.exact_extract),
            "strict_sampling": bool(session.strict_sampling),
            "extract_cache": str(
                _chapter_extract_cache_path(
                    archive=session.archive,
                    chapter_title=session.chapter,
                    ch_start=session.start_frame,
                    ch_end=session.end_frame,
                    debug_overlay=debug_overlay,
                    source_video=video,
                ).parent.name
            )
            if bool(session.exact_extract)
            else "",
            "gamma_profile": {
                "default_gamma": float(session.gamma_default),
                "ranges": list(session.gamma_ranges),
                "source": str(gamma_profile.get("source", "default")),
            },
            "people_profile": {
                "entries": list(session.people_entries),
                "source": "people_tsv",
            },
        }

        review = _build_review_payload(session, include_images=True)
        _set_load_progress(
            session,
            running=False,
            progress=100.0,
            message=f"Loaded {len(session.fids)} frame(s).",
            sample_done=len(session.fids),
            sample_total=max(1, int(n_samp)),
        )
        session.load_cancel_requested = False
        self._send_json({"ok": True, "review": review, "settings": details})

    def _handle_toggle_frame(self, session: SessionState, fid: int) -> None:
        fid_i = int(fid)
        final_ids = {int(x) for x in session.fids}
        partial_ids = {int(x) for x in session.partial_fids}
        if fid_i not in final_ids and fid_i not in partial_ids:
            self._send_error_json("Frame is not in the sampled set.")
            return

        if session.fids and session.sigs and fid_i in final_ids:
            scores = combined_score(session.sigs, session.wc, session.wn, session.wt, session.ww)
            thr = float(compute_threshold(scores, session.t_mode, session.iqr_k, session.tval, session.bpct))
            index = {int(x): i for i, x in enumerate(session.fids)}
            pos = index[fid_i]
            score = float(scores[pos])
            effective, _src = _frame_status(session, fid_i, score, thr)
        else:
            partial_review = _build_partial_review_payload(session, include_images=False)
            current = next((f for f in partial_review["frames"] if int(f["fid"]) == fid_i), None)
            if not current:
                self._send_error_json("Frame is not available yet.")
                return
            effective = "bad" if str(current.get("status")) == "bad" else "good"

        session.overrides[fid_i] = "good" if effective == "bad" else "bad"

        if session.fids and session.sigs:
            frame_state = _build_review_payload(session, include_images=False)
        else:
            frame_state = _build_partial_review_payload(session, include_images=False)
        updated = next((f for f in frame_state["frames"] if int(f["fid"]) == fid_i), None)
        self._send_json({"ok": True, "frame": updated, "review": frame_state})

    def _handle_set_frame_range(self, session: SessionState, start_fid: int, end_fid: int, status: str) -> None:
        lo = int(min(int(start_fid), int(end_fid)))
        hi = int(max(int(start_fid), int(end_fid)))
        target_status = "good" if str(status).strip().lower() == "good" else "bad"
        if session.fids and session.sigs:
            current = _build_review_payload(session, include_images=False)
        else:
            current = _build_partial_review_payload(session, include_images=False)

        changed = 0
        for frame in list(current.get("frames", [])):
            try:
                fid_i = int(frame.get("fid"))
            except Exception:
                continue
            if fid_i < lo or fid_i > hi:
                continue
            session.overrides[fid_i] = target_status
            changed += 1

        if changed <= 0:
            self._send_error_json("No sampled frames are currently available in that range.")
            return

        if session.fids and session.sigs:
            review = _build_review_payload(session, include_images=False)
        else:
            review = _build_partial_review_payload(session, include_images=False)
        self._send_json(
            {
                "ok": True,
                "review": review,
                "range": {"start_fid": lo, "end_fid": hi},
                "status": target_status,
                "updated_count": int(changed),
            }
        )

    def _run_cmd(self, cmd: list[Any], label: str) -> tuple[bool, str]:
        proc = subprocess.run(
            [str(x) for x in cmd],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return True, ""
        detail = (proc.stderr or proc.stdout or "").strip()
        if not detail:
            detail = f"{label} failed with exit code {int(proc.returncode)}."
        else:
            detail = f"{label} failed: {detail}"
        return False, detail

    def _run_cmd_with_progress(
        self,
        cmd: list[Any],
        label: str,
        *,
        on_frame: Any | None = None,
    ) -> tuple[bool, str]:
        parts = [str(x) for x in cmd]
        if parts:
            ffmpeg_name = Path(parts[0]).name.lower()
            if ffmpeg_name in {"ffmpeg", "ffmpeg.exe"} and "-progress" not in parts:
                parts = [
                    parts[0],
                    "-progress",
                    "pipe:2",
                    "-stats_period",
                    "0.5",
                    *parts[1:],
                ]

        proc = subprocess.Popen(
            parts,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        err_lines: list[str] = []
        try:
            if proc.stderr is not None:
                for raw in proc.stderr:
                    line = str(raw or "").strip()
                    if line:
                        err_lines.append(line)
                        if len(err_lines) > 80:
                            err_lines = err_lines[-80:]
                    m = re.match(r"^frame\s*=\s*(\d+)$", line)
                    if m and on_frame is not None:
                        try:
                            on_frame(int(m.group(1)))
                        except Exception:
                            pass
        finally:
            rc = proc.wait()

        if rc == 0:
            return True, ""
        detail = "\n".join(err_lines).strip()
        if not detail:
            detail = f"{label} failed with exit code {int(rc)}."
        else:
            detail = f"{label} failed: {detail}"
        return False, detail

    def _handle_preview_render(self, session: SessionState, payload: dict[str, Any] | None = None) -> None:
        def fail(message: str) -> None:
            _set_preview_progress(
                session,
                running=False,
                progress=0.0,
                message=str(message),
            )
            self._send_error_json(message)

        if not session.fids or not session.sigs:
            fail("No loaded chapter data yet.")
            return
        if not session.archive or not session.chapter:
            fail("Archive and chapter context are missing.")
            return

        payload = payload or {}
        preview_mode = str(payload.get("preview_mode", "") or "").strip().lower()
        apply_freeze = True
        apply_gamma = True
        if preview_mode == "review":
            apply_gamma = False
        elif preview_mode == "gamma":
            apply_freeze = False
        elif preview_mode == "summary":
            apply_freeze = True
            apply_gamma = True

        raw_gamma_profile = payload.get("gamma_profile")
        if raw_gamma_profile is None:
            raw_gamma_profile = payload.get("gamma")
        if isinstance(raw_gamma_profile, dict):
            session.gamma_default = _normalize_gamma_value(
                raw_gamma_profile.get("default_gamma", session.gamma_default),
                default=session.gamma_default,
            )
            session.gamma_ranges = _normalize_gamma_ranges_payload(
                raw_gamma_profile.get("ranges", session.gamma_ranges),
                ch_start=session.start_frame,
                ch_end=session.end_frame,
            )

        try:
            from vhs_pipeline.render_pipeline import (
                BADFRAME_SOURCE_CLEARANCE,
                assert_expected_frame_count,
                local_bad_frames_to_repairs,
                make_create_avs,
                make_deinterlace,
                make_deinterlace_ffmpeg_fallback,
                make_freeze_only_avs,
                make_gamma_only_avs,
                make_render_avs_ffv1,
            )
        except Exception as exc:
            fail(f"Preview render is unavailable: {type(exc).__name__}: {exc}")
            return

        proxy_video = ARCHIVE_DIR / f"{session.archive}_proxy.mp4"
        archive_video = ARCHIVE_DIR / f"{session.archive}.mkv"
        if proxy_video.exists():
            source_video = proxy_video
            source_label = "proxy"
        elif archive_video.exists():
            source_video = archive_video
            source_label = "archive (proxy missing)"
        else:
            fail(f"No source video found for '{session.archive}'.")
            return

        start_frame, end_frame = _normalize_frame_span(session.start_frame, session.end_frame)
        chapter_len = max(1, int(end_frame) - int(start_frame))
        debug_overlay = bool(session.debug_extract) or _env_truthy(TUNER_DEBUG_EXTRACT_ENV) or _env_truthy(
            RENDER_DEBUG_EXTRACT_FRAME_NUMBERS_ENV
        )
        extracted, ex_err = _ensure_render_chapter_extract(
            source_video=source_video,
            archive=session.archive,
            chapter_title=session.chapter,
            ch_start=start_frame,
            ch_end=end_frame,
            debug_overlay=debug_overlay,
        )
        if ex_err or extracted is None:
            fail(ex_err or "Failed to extract preview chapter segment.")
            return

        bad_global = [
            int(fid)
            for fid in _selected_bad_frame_ids(session)
            if int(start_frame) <= int(fid) < int(end_frame)
        ]
        local_bad = [int(fid) - int(start_frame) for fid in bad_global] if apply_freeze else []
        local_repairs = local_bad_frames_to_repairs(local_bad) if local_bad else []

        preview_root = PROJECT_ROOT / "tmp" / "plain_html_wizard_preview"
        preview_dir = preview_root / (
            f"{session.archive}__{slugify(session.chapter)}__{int(start_frame)}_{int(end_frame)}"
        )
        preview_dir.mkdir(parents=True, exist_ok=True)

        freeze_avs = preview_dir / "freeze.avs"
        filter_avs = preview_dir / "script.avs"
        repaired_extracted = preview_dir / "repaired_extracted.mkv"
        qtgmc = preview_dir / "qtgmc.mkv"
        preview_video = preview_dir / "preview_render.mp4"

        filter_script = METADATA_DIR / session.archive / "filter.avs"
        chapter_filter_script = METADATA_DIR / session.archive / f"{session.chapter}.avs"
        if chapter_filter_script.exists():
            filter_script = chapter_filter_script

        freeze_input = extracted
        used_non_windows_fallback = False
        gamma_only_mode = preview_mode == "gamma"
        windows_filter = bool(
            sys.platform == "win32"
            and apply_gamma
            and (gamma_only_mode or filter_script.exists())
        )
        windows_freeze = bool(sys.platform == "win32" and bool(local_bad))
        stage_names: list[str] = []
        if windows_freeze:
            stage_names.append("Applying FreezeFrame repairs")
        if windows_filter:
            stage_names.append("Applying gamma correction" if gamma_only_mode else "Deinterlacing/filtering")
        elif sys.platform != "win32":
            stage_names.append("Fallback deinterlacing")
        stage_names.append("Encoding preview")
        total_stages = max(1, len(stage_names))
        total_frames_all = max(1, chapter_len * total_stages)

        _set_preview_progress(
            session,
            running=True,
            progress=1.0,
            message="Preparing preview render...",
            frame_done=0,
            frame_total=total_frames_all,
        )

        def _set_stage_progress(stage_idx: int, frame_done: int, stage_label: str) -> None:
            done = max(0, min(chapter_len, int(frame_done)))
            overall_done = min(total_frames_all, (stage_idx * chapter_len) + done)
            frac = float(done) / float(max(1, chapter_len))
            pct = ((float(stage_idx) + frac) / float(total_stages)) * 100.0
            _set_preview_progress(
                session,
                running=True,
                progress=max(1.0, min(99.5, pct)),
                message=f"{stage_label}... ({done}/{chapter_len} frames)",
                frame_done=overall_done,
                frame_total=total_frames_all,
            )

        stage_idx = 0

        try:
            if sys.platform == "win32":
                if local_bad:
                    stage_label = stage_names[stage_idx]
                    freeze_script = make_freeze_only_avs(
                        extracted,
                        bad_source_frames=local_bad,
                        bad_repair_ranges=local_repairs,
                        chapter_start_frame=start_frame,
                        chapter_end_frame=end_frame,
                        source_clearance=BADFRAME_SOURCE_CLEARANCE,
                    )
                    freeze_avs.write_text(freeze_script, encoding="ascii")
                    ok, detail = self._run_cmd_with_progress(
                        make_render_avs_ffv1(freeze_avs, extracted, repaired_extracted),
                        "Preview freeze stage",
                        on_frame=lambda n: _set_stage_progress(stage_idx, n, stage_label),
                    )
                    if not ok:
                        fail(detail)
                        return
                    _set_stage_progress(stage_idx, chapter_len, stage_label)
                    stage_idx += 1
                    assert_expected_frame_count(
                        repaired_extracted,
                        chapter_len,
                        f"preview repaired chapter '{session.chapter}'",
                    )
                    freeze_input = repaired_extracted

                if windows_filter:
                    stage_label = stage_names[stage_idx]
                    gamma_default = _normalize_gamma_value(session.gamma_default, default=1.0)
                    gamma_ranges = _normalize_gamma_ranges_payload(
                        session.gamma_ranges,
                        ch_start=start_frame,
                        ch_end=end_frame,
                    )
                    if gamma_only_mode:
                        script_text = make_gamma_only_avs(
                            freeze_input,
                            chapter_start_frame=start_frame,
                            chapter_end_frame=end_frame,
                            gamma_default=gamma_default,
                            gamma_ranges=gamma_ranges,
                        )
                    else:
                        script_text = make_create_avs(
                            freeze_input,
                            filter_script,
                            bad_source_frames=[],
                            bad_repair_ranges=[],
                            chapter_start_frame=start_frame,
                            chapter_end_frame=end_frame,
                            gamma_default=gamma_default,
                            gamma_ranges=gamma_ranges,
                            no_bob=False,
                            source_clearance=0,
                        )
                    filter_avs.write_text(script_text, encoding="ascii")
                    stage_cmd_label = "Preview gamma stage" if gamma_only_mode else "Preview deinterlace stage"
                    ok, detail = self._run_cmd_with_progress(
                        make_deinterlace(filter_avs, freeze_input, qtgmc),
                        stage_cmd_label,
                        on_frame=lambda n: _set_stage_progress(stage_idx, n, stage_label),
                    )
                    if not ok:
                        fail(detail)
                        return
                    _set_stage_progress(stage_idx, chapter_len, stage_label)
                    stage_idx += 1
                else:
                    shutil.copy2(freeze_input, qtgmc)
            else:
                used_non_windows_fallback = True
                stage_label = stage_names[stage_idx]
                ok, detail = self._run_cmd_with_progress(
                    make_deinterlace_ffmpeg_fallback(extracted, qtgmc, no_bob=False),
                    "Preview fallback deinterlace stage",
                    on_frame=lambda n: _set_stage_progress(stage_idx, n, stage_label),
                )
                if not ok:
                    fail(detail)
                    return
                _set_stage_progress(stage_idx, chapter_len, stage_label)
                stage_idx += 1

            assert_expected_frame_count(
                qtgmc,
                chapter_len,
                f"preview qtgmc chapter '{session.chapter}'",
            )
        except Exception as exc:
            fail(f"Preview render failed: {type(exc).__name__}: {exc}")
            return

        stage_label = stage_names[stage_idx if stage_idx < len(stage_names) else (len(stage_names) - 1)]
        ok, detail = self._run_cmd_with_progress(
            [
                FFMPEG_BIN,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(qtgmc),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode:v:0",
                "passthrough",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-ar",
                "48000",
                "-ac",
                "1",
                "-movflags",
                "+faststart",
                "-y",
                str(preview_video),
            ],
            "Preview encode stage",
            on_frame=lambda n: _set_stage_progress(stage_idx, n, stage_label),
        )
        if not ok:
            fail(detail)
            return
        _set_stage_progress(stage_idx, chapter_len, stage_label)

        session.preview_video_path = str(preview_video.resolve())
        mode_desc = preview_mode if preview_mode in {"review", "gamma", "people", "summary"} else "combined"
        msg = (
            f"Preview render ready for {session.chapter}: "
            f"mode={mode_desc}, freeze={'on' if apply_freeze else 'off'}, gamma={'on' if apply_gamma else 'off'}. "
            f"{len(local_bad)} sampled bad frame(s) applied from current review state. "
            f"Source: {source_label}."
        )
        if used_non_windows_fallback and local_bad:
            msg += " Note: non-Windows fallback cannot apply AviSynth FreezeFrame repair logic."
        _set_preview_progress(
            session,
            running=False,
            progress=100.0,
            message="Preview render complete.",
            frame_done=total_frames_all,
            frame_total=total_frames_all,
        )
        self._send_json(
            {
                "ok": True,
                "message": msg,
                "preview_path": str(preview_video),
                "preview_url": "/api/preview_video",
                "preview_page_url": "/preview",
                "bad_sampled_count": int(len(local_bad)),
            }
        )

    def _handle_save(self, session: SessionState, payload: dict[str, Any] | None = None) -> None:
        if not session.fids:
            self._send_error_json("No loaded chapter data yet.")
            return

        _apply_profiles_from_payload(session, payload)
        try:
            out_path, gamma_path, count, analyzed, people_count = _persist_session_progress(session)
        except Exception as exc:
            self._send_error_json(str(exc))
            return
        gamma_count = len(session.gamma_ranges)
        people_path = METADATA_DIR / str(session.archive or "").strip() / "people.tsv"

        archive_state = _archive_state(session, session.archive, selected_title=session.chapter)
        self._send_json(
            {
                "ok": True,
                "message": (
                    f"Saved BAD_FRAMES for {session.chapter} "
                    f"({int(analyzed)} analyzed, {int(count)} bad). "
                    f"Saved gamma ranges: {int(gamma_count)}. "
                    f"Saved people entries: {int(people_count)}."
                ),
                "metadata_path": str(gamma_path or out_path or people_path) if (gamma_path or out_path or people_path) else "",
                "archive_state": archive_state,
            }
        )

    def _handle_save_progress(self, session: SessionState, payload: dict[str, Any] | None = None) -> None:
        if not session.fids:
            self._send_error_json("No loaded chapter data yet.")
            return
        _apply_profiles_from_payload(session, payload)
        try:
            out_path, gamma_path, count, analyzed, people_count = _persist_session_progress(session)
        except Exception as exc:
            self._send_error_json(str(exc))
            return
        self._send_json(
            {
                "ok": True,
                "message": (
                    f"Progress saved for {session.chapter}: "
                    f"BAD_FRAMES {int(count)}/{int(analyzed)}, "
                    f"gamma ranges {int(len(session.gamma_ranges))}, "
                    f"people entries {int(people_count)}."
                ),
                "metadata_path": str(gamma_path or out_path or (METADATA_DIR / str(session.archive or '').strip() / 'people.tsv')),
            }
        )


def run(host: str = "0.0.0.0", port: int = 8092) -> None:
    server = ThreadingHTTPServer((host, int(port)), WizardHandler)
    print(f"VHS Tuner running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
