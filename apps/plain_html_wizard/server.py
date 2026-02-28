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

from common import ARCHIVE_DIR, FFMPEG_BIN, METADATA_DIR, combined_score, compute_threshold
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
    preview_video_path: str = ""


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


def _normalize_iqr_k(raw: Any, default: float = 3.5) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(0.0, min(12.0, float(value)))


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
        f"Manual overrides: {review['stats']['overrides']}\n\n"
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
            "body{margin:0;background:#111;color:#eee;font-family:Segoe UI,Arial,sans-serif;}"
            ".wrap{display:grid;grid-template-rows:auto 1fr;height:100vh;gap:8px;padding:10px;box-sizing:border-box;}"
            ".meta{font-size:13px;opacity:.9;}"
            "video{width:100%;height:100%;background:#000;border:1px solid #333;border-radius:8px;}"
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
            if not session.fids:
                self._send_error_json("No loaded chapter data yet.")
                return
            try:
                fid = int(payload.get("fid"))
            except Exception:
                self._send_error_json("Missing or invalid frame id.")
                return
            self._handle_toggle_frame(session, fid)
            return

        if parsed.path == "/api/preview_render":
            self._handle_preview_render(session)
            return

        if parsed.path == "/api/save":
            self._handle_save(session)
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
                )
                if err or fids is None or b64 is None or sigs is None:
                    fail(err or "Failed to extract focus frames.")
                    return

        session.fids = [int(x) for x in fids]
        session.b64 = list(b64)
        session.sigs = dict(sigs)
        session.overrides = _chapter_bad_overrides(
            archive=session.archive,
            chapter_title=session.chapter,
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
                ).parent.name
            )
            if bool(session.exact_extract)
            else "",
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
        if int(fid) not in {int(x) for x in session.fids}:
            self._send_error_json("Frame is not in the sampled set.")
            return

        scores = combined_score(session.sigs, session.wc, session.wn, session.wt, session.ww)
        thr = float(compute_threshold(scores, session.t_mode, session.iqr_k, session.tval, session.bpct))
        index = {int(x): i for i, x in enumerate(session.fids)}
        pos = index[int(fid)]
        score = float(scores[pos])
        effective, _src = _frame_status(session, int(fid), score, thr)
        session.overrides[int(fid)] = "good" if effective == "bad" else "bad"

        frame_state = _build_review_payload(session, include_images=False)
        updated = next((f for f in frame_state["frames"] if int(f["fid"]) == int(fid)), None)
        self._send_json({"ok": True, "frame": updated, "review": frame_state})

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

    def _handle_preview_render(self, session: SessionState) -> None:
        if not session.fids or not session.sigs:
            self._send_error_json("No loaded chapter data yet.")
            return
        if not session.archive or not session.chapter:
            self._send_error_json("Archive and chapter context are missing.")
            return

        try:
            from vhs_pipeline.render_pipeline import (
                BADFRAME_SOURCE_CLEARANCE,
                assert_expected_frame_count,
                local_bad_frames_to_repairs,
                make_create_avs,
                make_deinterlace,
                make_deinterlace_ffmpeg_fallback,
                make_freeze_only_avs,
                make_render_avs_ffv1,
            )
        except Exception as exc:
            self._send_error_json(f"Preview render is unavailable: {type(exc).__name__}: {exc}")
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
            self._send_error_json(f"No source video found for '{session.archive}'.")
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
            self._send_error_json(ex_err or "Failed to extract preview chapter segment.")
            return

        bad_global = [
            int(fid)
            for fid in _selected_bad_frame_ids(session)
            if int(start_frame) <= int(fid) < int(end_frame)
        ]
        local_bad = [int(fid) - int(start_frame) for fid in bad_global]
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

        try:
            if sys.platform == "win32":
                if local_bad:
                    freeze_script = make_freeze_only_avs(
                        extracted,
                        bad_source_frames=local_bad,
                        bad_repair_ranges=local_repairs,
                        chapter_start_frame=start_frame,
                        chapter_end_frame=end_frame,
                        source_clearance=BADFRAME_SOURCE_CLEARANCE,
                    )
                    freeze_avs.write_text(freeze_script, encoding="ascii")
                    ok, detail = self._run_cmd(
                        make_render_avs_ffv1(freeze_avs, extracted, repaired_extracted),
                        "Preview freeze stage",
                    )
                    if not ok:
                        self._send_error_json(detail)
                        return
                    assert_expected_frame_count(
                        repaired_extracted,
                        chapter_len,
                        f"preview repaired chapter '{session.chapter}'",
                    )
                    freeze_input = repaired_extracted

                if filter_script.exists():
                    script_text = make_create_avs(
                        freeze_input,
                        filter_script,
                        bad_source_frames=[],
                        bad_repair_ranges=[],
                        chapter_start_frame=start_frame,
                        chapter_end_frame=end_frame,
                        no_bob=False,
                        source_clearance=0,
                    )
                    filter_avs.write_text(script_text, encoding="ascii")
                    ok, detail = self._run_cmd(
                        make_deinterlace(filter_avs, freeze_input, qtgmc),
                        "Preview deinterlace stage",
                    )
                    if not ok:
                        self._send_error_json(detail)
                        return
                else:
                    shutil.copy2(freeze_input, qtgmc)
            else:
                used_non_windows_fallback = True
                ok, detail = self._run_cmd(
                    make_deinterlace_ffmpeg_fallback(extracted, qtgmc, no_bob=False),
                    "Preview fallback deinterlace stage",
                )
                if not ok:
                    self._send_error_json(detail)
                    return

            assert_expected_frame_count(
                qtgmc,
                chapter_len,
                f"preview qtgmc chapter '{session.chapter}'",
            )
        except Exception as exc:
            self._send_error_json(f"Preview render failed: {type(exc).__name__}: {exc}")
            return

        ok, detail = self._run_cmd(
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
        )
        if not ok:
            self._send_error_json(detail)
            return

        session.preview_video_path = str(preview_video.resolve())
        msg = (
            f"Preview render ready for {session.chapter}: "
            f"{len(local_bad)} sampled bad frame(s) applied from current review state. "
            f"Source: {source_label}."
        )
        if used_non_windows_fallback and local_bad:
            msg += " Note: non-Windows fallback cannot apply AviSynth FreezeFrame repair logic."
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

    def _handle_save(self, session: SessionState) -> None:
        if not session.fids:
            self._send_error_json("No loaded chapter data yet.")
            return

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
            self._send_error_json(err)
            return

        archive_state = _archive_state(session, session.archive, selected_title=session.chapter)
        self._send_json(
            {
                "ok": True,
                "message": (
                    f"Saved BAD_FRAMES for {session.chapter} "
                    f"({int(analyzed)} analyzed, {int(count)} bad)."
                ),
                "metadata_path": str(out_path) if out_path else "",
                "archive_state": archive_state,
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
