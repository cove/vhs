from pathlib import Path
import shutil

import numpy as np

from common import update_chapter_bad_frames_in_render_settings
from libs.vhs_tuner_core import _chapter_bad_overrides
from apps.plain_html_wizard.server import (
    SessionState,
    _build_review_payload,
    _build_partial_review_payload,
    _normalize_iqr_k,
    _set_load_progress,
    WizardHandler,
)


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "apps" / "plain_html_wizard" / "static" / "index.html"


def _make_session() -> SessionState:
    n = 120
    fids = list(range(1000, 1000 + n))
    chroma = np.linspace(0.0, 1.0, n, dtype=np.float64)
    chroma[-8:] += np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    sigs = {
        "chroma": chroma,
        "noise": np.zeros(n, dtype=np.float64),
        "tear": np.zeros(n, dtype=np.float64),
        "wave": np.zeros(n, dtype=np.float64),
    }
    return SessionState(fids=fids, sigs=sigs, overrides={})


class _HandlerStub:
    def __init__(self) -> None:
        self.payload = None
        self.error = None

    def _send_json(self, payload, code=200) -> None:
        _ = code
        self.payload = payload

    def _send_error_json(self, message, code=400) -> None:
        _ = code
        self.error = str(message)


def test_normalize_iqr_k_clamps_and_parses() -> None:
    assert _normalize_iqr_k(-1) == 0.0
    assert _normalize_iqr_k(99) == 12.0
    assert _normalize_iqr_k("2.75") == 2.75
    assert _normalize_iqr_k("bad", default=4.2) == 4.2


def test_review_payload_reprocesses_bad_counts_when_iqr_changes() -> None:
    session = _make_session()
    session.iqr_k = 12.0
    high_k = _build_review_payload(session, include_images=False)

    session.iqr_k = 0.0
    low_k = _build_review_payload(session, include_images=False)

    assert low_k["threshold"] <= high_k["threshold"]
    assert low_k["stats"]["bad"] >= high_k["stats"]["bad"]
    assert low_k["stats"]["bad"] > high_k["stats"]["bad"]
    assert len(low_k["frames"]) == len(session.fids)


def test_review_payload_honors_manual_overrides_after_iqr_change() -> None:
    session = _make_session()
    target_fid = session.fids[-1]
    session.overrides = {int(target_fid): "good"}
    session.iqr_k = 0.0

    review = _build_review_payload(session, include_images=False)
    frame = next(f for f in review["frames"] if int(f["fid"]) == int(target_fid))

    assert frame["status"] == "good"
    assert frame["source"] == "MG"


def test_static_html_contains_live_iqr_spark_and_fullscreen_controls() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'id="helpBtn"' in html
    assert 'id="helpModal"' in html
    assert 'id="helpCloseBtn"' in html
    assert "openHelpModal()" in html
    assert "closeHelpModal()" in html
    assert "helpBtnEl.addEventListener('click', openHelpModal)" in html
    assert "event.key === 'Escape'" in html
    assert 'id="stepPills"' in html
    assert "const STEP_DEFS = [" in html
    assert "const STEP_MODE_TO_FIRST = new Map();" in html
    assert "const navActionButtons = new Map();" in html
    assert "setStepByMode(" in html
    assert "setStep(stepAtOffset(state.wizardStep, -1).num);" in html
    assert "bindWizardNavigation()" in html

    assert 'id="iqrK" type="range" min="0" max="12"' in html
    assert 'id="gammaLevel" type="range" min="0.05" max="8.00"' in html
    assert 'id="gammaMode"' in html
    assert 'id="nextToGamma"' in html
    assert 'id="sparkPlayBtn"' in html
    assert 'id="iqrSpark"' in html
    assert 'id="toggleFullscreen"' in html
    assert 'id="previewRender"' in html
    assert "grid-template-rows: auto auto minmax(0, 1fr) auto;" in html
    assert "iqrEl.addEventListener('input'" in html
    assert "iqrEl.addEventListener('change'" in html
    assert "scheduleAutoApplyIqr()" in html
    assert "scheduleVisibleRangeRefresh()" in html
    assert "frameGridEl.addEventListener('scroll'" in html
    assert "normalizeWheelToPixels(" in html
    assert "window.addEventListener('wheel', relayWheelToFrameGrid, { passive: false, capture: true })" in html
    assert "frameGridEl.scrollBy({ top: deltaPx * 1.75, behavior: 'auto' })" in html
    assert 'id="overlayProgressFill"' in html
    assert 'id="overlayProgressText"' in html
    assert 'id="overlayEtaText"' in html
    assert 'id="overlayCancelBtn"' in html
    assert "startLoadProgress(" in html
    assert "finishLoadProgress(" in html
    assert "pollLoadProgressOnce()" in html
    assert "api('/api/load_progress')" in html
    assert "api('/api/preview_render', 'POST'" in html
    assert "renderReadyAtFromSamples(" in html
    assert "Estimated ready at" in html
    assert "20 frame sample" in html
    assert "api('/api/cancel_load', 'POST', {})" in html
    assert "seekFrameGridFromSparkClientX(" in html
    assert "queueSparkDragSeek(" in html
    assert "toggleSparkWindowPlayback(" in html
    assert "window.setInterval(stepSparkWindowRight, 100)" in html
    assert "sparkPlayBtnEl.addEventListener('click', openFlipbookPanel)" in html
    assert "iqrSparkEl.addEventListener('pointerdown'" in html
    assert "iqrSparkEl.addEventListener('pointermove'" in html
    assert "frameGridEl.scrollTo({ top, behavior: 'auto' })" in html
    assert "const sparkThreshold = themeVar('--spark-threshold', '#ff646e');" in html
    assert 'clipPath id="sparkAboveThresholdClip"' in html
    assert "frame.status === 'bad'" in html
    assert "event.target.closest('.frame-card')" in html
    assert "window.open(target, '_blank', 'noopener')" in html
    assert "Toggle Good/Bad" not in html


def test_set_load_progress_updates_and_clamps_state() -> None:
    session = SessionState()
    _set_load_progress(
        session,
        running=True,
        progress=133.0,
        message="Sampling frames",
        sample_done=42,
        sample_total=100,
    )
    assert session.load_running is True
    assert session.load_progress == 100.0
    assert session.load_message == "Sampling frames"
    assert session.load_sample_done == 42
    assert session.load_sample_total == 100

    _set_load_progress(session, running=False, progress=-7.0, sample_done=-2, sample_total=-9)
    assert session.load_running is False
    assert session.load_progress == 0.0
    assert session.load_sample_done == 0
    assert session.load_sample_total == 0


def test_chapter_bad_overrides_load_saved_bad_frames_from_render_settings() -> None:
    archive = "__plain_wizard_unit_archive"
    title = "Unit Chapter"
    archive_meta_dir = ROOT / "metadata" / archive
    try:
        update_chapter_bad_frames_in_render_settings(
            archive,
            {title: [100, 105, 205]},
        )
        overrides = _chapter_bad_overrides(
            archive=archive,
            chapter_title=title,
            ch_start=100,
            ch_end=200,
        )
        assert overrides == {100: "bad", 105: "bad"}
    finally:
        shutil.rmtree(archive_meta_dir, ignore_errors=True)


def test_toggle_frame_allows_partial_frames_before_full_load() -> None:
    session = SessionState(
        start_frame=1000,
        partial_fids=[1000, 1001, 1002],
        partial_b64=["", "", ""],
        partial_sigs={
            "chroma": [0.1, 10.0, 0.1],
            "noise": [0.0, 0.0, 0.0],
            "tear": [0.0, 0.0, 0.0],
            "wave": [0.0, 0.0, 0.0],
        },
    )
    review_before = _build_partial_review_payload(session, include_images=False)
    frame_before = next(f for f in review_before["frames"] if int(f["fid"]) == 1001)
    status_before = str(frame_before["status"])

    handler = _HandlerStub()
    WizardHandler._handle_toggle_frame(handler, session, 1001)

    assert handler.error is None
    assert handler.payload is not None
    frame_after = next(f for f in handler.payload["review"]["frames"] if int(f["fid"]) == 1001)
    assert str(frame_after["status"]) != status_before
    assert session.overrides[1001] == ("good" if status_before == "bad" else "bad")


def test_set_frame_range_marks_partial_frames_bad() -> None:
    session = SessionState(
        start_frame=1000,
        partial_fids=[1000, 1001, 1002, 1003, 1004],
        partial_b64=["", "", "", "", ""],
        partial_sigs={
            "chroma": [0.1, 0.2, 0.3, 0.4, 0.5],
            "noise": [0.0, 0.0, 0.0, 0.0, 0.0],
            "tear": [0.0, 0.0, 0.0, 0.0, 0.0],
            "wave": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
    handler = _HandlerStub()
    WizardHandler._handle_set_frame_range(handler, session, 1001, 1003, "bad")

    assert handler.error is None
    assert handler.payload is not None
    assert int(handler.payload.get("updated_count", 0)) == 3
    for fid in (1001, 1002, 1003):
        assert session.overrides[fid] == "bad"
