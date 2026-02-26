from pathlib import Path

import numpy as np

from apps.plain_html_wizard.server import (
    SessionState,
    _build_review_payload,
    _normalize_iqr_k,
    _set_load_progress,
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

    assert 'id="iqrK" type="range" min="0" max="12"' in html
    assert 'id="iqrSpark"' in html
    assert 'id="toggleFullscreen"' in html
    assert "grid-template-rows: auto auto minmax(0, 1fr) auto;" in html
    assert "iqrEl.addEventListener('input'" in html
    assert "iqrEl.addEventListener('change'" in html
    assert "scheduleAutoApplyIqr()" in html
    assert "scheduleVisibleRangeRefresh()" in html
    assert "frameGridEl.addEventListener('scroll'" in html
    assert 'id="overlayProgressFill"' in html
    assert 'id="overlayProgressText"' in html
    assert 'id="overlayCancelBtn"' in html
    assert "startLoadProgress(" in html
    assert "finishLoadProgress(" in html
    assert "pollLoadProgressOnce()" in html
    assert "api('/api/load_progress')" in html
    assert "api('/api/cancel_load', 'POST', {})" in html
    assert "seekFrameGridFromSparkClientX(" in html
    assert "queueSparkDragSeek(" in html
    assert "iqrSparkEl.addEventListener('pointerdown'" in html
    assert "iqrSparkEl.addEventListener('pointermove'" in html
    assert "frameGridEl.scrollTo({ top, behavior: 'auto' })" in html
    assert "const solarizedOrange = '#cb4b16';" in html
    assert 'clipPath id="sparkAboveThresholdClip"' in html
    assert "frame.status === 'bad'" in html


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
