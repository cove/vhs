from pathlib import Path

import numpy as np

import libs.vhs_tuner_core as core


def _sample_sigs() -> dict[str, np.ndarray]:
    return {
        "chroma": np.asarray([0.1, 0.2], dtype=np.float64),
        "noise": np.asarray([1.1, 1.2], dtype=np.float64),
        "tear": np.asarray([2.1, 2.2], dtype=np.float64),
        "wave": np.asarray([3.1, 3.2], dtype=np.float64),
    }


def test_cached_signals_round_trip_and_chapter_span_invalidation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(core, "TUNER_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(core, "TUNER_FRAME_CACHE_DIR", tmp_path / "frame_samples")

    video = tmp_path / "archive.mkv"
    video.write_bytes(b"abc")
    fids = [100, 110]
    sigs = _sample_sigs()
    thumbs = {
        100: "data:image/jpeg;base64,AA",
        110: "data:image/jpeg;base64,BB",
    }

    core.save_cached_signals(
        archive="sample_archive",
        ch_title="Sample Chapter",
        video_path=video,
        start_frame=100,
        end_frame=200,
        frame_read_offset=0,
        fids=fids,
        sigs=sigs,
        thumbs_by_fid=thumbs,
    )

    got_fids, got_sigs, got_thumbs = core.load_cached_signals(
        "sample_archive",
        "Sample Chapter",
        video_path=video,
        start_frame=100,
        end_frame=200,
        frame_read_offset=0,
    )

    assert got_fids == fids
    assert got_sigs is not None
    assert np.allclose(got_sigs["chroma"], sigs["chroma"])
    assert np.allclose(got_sigs["noise"], sigs["noise"])
    assert np.allclose(got_sigs["tear"], sigs["tear"])
    assert np.allclose(got_sigs["wave"], sigs["wave"])
    assert got_thumbs is not None
    assert got_thumbs[100] == thumbs[100]
    assert got_thumbs[110] == thumbs[110]

    missed = core.load_cached_signals(
        "sample_archive",
        "Sample Chapter",
        video_path=video,
        start_frame=100,
        end_frame=201,
        frame_read_offset=0,
    )
    assert missed == (None, None, None)


def test_cache_keys_change_when_source_video_changes(tmp_path: Path) -> None:
    video = tmp_path / "source.mkv"
    video.write_bytes(b"a")

    signals_path_before = core._signals_cache_path(
        archive="sample_archive",
        ch_title="Sample Chapter",
        video_path=video,
        start_frame=100,
        end_frame=200,
        frame_read_offset=0,
    )
    extract_path_before = core._chapter_extract_cache_path(
        archive="sample_archive",
        chapter_title="Sample Chapter",
        ch_start=100,
        ch_end=200,
        debug_overlay=False,
        source_video=video,
    )

    video.write_bytes(b"changed-size")

    signals_path_after = core._signals_cache_path(
        archive="sample_archive",
        ch_title="Sample Chapter",
        video_path=video,
        start_frame=100,
        end_frame=200,
        frame_read_offset=0,
    )
    extract_path_after = core._chapter_extract_cache_path(
        archive="sample_archive",
        chapter_title="Sample Chapter",
        ch_start=100,
        ch_end=200,
        debug_overlay=False,
        source_video=video,
    )

    assert signals_path_after != signals_path_before
    assert extract_path_after != extract_path_before

