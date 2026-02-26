from pathlib import Path

import libs.vhs_tuner_core as core


def test_resolve_archive_video_prefers_proxy_when_both_exist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(core, "ARCHIVE_DIR", tmp_path)
    (tmp_path / "sample_archive.mkv").write_bytes(b"mkv")
    proxy = tmp_path / "sample_archive_proxy.mp4"
    proxy.write_bytes(b"mp4")

    resolved = core._resolve_archive_video("sample_archive")
    assert resolved == proxy


def test_resolve_archive_video_falls_back_to_mkv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(core, "ARCHIVE_DIR", tmp_path)
    mkv = tmp_path / "sample_archive.mkv"
    mkv.write_bytes(b"mkv")

    resolved = core._resolve_archive_video("sample_archive")
    assert resolved == mkv


def test_resolve_archive_video_returns_none_when_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(core, "ARCHIVE_DIR", tmp_path)

    assert core._resolve_archive_video("sample_archive") is None
