from pathlib import Path

import vhs_pipeline.proxy as proxy_mod


def test_build_proxy_command_uses_half_scale_and_lockstep_flags(tmp_path: Path) -> None:
    src = tmp_path / "archive.mkv"
    ffmeta = tmp_path / "chapters.ffmetadata"
    out = tmp_path / "archive_proxy.mp4"
    cmd = proxy_mod.build_proxy_command(src, ffmeta, out)

    joined = " ".join(cmd)
    assert "scale=iw/2:ih/2:flags=lanczos,setpts=N/(30000/1001*TB)" in joined
    assert "drawtext=" in joined
    assert "frame=%{eif\\:n\\:d}" in joined
    assert "-r 30000/1001" in joined
    assert "-fps_mode:v:0 cfr" in joined
    assert "-vsync cfr" in joined
    assert str(out) == cmd[-1]


def test_make_proxies_invokes_ffmpeg_with_lockstep_command(tmp_path: Path, monkeypatch) -> None:
    archive_dir = tmp_path / "Archive"
    metadata_dir = tmp_path / "metadata"
    archive_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    src = archive_dir / "unit_archive.mkv"
    src.write_bytes(b"mkv")
    md = metadata_dir / "unit_archive"
    md.mkdir(parents=True, exist_ok=True)
    (md / "chapters.ffmetadata").write_text(";FFMETADATA1\n", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, cwd=None):
        _ = cwd
        captured["cmd"] = [str(x) for x in cmd]

    monkeypatch.setattr(proxy_mod, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(proxy_mod, "METADATA_DIR", metadata_dir)
    monkeypatch.setattr(proxy_mod, "run", _fake_run)

    rc = proxy_mod.make_proxies()
    assert rc == 0
    assert "cmd" in captured
    cmd = captured["cmd"]
    vf_parts = [part for part in cmd if "scale=iw/2:ih/2:flags=lanczos,setpts=N/(30000/1001*TB)" in part]
    assert vf_parts, "Expected lockstep half-scale filter chain in ffmpeg command."
    assert any("drawtext=" in part for part in vf_parts), "Expected frame number drawtext in proxy filter chain."
    assert "30000/1001" in cmd
