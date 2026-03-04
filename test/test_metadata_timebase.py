from pathlib import Path

from vhs_pipeline.metadata import generate_mkv_chapters_xml, generate_tsv_metadata


def _write_ffmetadata(path: Path, *, timebase: str, start: int, end: int) -> None:
    path.write_text(
        "\n".join(
            [
                ";FFMETADATA1",
                "title=Unit Archive",
                "author=Unit Tester",
                "",
                "[CHAPTER]",
                f"TIMEBASE={timebase}",
                f"START={start}",
                f"END={end}",
                "title=Unit Chapter",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_generate_tsv_metadata_uses_normalized_seconds_for_1001_30000(tmp_path: Path) -> None:
    ffmeta = tmp_path / "chapters.ffmetadata"
    out = tmp_path / "markers.tsv"
    _write_ffmetadata(ffmeta, timebase="1001/30000", start=101, end=303)

    generate_tsv_metadata(ffmeta, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    row = lines[1].split("\t")
    assert row[3] == "3.37"
    assert row[4] == "10.11"


def test_generate_tsv_metadata_uses_normalized_seconds_for_1_100(tmp_path: Path) -> None:
    ffmeta = tmp_path / "chapters.ffmetadata"
    out = tmp_path / "markers.tsv"
    _write_ffmetadata(ffmeta, timebase="1/100", start=6315, end=6603)

    generate_tsv_metadata(ffmeta, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    row = lines[1].split("\t")
    assert row[3] == "63.15"
    assert row[4] == "66.03"


def test_generate_mkv_chapters_xml_uses_normalized_seconds(tmp_path: Path) -> None:
    ffmeta = tmp_path / "chapters.ffmetadata"
    out = tmp_path / "markers.mkvchapters.xml"
    _write_ffmetadata(ffmeta, timebase="1001/30000", start=101, end=303)

    generate_mkv_chapters_xml(ffmeta, out)
    xml = out.read_text(encoding="utf-8")
    assert "<ChapterTimeStart>00:00:03.370</ChapterTimeStart>" in xml
    assert "<ChapterTimeEnd>00:00:10.110</ChapterTimeEnd>" in xml
