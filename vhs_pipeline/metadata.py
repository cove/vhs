from __future__ import annotations

import glob
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from common import (
    ARCHIVE_CHECKSUM_FILE,
    ARCHIVE_DIR,
    MEDIAINFO_BIN,
    METADATA_DIR,
    parse_chapters,
    write_sha3_manifest,
)


def generate_tsv_metadata(ffmetadata_path: Path, out_path: Path):
    ffmeta, chapters = parse_chapters(ffmetadata_path)
    lines = ["Title\tAuthor\tChapterTitle\tStartSeconds\tEndSeconds\tLocation"]

    for chapter in chapters:
        start = chapter.get("start_seconds")
        end = chapter.get("end_seconds")
        if start is None or end is None:
            timebase = chapter.get("timebase", "1/1")
            num, den = timebase.split("/", 1)
            start = round(int(chapter["start"]) * (int(num) / int(den)), 3)
            end = round(int(chapter["end"]) * (int(num) / int(den)), 3)

        lines.append(
            "\t".join(
                [
                    ffmeta.get("title", ""),
                    ffmeta.get("author", ""),
                    chapter.get("title", ""),
                    str(start),
                    str(end),
                    chapter.get("location", ""),
                ]
            )
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Generated TSV metadata: {out_path}")


def generate_mkv_chapters_xml(ffmetadata_path: Path, out_path: Path):
    _ffmeta, chapters = parse_chapters(ffmetadata_path)

    def _fmt(seconds: float):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    root = ET.Element("Chapters")
    edition = ET.SubElement(root, "EditionEntry")

    for chapter in chapters:
        start = chapter.get("start_seconds")
        end = chapter.get("end_seconds")
        if start is None or end is None:
            timebase = chapter.get("timebase", "1/1")
            num, den = map(int, timebase.split("/", 1))
            start = round(int(chapter["start"]) * (num / den), 3)
            end = round(int(chapter["end"]) * (num / den), 3)

        atom = ET.SubElement(edition, "ChapterAtom")
        ET.SubElement(atom, "ChapterTimeStart").text = _fmt(start)
        ET.SubElement(atom, "ChapterTimeEnd").text = _fmt(end)
        display = ET.SubElement(atom, "ChapterDisplay")
        ET.SubElement(display, "ChapterString").text = chapter.get("title", "") or ""
        ET.SubElement(display, "ChapterLanguage").text = "und"

    try:
        ET.indent(root, space="  ", level=0)
    except AttributeError:
        pass

    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE Chapters SYSTEM "matroskachapters.dtd">\n'
        + ET.tostring(root, encoding="unicode")
        + "\n",
        encoding="utf-8",
    )
    print(f"  Generated MKV chapters XML: {out_path}")


def write_mediainfo_outputs(input_path: Path, output_dir: Path):
    source = Path(input_path)
    outputs = [("Text", f"{source.stem}_mediainfo.txt"), ("XML", f"{source.stem}_mediainfo.xml")]

    for fmt, filename in outputs:
        out_path = output_dir / filename
        cmd = [str(MEDIAINFO_BIN), f"--Output={fmt}", str(source)]
        try:
            with out_path.open("w", encoding="utf-8") as out:
                result = subprocess.run(cmd, cwd=output_dir, stdout=out, text=True)
                if result.returncode:
                    print(f"  ERROR: mediainfo {fmt} failed for {source}")
                    return int(result.returncode)
        except FileNotFoundError:
            print(f"  ERROR: mediainfo command not found: {MEDIAINFO_BIN}")
            print(
                "  Install MediaInfo CLI (e.g. sudo apt-get install mediainfo) "
                "or set MEDIAINFO_BIN."
            )
            return 1
    return 0


def generate_archive_metadata(root_dir: Path = ARCHIVE_DIR):
    files = sorted(
        glob.glob(str(root_dir / "*.mkv")) + glob.glob(str(root_dir / "*.flac")),
        key=lambda x: x.lower(),
    )
    if not files:
        print("No files found.")
        return 1

    print(f"Processing directory: {Path(root_dir).resolve()}")
    for fn in files:
        source = Path(fn)
        print(f"Processing: {source}")

        rc = write_mediainfo_outputs(source, ARCHIVE_DIR)
        if rc:
            return rc

        ffmetadata_path = METADATA_DIR / source.stem / "chapters.ffmetadata"
        tsv_path = METADATA_DIR / source.stem / "markers.tsv"
        mkv_chapter_path = METADATA_DIR / source.stem / "markers.mkvchapters.xml"
        if not ffmetadata_path.exists():
            print(f"  Missing metadata file: {ffmetadata_path}")
            return 1

        generate_tsv_metadata(ffmetadata_path, tsv_path)
        generate_mkv_chapters_xml(ffmetadata_path, mkv_chapter_path)

        metadata_dir = ffmetadata_path.parent
        archive_metadata_dir = ARCHIVE_DIR / f"{source.stem}_metadata"
        shutil.copytree(metadata_dir, archive_metadata_dir, dirs_exist_ok=True)

    write_sha3_manifest(ARCHIVE_DIR, ARCHIVE_CHECKSUM_FILE)
    print(f"Checksum manifest: {ARCHIVE_CHECKSUM_FILE}")
    print("All done.")
    return 0


def main(argv=None):
    _ = argv
    return generate_archive_metadata(ARCHIVE_DIR)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

