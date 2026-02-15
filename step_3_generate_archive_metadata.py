#
# Generates archival metadata for MKV files, including TSV chapter markers, MKV chapter XML,
# Mediainfo outputs, and SHA3-256 checksums. Copies metadata to archive directories for preservation.
#
import glob, shutil
import xml.etree.ElementTree as ET
from common import *

def generate_tsv_metadata(ffmetadata_path, out_path):
    ffmeta, chapters = parse_chapters(ffmetadata_path)

    lines = ["Title\tAuthor\tChapterTitle\tStartSeconds\tEndSeconds\tLocation"]
    for ch in chapters:
        start = ch.get("start_seconds")
        end = ch.get("end_seconds")
        if start is None or end is None:
            # Calculate using TIMEBASE if not already done
            timebase = ch.get("timebase", "1/1")
            num, den = timebase.split("/", 1)
            num = int(num)
            den = int(den)
            start = round(int(ch["start"]) * (num / den), 3)
            end = round(int(ch["end"]) * (num / den), 3)

        line = "\t".join([
            ffmeta.get("title", ""),
            ffmeta.get("author", ""),
            ch.get("title", ""),
            str(start),
            str(end),
            ch.get("location", ""),
        ])
        lines.append(line)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("  Generated TSV metadata:", out_path)

def generate_mkv_chapters_xml(ffmetadata_path, out_path):
    ffmeta, chapters = parse_chapters(ffmetadata_path)

    def fmt(t):
        # seconds → HH:MM:SS.mmm
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    root = ET.Element("Chapters")
    edition = ET.SubElement(root, "EditionEntry")

    for ch in chapters:
        # Resolve seconds
        start = ch.get("start_seconds")
        end = ch.get("end_seconds")
        if start is None or end is None:
            timebase = ch.get("timebase", "1/1")
            num, den = map(int, timebase.split("/", 1))
            start = round(int(ch["start"]) * (num / den), 3)
            end = round(int(ch["end"]) * (num / den), 3)

        atom = ET.SubElement(edition, "ChapterAtom")

        ET.SubElement(atom, "ChapterTimeStart").text = fmt(start)
        ET.SubElement(atom, "ChapterTimeEnd").text = fmt(end)

        disp = ET.SubElement(atom, "ChapterDisplay")
        ET.SubElement(disp, "ChapterString").text = ch.get("title", "") or ""
        ET.SubElement(disp, "ChapterLanguage").text = "und"

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

    print("  Generated MKV chapters XML:", out_path)


def copy_metadata_folder(metadata_dir, archive_name):
    dest_path = ARCHIVE_DIR / archive_name
    shutil.copytree(metadata_dir, dest_path)

def compute_checksums(root_dir, manifest_path):
    write_sha3_manifest(root_dir, manifest_path)

def write_mediainfo_outputs(input_path, output_dir):
    input_path = Path(input_path)
    stem = input_path.stem

    outputs = [
        ("Text", f"{stem}_mediainfo.txt"),
        ("XML",  f"{stem}_mediainfo.xml"),
    ]

    for fmt, filename in outputs:
        out_path = output_dir / filename
        cmd = [str(MEDIAINFO_BIN), f"--Output={fmt}", str(input_path)]

        with open(out_path, "w", encoding="utf-8") as out:
            r = subprocess.run(cmd, cwd=output_dir, stdout=out, text=True)
            if r.returncode:
                print(f"  ERROR: mediainfo {fmt} failed for {input_path}")
                sys.exit(r.returncode)

    return True

def generate_metadata(root_dir):
    files = (
            glob.glob(str(root_dir / "*.mkv")) + glob.glob(str(root_dir / "*.flac"))
    )
    if not files:
        print("No files found.")
        sys.exit(1)

    print(f"Processing directory: {Path(root_dir).resolve()}")
    for fn in files:
        print("Processing:", fn)
        path = Path(fn)
        file_name = path.name

        write_mediainfo_outputs(file_name, ARCHIVE_DIR)

        ffmetadata_path = METADATA_DIR / path.stem / "chapters.ffmetadata"
        tsv_path = METADATA_DIR / path.stem / "markers.tsv"
        generate_tsv_metadata(ffmetadata_path, tsv_path)

        mkv_chapter_path = METADATA_DIR / path.stem / "markers.mkvchapters.xml"
        generate_mkv_chapters_xml(ffmetadata_path, mkv_chapter_path)

        metadata_dir = ffmetadata_path.parent
        archive_name = ARCHIVE_DIR / f"{path.stem}_metadata"  # or generate based on context/path
        copy_metadata_folder(metadata_dir, archive_name)

def main():
    generate_metadata(ARCHIVE_DIR)
    compute_checksums(ARCHIVE_DIR, ARCHIVE_CHECKSUM_FILE)

    print("Checksum manifest:", ARCHIVE_CHECKSUM_FILE)
    print("All done.")


if __name__ == "__main__":
    main()

