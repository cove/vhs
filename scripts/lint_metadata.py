from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import chapter_frame_bounds, parse_chapters


DERIVED_CHAPTER_KEYS = {"start_raw", "end_raw", "timebase_num", "timebase_den"}

CANONICAL_CHAPTER_KEYS = {
    "timebase",
    "start",
    "end",
    "title",
    "author",
    "artist",
    "creation_time",
    "location",
    "comment",
    "bad_frames",
    "recording_time",
    "recording_date",
    "recording_location",
    "date_recorded",
    "date_digitized",
    "audio",
    "grouping",
}

LIKELY_TYPO_MAP = {
    "autor": "author",
}

FPS = Fraction(30000, 1001)


@dataclass(frozen=True)
class Finding:
    level: str
    archive: str
    message: str


def _parse_fraction(text: str) -> Fraction:
    raw = str(text or "").strip()
    if "/" in raw:
        a, b = raw.split("/", 1)
        num = int(a)
        den = int(b)
    else:
        num = int(raw)
        den = 1
    if den == 0:
        raise ValueError("Timebase denominator cannot be 0.")
    return Fraction(num, den)


def _round_fraction_nearest_int(frac: Fraction) -> int:
    frac = Fraction(frac)
    if frac >= 0:
        return int((frac.numerator * 2 + frac.denominator) // (2 * frac.denominator))
    pos = -frac
    return -int((pos.numerator * 2 + pos.denominator) // (2 * pos.denominator))


def _iter_archive_dirs(globs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(Path().glob(pattern)):
            if not path.is_dir():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


def _lint_archive(
    archive_dir: Path,
    *,
    target_timebase: Fraction,
    require_render_settings: bool,
    simulate_conversion: bool,
) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    archive = archive_dir.name
    ffmeta_path = archive_dir / "chapters.ffmetadata"
    render_settings_path = archive_dir / "render_settings.json"
    summary = {
        "archive": archive,
        "chapters": 0,
        "timebases": set(),
        "has_render_settings": render_settings_path.exists(),
    }

    if not ffmeta_path.exists():
        findings.append(Finding("ERROR", archive, "Missing chapters.ffmetadata"))
        return findings, summary

    raw_lines = ffmeta_path.read_text(encoding="utf-8", errors="replace").splitlines()
    first_nonblank = ""
    for line in raw_lines:
        text = line.strip()
        if text:
            first_nonblank = text
            break
    if first_nonblank != ";FFMETADATA1":
        findings.append(Finding("WARN", archive, "First non-empty line is not ';FFMETADATA1'"))

    if require_render_settings and not render_settings_path.exists():
        findings.append(Finding("WARN", archive, "Missing render_settings.json"))
    if not require_render_settings and not render_settings_path.exists():
        findings.append(Finding("INFO", archive, "Missing render_settings.json (created automatically when needed)"))

    try:
        ffmeta, chapters = parse_chapters(ffmeta_path)
    except Exception as exc:
        findings.append(Finding("ERROR", archive, f"Parser failure: {exc}"))
        return findings, summary

    if not chapters:
        findings.append(Finding("ERROR", archive, "No chapters parsed"))
        return findings, summary

    summary["chapters"] = len(chapters)

    if not str(ffmeta.get("title", "")).strip():
        findings.append(Finding("WARN", archive, "Missing global title="))
    if not str(ffmeta.get("author", "")).strip():
        findings.append(Finding("WARN", archive, "Missing global author="))

    titles = [str(ch.get("title", "")).strip() for ch in chapters]
    duplicates = [title for title, count in Counter(titles).items() if title and count > 1]
    if duplicates:
        findings.append(Finding("WARN", archive, f"Duplicate chapter titles ({len(duplicates)}): {duplicates[0]!r}"))

    previous = None
    max_sec_drift = Fraction(0, 1)
    conversion_mismatches = 0

    for idx, ch in enumerate(chapters, start=1):
        label = f"ch{idx}"
        title = str(ch.get("title", "")).strip()
        if not title:
            findings.append(Finding("ERROR", archive, f"{label}: missing title"))

        for req in ("start_raw", "end_raw"):
            if req not in ch:
                findings.append(Finding("ERROR", archive, f"{label}: missing {req.replace('_raw', '').upper()}"))

        try:
            start_raw = int(ch.get("start_raw"))
            end_raw = int(ch.get("end_raw"))
        except Exception:
            findings.append(Finding("ERROR", archive, f"{label}: non-integer START/END"))
            continue

        if end_raw < start_raw:
            findings.append(Finding("ERROR", archive, f"{label}: END ({end_raw}) < START ({start_raw})"))

        if "timebase" not in ch:
            findings.append(Finding("WARN", archive, f"{label}: missing TIMEBASE (defaults to 1/1)"))
            tb = Fraction(1, 1)
        else:
            try:
                tb = _parse_fraction(str(ch.get("timebase")))
            except Exception as exc:
                findings.append(Finding("ERROR", archive, f"{label}: invalid TIMEBASE ({exc})"))
                continue

        tb_text = f"{tb.numerator}/{tb.denominator}"
        summary["timebases"].add(tb_text)
        if tb != target_timebase:
            findings.append(Finding("INFO", archive, f"{label}: legacy TIMEBASE={tb_text}"))

        unknown_keys: list[str] = []
        for key in ch.keys():
            key_s = str(key)
            if key_s in DERIVED_CHAPTER_KEYS:
                continue
            if key_s.startswith("#"):
                findings.append(Finding("WARN", archive, f"{label}: comment-style key parsed literally: {key_s}"))
                continue
            if key_s in CANONICAL_CHAPTER_KEYS:
                continue
            if key_s.startswith("recording_"):
                continue
            if key_s in LIKELY_TYPO_MAP:
                findings.append(
                    Finding(
                        "WARN",
                        archive,
                        f"{label}: probable typo key '{key_s}' (did you mean '{LIKELY_TYPO_MAP[key_s]}')",
                    )
                )
                continue
            unknown_keys.append(key_s)
        for key_s in sorted(set(unknown_keys)):
            findings.append(Finding("WARN", archive, f"{label}: unknown chapter key '{key_s}'"))

        start_frame, end_frame = chapter_frame_bounds(ch, fps_num=30000, fps_den=1001)
        if previous is not None:
            prev_start, prev_end, prev_label = previous
            if start_frame < prev_end:
                findings.append(
                    Finding(
                        "WARN",
                        archive,
                        (
                            f"{label}: overlaps {prev_label} "
                            f"(frames {start_frame}-{max(start_frame, end_frame - 1)} "
                            f"vs previous {prev_start}-{max(prev_start, prev_end - 1)})"
                        ),
                    )
                )
        previous = (start_frame, end_frame, label)

        if simulate_conversion:
            converted_start = _round_fraction_nearest_int(Fraction(start_raw) * tb / target_timebase)
            converted_end = _round_fraction_nearest_int(Fraction(end_raw) * tb / target_timebase)

            original_start_frame = _round_fraction_nearest_int(Fraction(start_raw) * tb * FPS)
            original_end_frame = _round_fraction_nearest_int(Fraction(end_raw) * tb * FPS)
            converted_start_frame = _round_fraction_nearest_int(Fraction(converted_start) * target_timebase * FPS)
            converted_end_frame = _round_fraction_nearest_int(Fraction(converted_end) * target_timebase * FPS)

            if (converted_start_frame, converted_end_frame) != (original_start_frame, original_end_frame):
                conversion_mismatches += 1
                findings.append(
                    Finding(
                        "ERROR",
                        archive,
                        (
                            f"{label}: converted frame mismatch "
                            f"({converted_start_frame},{converted_end_frame}) != "
                            f"({original_start_frame},{original_end_frame})"
                        ),
                    )
                )

            start_drift = abs(Fraction(converted_start) * target_timebase - Fraction(start_raw) * tb)
            end_drift = abs(Fraction(converted_end) * target_timebase - Fraction(end_raw) * tb)
            max_sec_drift = max(max_sec_drift, start_drift, end_drift)

    if simulate_conversion:
        allowed = target_timebase / 2
        if max_sec_drift > allowed:
            findings.append(
                Finding(
                    "ERROR",
                    archive,
                    (
                        "Simulated conversion drift exceeds half target tick: "
                        f"max={float(max_sec_drift):.9f}s allowed={float(allowed):.9f}s"
                    ),
                )
            )
        findings.append(
            Finding(
                "INFO",
                archive,
                (
                    "Simulated conversion summary: "
                    f"frame_mismatches={conversion_mismatches}, "
                    f"max_time_drift={float(max_sec_drift):.9f}s, "
                    f"allowed_half_tick={float(allowed):.9f}s"
                ),
            )
        )

    return findings, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lint ffmetadata archives and optionally simulate timebase conversion."
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Archive directory glob(s). Repeatable. Default: metadata/*_archive",
    )
    parser.add_argument(
        "--target-timebase",
        default="1001/30000",
        help="Canonical chapter TIMEBASE to compare/convert against. Default: 1001/30000",
    )
    parser.add_argument(
        "--require-render-settings",
        action="store_true",
        help="Warn when render_settings.json is missing.",
    )
    parser.add_argument(
        "--simulate-timebase-conversion",
        action="store_true",
        help="Simulate conversion to --target-timebase and verify frame-bound equivalence.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return non-zero exit code when WARN findings exist.",
    )
    parser.add_argument(
        "--show-info",
        action="store_true",
        help="Print INFO findings. Default prints only WARN/ERROR.",
    )
    args = parser.parse_args()

    target_timebase = _parse_fraction(args.target_timebase)
    globs = list(args.glob) if args.glob else ["metadata/*_archive"]
    archive_dirs = _iter_archive_dirs(globs)
    if not archive_dirs:
        print("No archive directories matched the provided --glob pattern(s).")
        return 1

    all_findings: list[Finding] = []
    summaries: list[dict] = []

    for archive_dir in archive_dirs:
        findings, summary = _lint_archive(
            archive_dir,
            target_timebase=target_timebase,
            require_render_settings=bool(args.require_render_settings),
            simulate_conversion=bool(args.simulate_timebase_conversion),
        )
        all_findings.extend(findings)
        summaries.append(summary)

    print("Archive summary:")
    for item in summaries:
        tbs = ",".join(sorted(item["timebases"])) if item["timebases"] else "(none)"
        print(
            f"  - {item['archive']}: chapters={item['chapters']}, "
            f"render_settings={item['has_render_settings']}, timebases={tbs}"
        )

    levels = Counter(f.level for f in all_findings)
    print(
        "Findings: "
        f"errors={levels.get('ERROR', 0)}, "
        f"warnings={levels.get('WARN', 0)}, "
        f"info={levels.get('INFO', 0)}"
    )

    for f in all_findings:
        if f.level == "INFO" and not args.show_info:
            continue
        print(f"[{f.level}] {f.archive}: {f.message}")

    if levels.get("ERROR", 0) > 0:
        return 1
    if args.fail_on_warn and levels.get("WARN", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
