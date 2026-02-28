from pathlib import Path

from common import chapter_frame_bounds, load_bad_frames_by_chapter_from_render_settings, parse_chapters
from vhs_pipeline import render_pipeline


ARCHIVE = "callahan_01_archive"
RICE_TOSS_TITLE = "1995 - Jim & Linda Wedding - 10 Rice Toss Send-Off (March 18, 1995)"


def test_rice_toss_render_settings_sources_are_clean_and_in_bounds() -> None:
    root = Path(__file__).resolve().parents[1]
    meta_dir = root / "metadata" / ARCHIVE
    chapters_path = meta_dir / "chapters.ffmetadata"
    settings_path = meta_dir / "render_settings.json"
    if not chapters_path.exists() or not settings_path.exists():
        return

    bad_by_title = load_bad_frames_by_chapter_from_render_settings(ARCHIVE)
    bad_global = sorted(int(f) for f in (bad_by_title.get(RICE_TOSS_TITLE) or []))
    assert bad_global, "Rice Toss bad frame list is empty in render_settings.json."

    _ffm, chapters = parse_chapters(chapters_path)
    chapter = next((c for c in chapters if str(c.get("title", "")).strip() == RICE_TOSS_TITLE), None)
    assert chapter is not None, "Rice Toss chapter title not found in chapters.ffmetadata."

    start_frame, end_frame = chapter_frame_bounds(chapter, fps_num=30000, fps_den=1001)
    max_local = int(end_frame) - int(start_frame) - 1
    assert max_local >= 0

    local_bad = sorted(
        {int(f) - int(start_frame) for f in bad_global if int(start_frame) <= int(f) < int(end_frame)}
    )
    assert local_bad, "Rice Toss has no in-bounds BAD_FRAMES after chapter mapping."

    local_repairs = render_pipeline.local_bad_frames_to_repairs(local_bad, max_frame=max_local)
    resolved = render_pipeline._resolve_badframe_repair_ranges(
        bad_repair_ranges=local_repairs,
        max_source_frame=max_local,
        source_clearance=render_pipeline.BADFRAME_SOURCE_CLEARANCE,
    )
    assert resolved, "No resolved FreezeFrame repairs were produced for Rice Toss."

    bad_set = set(local_bad)
    for a, b, src in resolved:
        assert 0 <= int(src) <= max_local
        assert int(src) not in bad_set


def test_auto_policy_uses_next_clean_source_for_singleton() -> None:
    resolved = render_pipeline._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(10, 10, None)],
        max_source_frame=30,
        source_clearance=render_pipeline.BADFRAME_SOURCE_CLEARANCE,
    )
    assert resolved == [(10, 10, 11)]


def test_auto_policy_falls_back_to_previous_source_at_chapter_end() -> None:
    resolved = render_pipeline._resolve_badframe_repair_ranges(
        bad_repair_ranges=[(8, 10, None)],
        max_source_frame=10,
        source_clearance=render_pipeline.BADFRAME_SOURCE_CLEARANCE,
    )
    assert resolved == [(8, 10, 7)]


def test_auto_policy_does_not_bridge_across_clean_gap() -> None:
    local_bad = [31021, 31024, 31025]
    repairs = render_pipeline.local_bad_frames_to_repairs(local_bad, max_frame=40000)
    assert repairs == [(31021, 31021, None), (31024, 31025, None)]

    resolved = render_pipeline._resolve_badframe_repair_ranges(
        bad_repair_ranges=repairs,
        max_source_frame=40000,
        source_clearance=render_pipeline.BADFRAME_SOURCE_CLEARANCE,
    )
    assert resolved == [(31021, 31021, 31022), (31024, 31025, 31026)]
