from pathlib import Path
import shutil

from common import (
    get_gamma_profile_for_chapter,
    load_render_settings,
    save_render_settings,
    update_chapter_gamma_in_render_settings,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gamma_profile_loads_archive_and_chapter_overrides() -> None:
    archive = "__gamma_unit_archive"
    title = "Gamma Unit Chapter"
    archive_meta_dir = ROOT / "metadata" / archive
    try:
        _path, settings = load_render_settings(archive, create=True)
        settings["archive_settings"]["gamma_default"] = 1.0
        settings["archive_settings"]["gamma_ranges"] = [
            {"start_frame": 100, "end_frame": 200, "gamma": 1.3},
            {"start_frame": 210, "end_frame": 260, "gamma": 1.6},
        ]
        save_render_settings(archive, settings)

        archive_profile = get_gamma_profile_for_chapter(
            archive,
            title,
            ch_start=120,
            ch_end=230,
        )
        assert archive_profile["source"] == "archive"
        assert archive_profile["ranges"] == [
            {"start_frame": 120, "end_frame": 200, "gamma": 1.3},
            {"start_frame": 210, "end_frame": 230, "gamma": 1.6},
        ]

        update_chapter_gamma_in_render_settings(
            archive,
            title,
            gamma_ranges=[
                {"start_frame": 150, "end_frame": 180, "gamma": 1.8},
                {"start_frame": 170, "end_frame": 190, "gamma": 1.5},
            ],
            default_gamma=1.0,
        )
        chapter_profile = get_gamma_profile_for_chapter(
            archive,
            title,
            ch_start=120,
            ch_end=230,
        )
        assert chapter_profile["source"] == "chapter"
        assert chapter_profile["ranges"] == [
            {"start_frame": 150, "end_frame": 170, "gamma": 1.8},
            {"start_frame": 170, "end_frame": 190, "gamma": 1.5},
        ]
    finally:
        shutil.rmtree(archive_meta_dir, ignore_errors=True)
