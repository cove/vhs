#!/usr/bin/env python3.11
"""
VHS Bad Frame Tuner - Gradio edition
=====================================
Run:  python vhs_tuner.py

Requires:  pip install gradio opencv-python-headless numpy pillow pandas

Metadata layout (all under metadata/<archive>/)
------------------------------------------------
  chapters.ffmetadata           per-chapter BAD_FRAMES=<csv global frame ids>

legacy_steps/step_6_make_videos.py reads chapter BAD_FRAMES lists directly from chapters.ffmetadata.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

# -- Project paths -------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "scripts" else _HERE
sys.path.insert(0, str(PROJECT_ROOT))

from common import combined_score, compute_threshold
from libs.vhs_tuner_core import (
    CHAPTER_MISSING_LABEL,
    CHAPTER_SELECT_LABEL,
    STEP6_DEBUG_EXTRACT_FRAME_NUMBERS_ENV,
    TUNER_DEBUG_EXTRACT_ENV,
    _chapter_bad_overrides,
    _chapter_details_md,
    _chapter_extract_cache_path,
    _ensure_step6_chapter_extract,
    _env_truthy,
    _find_chapter,
    _get_archives,
    _normalize_frame_span,
    _resolve_archive_video,
    _sparkline_svg,
    apply_manual_click_override,
    build_archive_state,
    build_finalize_summary,
    build_review_data,
    extract_frames,
    persist_bad_frames_for_chapter,
    sample_count_from_stride,
    select_focus_frame_ids,
)

_DARK_CSS = """
html, body {
  margin: 0 !important;
  padding: 0 !important;
  background:#0d0d0d !important;
  height: 100vh !important;
  overflow: hidden !important;
}
body, .gradio-container { background:#0d0d0d !important; color:#ccc; }
/* Full-bleed layout for maximum usable area. */
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  min-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}
.gradio-container .main {
  margin: 0 !important;
  padding: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
}
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .app {
  margin: 0 !important;
  padding: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
}
.gr-row, .gr-form, .gr-group { gap: 3px !important; margin: 0 !important; padding: 0 !important; }
.gr-box, .gr-padded { background:#141414 !important; }
.gr-button-primary { background:#1a6b3a !important; border-color:#27a85a !important; }
.gr-button { background:#222 !important; color:#bbb !important; border-color:#333 !important; }
label { color:#999 !important; font-family:'Courier New',monospace !important; font-size:0.72rem !important; }
input[type=range] { accent-color:#27a85a; }
# Compact common controls to help fit one screen.
.gradio-container input,
.gradio-container textarea,
.gradio-container .gr-button,
.gradio-container .gr-markdown,
.gradio-container .gr-form,
.gradio-container .gr-slider,
.gradio-container .gr-number,
.gradio-container .gr-dropdown {
  font-size: 11px !important;
}
.gradio-container .gr-button {
  min-height: 26px !important;
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}
.gradio-container .gr-slider,
.gradio-container .gr-number,
.gradio-container .gr-dropdown,
.gradio-container .gr-radio {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
.gradio-container .gr-accordion {
  margin: 0 !important;
}
.gradio-container .gr-accordion summary {
  padding: 2px 8px !important;
  min-height: 22px !important;
  font-size: 11px !important;
}
.gradio-container .gr-accordion > div {
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}
.gradio-container .gr-block.gr-box,
.gradio-container .block {
  margin: 0 !important;
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}
.gradio-container .prose,
.gradio-container .gr-markdown {
  margin: 0 !important;
  padding: 0 !important;
}
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p {
  margin-top: 2px !important;
  margin-bottom: 2px !important;
}
#vhs-stats { font-family:'Courier New',monospace; font-size:11px;
             background:#111; padding:5px 8px; border-left:2px solid #27a85a; }
#vhs-apply-log textarea  { font-family:'Courier New',monospace; font-size:11px;
                           background:#0a0a0a !important; color:#9fdfb8 !important; }
#vhs-render-log textarea { font-family:'Courier New',monospace; font-size:11px;
                           background:#0a0a0a !important; color:#9fdfb8 !important; }
#vhs-grid-gallery [class*="caption"] {
  font-family:'Courier New',monospace !important;
  font-size:9px !important;
  text-align:left !important;
  white-space:normal !important;
  overflow-wrap:anywhere !important;
  line-height:1.15 !important;
  padding-left:2px !important;
}
#vhs-hover-preview {
  position: fixed;
  z-index: 9999;
  display: none;
  pointer-events: none;
  border: 2px solid #2a2a2a;
  background: #050505;
  box-shadow: 0 8px 24px rgba(0,0,0,.6);
  border-radius: 3px;
  overflow: hidden;
}
#vhs-hover-preview img {
  display: block;
  max-width: min(70vw, 1280px);
  max-height: min(70vh, 720px);
  width: auto;
  height: auto;
}
#vhs-main-panel {
  display: flex !important;
  flex-direction: column !important;
  height: 100dvh !important;
  overflow: hidden !important;
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}
#vhs-step-title {
  margin-bottom: 4px !important;
}
#vhs-shell {
  display: flex !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
  gap: 8px !important;
}
#vhs-rail-expanded,
#vhs-rail-collapsed {
  display: flex !important;
  flex-direction: column !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-rail-expanded > .column,
#vhs-rail-collapsed > .column {
  min-height: 0 !important;
}
#vhs-rail-collapse button,
#vhs-rail-expand button {
  min-height: 22px !important;
  padding: 1px 7px !important;
}
#vhs-work-col {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-step-load,
#vhs-step-review,
#vhs-step-final {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-review-shell {
  display: flex !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
  gap: 8px !important;
}
#vhs-review-main {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-review-inspector {
  display: flex !important;
  flex-direction: column !important;
  min-height: 0 !important;
  overflow: auto !important;
}
#vhs-stats {
  flex: 0 0 auto !important;
}
#vhs-grid-gallery {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  width: 100% !important;
  min-height: 0 !important;
  max-height: none !important;
  height: 100% !important;
  overflow: hidden !important;
}
#vhs-grid-gallery > div {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-grid-gallery .gallery-container {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}
#vhs-grid-gallery .grid-wrap,
#vhs-grid-gallery .grid-wrap.fixed-height {
  display: block !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: 100% !important;
  max-height: none !important;
  overflow: auto !important;
}
#vhs-grid-gallery .grid-container {
  min-height: 0 !important;
  align-content: start !important;
}
#vhs-grid-gallery [class*="gallery"] {
  min-height: 0 !important;
}
#vhs-grid-gallery [class*="grid"] {
  min-height: 0 !important;
}
/* Reduce distracting gallery redraw effects on click. */
#vhs-grid-gallery,
#vhs-grid-gallery *,
#vhs-grid-gallery [class*="image"],
#vhs-grid-gallery [class*="gallery"] {
  transition: none !important;
  animation: none !important;
}
.vhs-spark svg {
  width: 220px !important;
  max-width: 100% !important;
  height: 24px !important;
}
.vhs-spark-score svg {
  height: 32px !important;
}
#vhs-chapter-table table,
#vhs-chapter-table-compact table {
  table-layout: fixed !important;
  width: 100% !important;
}
#vhs-chapter-table th,
#vhs-chapter-table td,
#vhs-chapter-table-compact th,
#vhs-chapter-table-compact td {
  font-family: "Courier New", monospace !important;
  font-size: 11px !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
#vhs-chapter-table th:first-child,
#vhs-chapter-table td:first-child {
  width: 64px !important;
  text-align: right !important;
}
.vhs-widget {
  border: 1px solid #262626;
  border-radius: 4px;
  background: #111;
  padding: 6px;
}
.vhs-widget-title {
  font-family: "Courier New", monospace;
  font-size: 11px;
  color: #9fb3a6;
  margin-bottom: 4px;
}
#vhs-loader-toggle button {
  min-height: 20px !important;
  padding: 1px 6px !important;
  font-size: 10px !important;
}
#vhs-runtime-js {
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
"""

_E_SIG   = _sparkline_svg(np.array([]), None,  "", height=24)
_E_SCORE = _sparkline_svg(np.array([]), None,  "", height=32)

with gr.Blocks(
    title="  VHS Frame Tuner",
) as demo:

    # -- Persistent state ------------------------------------------------------
    st_fids      = gr.State([])
    st_b64       = gr.State([])
    st_sigs      = gr.State({})
    st_overrides = gr.State({})
    st_visible_fids = gr.State([])
    st_last_click = gr.State({"fid": -1, "ts": -1})
    st_chapters  = gr.State([])
    st_chapter_titles = gr.State([])
    st_rail_collapsed = gr.State(False)

    # -- Main panel ------------------------------------------------------------
    with gr.Column(visible=True, elem_id="vhs-main-panel") as main_panel:
        status_md = gr.Markdown("`Select a chapter from the left rail.`")
        step_md = gr.Markdown("`Step 1/3: Load Chapter`", elem_id="vhs-step-title")

        with gr.Row(elem_id="vhs-shell"):
            with gr.Column(scale=1, min_width=300, elem_id="vhs-rail-expanded") as rail_expanded:
                with gr.Row():
                    rail_collapse_btn = gr.Button("<", elem_id="vhs-rail-collapse")
                _archives = _get_archives()
                archive_dd = gr.Dropdown(
                    choices=_archives,
                    value=_archives[0] if _archives else None,
                    label="Archive",
                    interactive=True,
                )
                chapter_dd = gr.Dropdown(
                    choices=[CHAPTER_SELECT_LABEL],
                    value=CHAPTER_SELECT_LABEL,
                    label="Chapter",
                    interactive=True,
                    visible=False,
                )
                chapter_table = gr.Dataframe(
                    headers=["#", "Chapter", "Time", "Frames", "BAD"],
                    datatype=["number", "str", "str", "number", "number"],
                    value=[],
                    row_count=(0, "dynamic"),
                    column_count=(5, "fixed"),
                    interactive=False,
                    wrap=False,
                    elem_id="vhs-chapter-table",
                )

            with gr.Column(scale=0, min_width=64, elem_id="vhs-rail-collapsed", visible=False) as rail_collapsed:
                with gr.Row():
                    rail_expand_btn = gr.Button(">", elem_id="vhs-rail-expand")
                chapter_compact_table = gr.Dataframe(
                    headers=["#", "BAD"],
                    datatype=["number", "number"],
                    value=[],
                    row_count=(0, "dynamic"),
                    column_count=(2, "fixed"),
                    interactive=False,
                    wrap=False,
                    elem_id="vhs-chapter-table-compact",
                )

            with gr.Column(scale=5, min_width=760, elem_id="vhs-work-col"):
                with gr.Column(visible=True, elem_id="vhs-step-load") as load_step:
                    load_chapter_md = gr.Markdown("`Select a chapter from the left rail.`")
                    with gr.Group(elem_classes=["vhs-widget"]):
                        gr.Markdown("Sampling Setup", elem_classes=["vhs-widget-title"])
                        with gr.Row():
                            start_n = gr.Number(label="Start", value=0, precision=0, elem_id="vhs-start-frame")
                            end_n = gr.Number(label="End (exclusive)", value=10000, precision=0)
                        sample_stride_sl = gr.Slider(
                            1, 300, value=20, step=1, label="Sample Rate (1 frame per N)"
                        )
                        context_sl = gr.Slider(
                            0, 200, value=10, step=1, label="Bad Batch Proximity (frames)"
                        )
                        with gr.Accordion("Advanced Sampling", open=False):
                            strict_sampling_cb = gr.Checkbox(label="Strict Sampling", value=True)
                            exact_extract_cb = gr.Checkbox(label="Use Step6 Extract", value=True)
                            debug_extract_cb = gr.Checkbox(label="Debug Frame IDs", value=False)
                    with gr.Row():
                        load_next_btn = gr.Button("Next: Review Frames", variant="primary")

                with gr.Column(visible=False, elem_id="vhs-step-review") as review_step:
                    with gr.Row(elem_id="vhs-review-shell"):
                        with gr.Column(scale=4, elem_id="vhs-review-main"):
                            stats_md = gr.Markdown("", elem_id="vhs-stats")
                            grid_gallery = gr.Gallery(
                                value=[],
                                label="Frames (click a tile to toggle good/bad)",
                                show_label=True,
                                columns=7,
                                object_fit="contain",
                                height="auto",
                                allow_preview=False,
                                elem_id="vhs-grid-gallery",
                            )
                            with gr.Row():
                                review_back_btn = gr.Button("Back", variant="secondary")
                                review_apply_btn = gr.Button("Apply and Continue", variant="primary")

                        with gr.Column(scale=1, min_width=260, elem_id="vhs-review-inspector"):
                            with gr.Group(elem_classes=["vhs-widget"]):
                                gr.Markdown("IQR Threshold", elem_classes=["vhs-widget-title"])
                                t_mode = gr.Radio(
                                    ["iqr"],
                                    value="iqr",
                                    label="Mode",
                                    interactive=False,
                                    visible=False,
                                )
                                iqr_sl = gr.Slider(1.0, 8.0, value=3.5, step=0.05, label="IQR k")
                                spark_score = gr.HTML(
                                    _E_SCORE, elem_classes=["vhs-spark", "vhs-spark-score"]
                                )
                                tval_sl = gr.Slider(
                                    -5.0, 15.0, value=1.0, step=0.05, label="Hard value", visible=False
                                )
                                bpct_sl = gr.Slider(
                                    1, 60, value=10, step=1, label="Bad %", visible=False
                                )

                            with gr.Group(elem_classes=["vhs-widget"]):
                                gr.Markdown("Display", elem_classes=["vhs-widget-title"])
                                cols_sl = gr.Slider(4, 16, value=7, step=1, label="Grid Columns")
                                thumb_ids_cb = gr.Checkbox(label="Show IDs On Images", value=False)
                                twidth_sl = gr.Slider(
                                    64, 220, value=120, step=8, label="Width", visible=False
                                )

                            with gr.Accordion("Advanced Signal Weights", open=False):
                                wc_sl = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="chroma")
                                spark_chroma = gr.HTML(_E_SIG, elem_classes=["vhs-spark"])
                                wn_sl = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="noise")
                                spark_noise = gr.HTML(_E_SIG, elem_classes=["vhs-spark"])
                                wt_sl = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="tear")
                                spark_tear = gr.HTML(_E_SIG, elem_classes=["vhs-spark"])
                                ww_sl = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="wave")
                                spark_wave = gr.HTML(_E_SIG, elem_classes=["vhs-spark"])

                with gr.Column(visible=False, elem_id="vhs-step-final") as final_step:
                    final_stats_md = gr.Markdown("`No summary available yet.`")
                    with gr.Row():
                        final_back_btn = gr.Button("Back", variant="secondary")
                        final_save_btn = gr.Button("Save and Return to Chapters", variant="primary")

        # Keep the runtime JS component out of layout flow to avoid panel overlap.
        gr.HTML("", elem_id="vhs-runtime-js", visible=False)

        click_recv = gr.Textbox(
            value="", label="",
            interactive=True, max_lines=1, visible=False,
            elem_id="vhs-click-recv",
        )

    # =========================================================================
    # Rebuild helper - grid + stats + 5 sparklines
    # =========================================================================

    def _rebuild(fids, b64, sigs, overrides, wc, wn, wt, ww,
                 t_mode, iqr_k, tval, bpct, cols, twidth, context, ch_start, show_image_ids):
        gallery_items, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids = build_review_data(
            fids=fids or [],
            b64=b64 or [],
            sigs=sigs or {},
            overrides=overrides or {},
            wc=float(wc),
            wn=float(wn),
            wt=float(wt),
            ww=float(ww),
            t_mode=str(t_mode),
            iqr_k=float(iqr_k),
            tval=float(tval),
            bpct=float(bpct),
            context=int(context),
            chapter_start_frame=int(ch_start),
            show_image_ids=bool(show_image_ids),
        )
        gallery_update = gr.update(value=gallery_items, columns=int(cols))
        return gallery_update, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids

    _RB_OUTS = [grid_gallery, stats_md,
                spark_chroma, spark_noise, spark_tear, spark_wave, spark_score, st_visible_fids]
    _STEP_OUTS = [step_md, load_step, review_step, final_step]

    def _step_updates(step: str):
        if step == "review":
            return (
                "`Step 2/3: Review Frames`",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )
        if step == "final":
            return (
                "`Step 3/3: Finalize`",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        return (
            "`Step 1/3: Load Chapter`",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # -- Archive change -----------------------------------------------------
    def on_archive(archive):
        data = build_archive_state(str(archive or ""))
        start_up = (
            gr.update(value=int(data["start_frame"]))
            if data.get("start_frame") is not None
            else gr.update()
        )
        end_up = (
            gr.update(value=int(data["end_frame"]))
            if data.get("end_frame") is not None
            else gr.update()
        )
        return (
            gr.update(choices=data["titles"], value=data["chapter_value"]),
            data["chapters"],
            data["chapter_titles"],
            gr.update(value=data["chapter_rows"]),
            gr.update(value=data["compact_rows"]),
            start_up,
            end_up,
            data["details"],
            data["status"],
        )

    def on_archive_keep_selected(archive, selected_title):
        data = build_archive_state(str(archive or ""), selected_title=str(selected_title or ""))
        start_up = (
            gr.update(value=int(data["start_frame"]))
            if data.get("start_frame") is not None
            else gr.update()
        )
        end_up = (
            gr.update(value=int(data["end_frame"]))
            if data.get("end_frame") is not None
            else gr.update()
        )
        return (
            gr.update(choices=data["titles"], value=data["chapter_value"]),
            data["chapters"],
            data["chapter_titles"],
            gr.update(value=data["chapter_rows"]),
            gr.update(value=data["compact_rows"]),
            start_up,
            end_up,
            data["details"],
        )

    # -- Chapter change -> frame range ---------------------------------------
    def on_chapter(title, chapters):
        ch = _find_chapter(chapters, title)
        if not ch:
            return gr.update(), gr.update(), "`Select a chapter from the left rail.`"
        return (
            gr.update(value=int(ch["start_frame"])),
            gr.update(value=int(ch["end_frame"])),
            _chapter_details_md(ch),
        )

    chapter_dd.change(on_chapter, [chapter_dd, st_chapters], [start_n, end_n, load_chapter_md])

    def on_chapter_table_pick(chapter_titles, evt: gr.SelectData):
        if evt is None or getattr(evt, "index", None) is None:
            return gr.update(), gr.update()
        idx_raw = evt.index
        try:
            if isinstance(idx_raw, (list, tuple)):
                row_idx = int(idx_raw[0]) if idx_raw else -1
            else:
                row_idx = int(idx_raw)
        except Exception:
            row_idx = -1
        titles = [str(x) for x in (chapter_titles or []) if str(x)]
        if row_idx < 0 or row_idx >= len(titles):
            return gr.update(), gr.update()
        picked = titles[row_idx]
        return gr.update(value=picked), f"`Selected chapter:` **{picked}**"

    chapter_table.select(on_chapter_table_pick, [st_chapter_titles], [chapter_dd, status_md]).then(
        on_chapter, [chapter_dd, st_chapters], [start_n, end_n, load_chapter_md]
    )
    chapter_compact_table.select(on_chapter_table_pick, [st_chapter_titles], [chapter_dd, status_md]).then(
        on_chapter, [chapter_dd, st_chapters], [start_n, end_n, load_chapter_md]
    )
    archive_dd.change(
        on_archive,
        [archive_dd],
        [
            chapter_dd,
            st_chapters,
            st_chapter_titles,
            chapter_table,
            chapter_compact_table,
            start_n,
            end_n,
            load_chapter_md,
            status_md,
        ],
    )
    demo.load(
        on_archive,
        [archive_dd],
        [
            chapter_dd,
            st_chapters,
            st_chapter_titles,
            chapter_table,
            chapter_compact_table,
            start_n,
            end_n,
            load_chapter_md,
            status_md,
        ],
    )

    def on_toggle_rail(is_collapsed):
        next_state = not bool(is_collapsed)
        return next_state, gr.update(visible=not next_state), gr.update(visible=next_state)

    rail_collapse_btn.click(
        on_toggle_rail,
        [st_rail_collapsed],
        [st_rail_collapsed, rail_expanded, rail_collapsed],
    )
    rail_expand_btn.click(
        on_toggle_rail,
        [st_rail_collapsed],
        [st_rail_collapsed, rail_expanded, rail_collapsed],
    )

    # -- Load chapter -------------------------------------------------------
    def on_load(archive, ch_title, chapters, start, end, sample_stride, strict_sampling,
                exact_extract, debug_extract,
                wc, wn, wt, ww, tmode, iqrk, tv, bp, cols, tw, context, show_image_ids,
                progress=gr.Progress()):
        FAIL = (
            "ERROR:  No chapter/video found.",
            [],
            [],
            {},
            {},
            {"fid": -1, "ts": -1},
            gr.update(value=[]),
            "*(no frames loaded)*",
            _E_SIG,
            _E_SIG,
            _E_SIG,
            _E_SIG,
            _E_SCORE,
            [],
        )

        def _status_only(msg: str):
            return (msg, *[gr.update() for _ in range(len(_LOAD_OUTS) - 1)])

        if not archive or not ch_title or ch_title in {CHAPTER_SELECT_LABEL, CHAPTER_MISSING_LABEL}:
            yield FAIL
            return
        video = _resolve_archive_video(str(archive or ""))
        if not video:
            yield FAIL
            return
        start_i, end_i = _normalize_frame_span(int(start), int(end))
        sample_stride_i = max(1, int(sample_stride))
        n_samp = sample_count_from_stride(start_i, end_i, sample_stride_i)
        read_video = video
        frame_read_offset = 0
        debug_overlay = bool(debug_extract) or _env_truthy(TUNER_DEBUG_EXTRACT_ENV) or _env_truthy(
            STEP6_DEBUG_EXTRACT_FRAME_NUMBERS_ENV
        )
        if bool(exact_extract):
            progress(0.0, desc="Preparing chapter extract...")
            extract_target = _chapter_extract_cache_path(
                archive=str(archive or ""),
                chapter_title=str(ch_title or ""),
                ch_start=start_i,
                ch_end=end_i,
                debug_overlay=debug_overlay,
            )
            yield _status_only(
                f"Extracting chapter `{ch_title}` from `{Path(video).name}` "
                f"to `{extract_target.parent.name}`..."
            )
            read_video_p, ex_err = _ensure_step6_chapter_extract(
                source_video=video,
                archive=str(archive or ""),
                chapter_title=str(ch_title or ""),
                ch_start=start_i,
                ch_end=end_i,
                debug_overlay=debug_overlay,
            )
            if ex_err or read_video_p is None:
                F2 = list(FAIL); F2[1] = f"ERROR:  {ex_err or 'Step6-style extraction failed'}"; yield tuple(F2); return
            read_video = read_video_p
            frame_read_offset = start_i
        progress(0.0, desc="Sampling frame signals...")
        yield _status_only(f"Loading sample frames from `{Path(read_video).name}`...")
        # Pass 1: uniform sample for coarse bad-frame detection.
        fids, b64, sigs, err = extract_frames(
            str(read_video), start_i, end_i, int(n_samp),
            archive, ch_title,
            frame_read_offset=frame_read_offset,
            progress=progress,
        )
        if err or fids is None:
            F2 = list(FAIL); F2[1] = f"ERROR:  {err or 'Extraction failed'}"; yield tuple(F2); return

        if not bool(strict_sampling):
            # Pass 2: if coarse sample detects bad frames, prioritize contiguous
            # neighbors around them (no sampling inside those local windows).
            sc0 = combined_score(sigs, wc, wn, wt, ww)
            thr0 = compute_threshold(sc0, tmode, iqrk, tv, bp)
            focus_fids = select_focus_frame_ids(
                start=start_i,
                end=end_i,
                max_frames=int(n_samp),
                coarse_fids=fids,
                coarse_scores=sc0,
                threshold=thr0,
                burst_radius=4,
            )
            if focus_fids != fids:
                progress(0.0, desc="Refining around likely bad ranges...")
                yield _status_only(
                    f"Refining sample around likely bad ranges in `{Path(read_video).name}`..."
                )
                fids, b64, sigs, err = extract_frames(
                    str(read_video), start_i, end_i, int(n_samp),
                    archive, ch_title,
                    frame_ids=focus_fids,
                    frame_read_offset=frame_read_offset,
                    progress=progress,
                )
                if err or fids is None:
                    F2 = list(FAIL); F2[1] = f"ERROR:  {err or 'Extraction failed'}"; yield tuple(F2); return
        # Seed overrides from chapter BAD_FRAMES in chapters.ffmetadata.
        overrides = _chapter_bad_overrides(
            archive=archive,
            chapter_title=ch_title,
            ch_start=int(start),
            ch_end=int(end),
        )
        gallery_update, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids = _rebuild(
            fids, b64, sigs, overrides, wc, wn, wt, ww, tmode, iqrk, tv, bp, cols, tw, context, int(start), bool(show_image_ids)
        )
        source_tag = f"`{Path(read_video).name}`"
        mode_tag = "step6-extract" if bool(exact_extract) else "direct-video"
        debug_tag = " Debug overlay ON (scores may shift)." if debug_overlay else ""
        yield (
            f"OK:  Loaded **{len(fids)}** sampled frames for **{ch_title}** "
            f"from {source_tag} ({mode_tag}), sample rate `1/{sample_stride_i}`."
            f"{debug_tag} Click `Apply and Continue` when ready.",
            fids,
            b64,
            sigs,
            overrides,
            {"fid": -1, "ts": -1},
            gallery_update,
            stats,
            sc_ch,
            sc_no,
            sc_te,
            sc_wa,
            sc_sc,
            vis_fids,
        )

    _LOAD_OUTS = [
        status_md,
        st_fids,
        st_b64,
        st_sigs,
        st_overrides,
        st_last_click,
    ] + _RB_OUTS
    _LOAD_INS  = [archive_dd, chapter_dd, st_chapters,
                  start_n, end_n, sample_stride_sl, strict_sampling_cb, exact_extract_cb, debug_extract_cb,
                  wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
                  cols_sl, twidth_sl, context_sl, thumb_ids_cb]

    def on_after_load(fids):
        return _step_updates("review" if bool(fids) else "load")

    load_next_btn.click(on_load, _LOAD_INS, _LOAD_OUTS).then(
        on_after_load,
        [st_fids],
        _STEP_OUTS,
    )

    def on_save_bad_frames(
        archive,
        ch_title,
        ch_start,
        ch_end,
        fids,
        sigs,
        overrides,
        wc,
        wn,
        wt,
        ww,
        tm,
        ik,
        tv,
        bp,
        progress=gr.Progress(),
    ):
        ch_text = str(ch_title or "").strip().lower()
        if (not archive or not ch_title
                or "select chapter" in ch_text
                or "no chapters" in ch_text):
            return "ERROR:  No chapter selected."
        path, count, analyzed, err = persist_bad_frames_for_chapter(
            archive=str(archive or ""),
            chapter_title=str(ch_title or ""),
            ch_start=int(ch_start),
            ch_end=int(ch_end),
            fids=[int(x) for x in (fids or [])],
            sigs=sigs or {},
            overrides=overrides or {},
            wc=wc,
            wn=wn,
            wt=wt,
            ww=ww,
            tm=tm,
            ik=ik,
            tv=tv,
            bp=bp,
            progress=progress,
        )
        if err:
            return f"ERROR:  {err}"
        return (
            f"Saved:  Saved BAD_FRAMES for **{ch_title}** from loaded frame set "
            f"({analyzed} frame(s) analyzed, {count} marked bad)."
        )

    _SAVE_INS = [
        archive_dd,
        chapter_dd,
        start_n,
        end_n,
        st_fids,
        st_sigs,
        st_overrides,
        wc_sl,
        wn_sl,
        wt_sl,
        ww_sl,
        t_mode,
        iqr_sl,
        tval_sl,
        bpct_sl,
    ]

    def on_prepare_finalize(
        ch_title,
        chapters,
        ch_start,
        ch_end,
        fids,
        sigs,
        overrides,
        vis_fids,
        sample_stride,
        context,
        wc,
        wn,
        wt,
        ww,
        tm,
        ik,
        tv,
        bp,
    ):
        chapter = _find_chapter(chapters or [], str(ch_title or ""))
        return build_finalize_summary(
            chapter_title=str(ch_title or ""),
            chapter=chapter,
            ch_start=int(ch_start),
            ch_end=int(ch_end),
            fids=[int(x) for x in (fids or [])],
            sigs=sigs or {},
            overrides=overrides or {},
            vis_fids=[int(x) for x in (vis_fids or [])],
            sample_stride=int(sample_stride),
            context=int(context),
            wc=float(wc),
            wn=float(wn),
            wt=float(wt),
            ww=float(ww),
            tm=str(tm),
            ik=float(ik),
            tv=float(tv),
            bp=float(bp),
        )

    def on_after_prepare(fids):
        return _step_updates("final" if bool(fids) else "review")

    review_apply_btn.click(
        on_prepare_finalize,
        [
            chapter_dd,
            st_chapters,
            start_n,
            end_n,
            st_fids,
            st_sigs,
            st_overrides,
            st_visible_fids,
            sample_stride_sl,
            context_sl,
            wc_sl,
            wn_sl,
            wt_sl,
            ww_sl,
            t_mode,
            iqr_sl,
            tval_sl,
            bpct_sl,
        ],
        [final_stats_md, status_md],
    ).then(on_after_prepare, [st_fids], _STEP_OUTS)

    review_back_btn.click(lambda: _step_updates("load"), None, _STEP_OUTS)
    final_back_btn.click(lambda: _step_updates("review"), None, _STEP_OUTS)

    def on_after_save_refresh(status_text, archive, selected_title, chapters_state, chapter_titles_state):
        if str(status_text).strip().upper().startswith("ERROR:"):
            return (
                gr.update(),
                chapters_state,
                chapter_titles_state,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        return on_archive_keep_selected(archive, selected_title)

    def on_after_save_step(status_text):
        if str(status_text).strip().upper().startswith("ERROR:"):
            return _step_updates("final")
        return _step_updates("load")

    final_save_btn.click(on_save_bad_frames, _SAVE_INS, [status_md]).then(
        on_after_save_refresh,
        [status_md, archive_dd, chapter_dd, st_chapters, st_chapter_titles],
        [
            chapter_dd,
            st_chapters,
            st_chapter_titles,
            chapter_table,
            chapter_compact_table,
            start_n,
            end_n,
            load_chapter_md,
        ],
    ).then(on_after_save_step, [status_md], _STEP_OUTS)

    # -- Live slider updates ------------------------------------------------
    def on_sliders(ch_start, fids, b64, sigs, ovr,
                   wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, show_image_ids):
        out = _rebuild(
            fids, b64, sigs, ovr, wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, int(ch_start), bool(show_image_ids)
        )
        return out

    _SL_INS = [start_n, st_fids, st_b64, st_sigs, st_overrides,
               wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
               cols_sl, twidth_sl, context_sl, thumb_ids_cb]
    for _s in [wc_sl, wn_sl, wt_sl, ww_sl, iqr_sl, cols_sl, context_sl, thumb_ids_cb]:
        _s.change(on_sliders, _SL_INS, _RB_OUTS)

    # -- Frame click toggle -------------------------------------------------
    def on_click(raw_click, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
                 wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, show_image_ids):
        if not raw_click or not raw_click.strip() or not fids:
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", last_click_event)

        new_ov, new_last_click, srv_dbg = apply_manual_click_override(
            raw_click=raw_click,
            fids=fids,
            sigs=sigs,
            overrides=overrides,
            archive=str(archive or ""),
            chapter_title=str(ch_title or ""),
            ch_start=int(ch_start),
            ch_end=int(ch_end),
            wc=wc,
            wn=wn,
            wt=wt,
            ww=ww,
            tm=tm,
            ik=ik,
            tv=tv,
            bp=bp,
            mark_mode="toggle",
            last_click_event=last_click_event,
        )
        if str(srv_dbg).startswith("ignored:"):
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", new_last_click)

        html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids = _rebuild(
            fids,
            b64,
            sigs,
            new_ov,
            wc,
            wn,
            wt,
            ww,
            tm,
            ik,
            tv,
            bp,
            cols,
            tw,
            context,
            int(ch_start),
            bool(show_image_ids),
        )
        return html, stats, sc_ch, sc_no, sc_te, sc_wa, sc_sc, vis_fids, new_ov, "", new_last_click

    click_recv.input(
        on_click,
        [click_recv, st_fids, st_b64, st_sigs, st_overrides, st_last_click,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl, context_sl, thumb_ids_cb],
        [*_RB_OUTS, st_overrides, click_recv, st_last_click],
        show_progress="minimal",
    )

    def on_gallery_select(vis_fids, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
                          wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, show_image_ids, evt: gr.SelectData):
        if evt is None or getattr(evt, "index", None) is None:
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", last_click_event)
        idx = int(evt.index)
        if idx < 0 or idx >= len(vis_fids):
            return (*[gr.update()] * len(_RB_OUTS), overrides, "", last_click_event)
        fid = int(vis_fids[idx])
        payload = f"{fid}:{int(time.time() * 1000)}"
        return on_click(
            payload, fids, b64, sigs, overrides, last_click_event, archive, ch_title, ch_start, ch_end,
            wc, wn, wt, ww, tm, ik, tv, bp, cols, tw, context, show_image_ids
        )

    grid_gallery.select(
        on_gallery_select,
        [st_visible_fids, st_fids, st_b64, st_sigs, st_overrides, st_last_click,
         archive_dd, chapter_dd, start_n, end_n,
         wc_sl, wn_sl, wt_sl, ww_sl, t_mode, iqr_sl, tval_sl, bpct_sl,
         cols_sl, twidth_sl, context_sl, thumb_ids_cb],
        [*_RB_OUTS, st_overrides, click_recv, st_last_click],
        show_progress="minimal",
    )

# -- Launch --------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
        css=_DARK_CSS,
        allowed_paths=["C:/Users/covec/Videos/Clips"],
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )





