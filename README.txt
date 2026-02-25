VHS Digital Archive Project
===========================

This repo contains a VHS/U-matic ingest and processing pipeline.
The primary interface is `vhs.py` (single CLI).

Quick Start
-----------

1. Create/install the project environment:
   - `python setup.py`
2. Run the unified CLI help:
   - `python vhs.py -h`

Main Commands
-------------

Convert source media into archive MKV:
- `python vhs.py convert avi <file1.avi> <file2.avi> ...`
- `python vhs.py convert umatic <file1.mov> <file2.mov> ...`

Build metadata + archive checksums:
- `python vhs.py metadata build`

Embed ffmetadata into existing archive MKVs (no re-encode):
- `python vhs.py metadata embed <archive1.mkv> <archive2.mkv> ...`

Verify manifests:
- `python vhs.py verify archive [--sha3|--blake3] [manifest_path]`
- `python vhs.py verify drive [--sha3|--blake3] [manifest_path]`

Generate proxies:
- `python vhs.py proxy`

Render delivery clips/videos (forwards args to render pipeline):
- `python vhs.py render [step_6 args]`

Build original-vs-processed chapter comparison videos:
- `python vhs.py compare [--archive ...] [--title ...] [--height ...]`

Generate drive-level checksum manifest:
- `python vhs.py checksum drive`

Interactive Tools
-----------------

Bad-frame tuner (Gradio UI):
- `python vhs_tuner.py`

Tracking-loss classifier utility:
- `python tracking_loss.py -h`

Directory Notes
---------------

- `metadata/` contains per-archive metadata (`chapters.ffmetadata`, markers, etc.).
- `vhs_pipeline/` contains command implementations used by `vhs.py`.
- `step_*.py` files are legacy entrypoints that now call `vhs_pipeline` modules.
- `models/`, `software/`, `manuals/`, `screenshots/` contain model/data/tool references.

Platform Notes
--------------

- Windows supports full AviSynth/QTGMC paths in rendering.
- Linux/macOS use FFmpeg fallback deinterlacing where AviSynth is unavailable.
- Linux FFmpeg archives in `bin/` stay compressed for Git compatibility; `setup.py`
  extracts runtime binaries into `bin/`.

