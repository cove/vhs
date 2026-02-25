VHS Digital Archive Project
===========================

This project contains tools and scripts for capturing, processing, and archiving VHS and U-matic tapes.
Most scripts assume a Windows environment for full AviSynth/QTGMC filtering. On Linux/macOS, the pipeline now uses an FFmpeg fallback path where AviSynth is unavailable.

Directory Structure
-------------------

bin/                - Helper binaries used by scripts (e.g., mediainfo)
common.py           - Shared Python functions, paths, and FFmpeg/metadata helpers
freeze.py           - Freezes installed Python packages into requirements.txt and archives wheels for target platforms
manuals/            - User guides or printed instructions for VCRs, capture cards, and workflow
metadata/           - Original per-archive metadata, including chapters and comments
models/             - Whisper AI speech-to-text models for transcription
package_archive/    - Archived Python packages downloaded for reproducibility
screenshots/        - Screenshots of capture and VirtualDub and VCR et al equipment settings for reference
software/           - Third-party software required for capture and encoding (VirtualDub, UT Video, FFmpeg-QTGMC Easy)
scripts/            - Optional scripts for setup or automation
test/               - Test environment for running scripts without affecting the main archive
venv-mac/           - Python virtual environment for macOS
requirements.txt    - Python dependencies used by scripts
setup.py            - Optional setup script for installing packages
vhs.py              - Unified CLI entrypoint for most workflows
vhs_pipeline/       - Shared command and conversion modules used by vhs.py and step wrappers

Unified CLI (Preferred)
-----------------------

vhs.py
    - Single command surface for conversion, metadata, proxy, render, compare, and checksum actions.
    - Example commands:
      python vhs.py convert avi tape1.avi tape2.avi
      python vhs.py convert umatic reel1.mov reel2.mov
      python vhs.py metadata build
      python vhs.py metadata embed Archive/callahan_01_archive.mkv
      python vhs.py verify archive
      python vhs.py proxy
      python vhs.py render --title "birthday"
      python vhs.py compare --archive callahan_01_archive
      python vhs.py checksum drive
      python vhs.py verify drive

Legacy Step Scripts (Still Supported)
-------------------------------------

step_1_capture_vhs_to_avi.txt
    - Instructions for capturing VHS tapes to AVI using VirtualDub.
    - Only captures; does not process or deinterlace.
    - Requires Windows and an analog capture card.

step_2_convert_umatic_prores_mov_to_archive.py
    - Converts U-matic or ProRes MOV files to the archive folder as MKV.
    - Uses FFV1 10-bit 4:2:2 video and 24-bit PCM audio for preservation.

step_2_convert_avi_to_mkv.py
    - Converts captured AVI files to archival MKV in-place.
    - Uses FFV1 4:2:2 video with SD color metadata tags and 16-bit PCM audio.

step_3_generate_archive_metadata.py
    - Reads archive files and existing chapters.ffmetadata to generate markers and mediainfo outputs.
    - Copies metadata folders into the archive and writes SHA3-256 checksums.

step_4_verify_archive.py
    - Verifies that all archive files exist and are not corrupted using SHA3-256 checksums
      (also supports legacy BLAKE3 manifests).

step_5_make_proxies.py
    - Generates proxy MP4 files (_proxy.mp4) from archive MKVs for easier playback.
    - Uses ffmpeg passthrough frame sync mode so proxy frame cadence/order matches source archives.

step_6_make_videos.py
    - Main script to produce final video clips from archive.
    - Performs deinterlacing and applies filters (Windows-only).
    - Extracts audio, transcribes with Whisper, generates subtitles (.srt, .vtt, .ass), and encodes final videos.
    - Chapter extraction uses frame-derived exact timestamps to keep chapter-local frame indices aligned with archive frames.
    - Manual bad-frame repair sidecar: metadata/<archive>/badframes.tsv (archive-global start/end frame ranges, applied automatically before QTGMC; optional source_frame column can force a specific replacement frame per range; optional no_pad boolean column (true/false) disables automatic pad per row; very long ranges are skipped unless note includes allow_long; adaptive pre-pad defaults to 0 for single-frame, 1 for 2-3 frame bursts, 2 for 4+ frame bursts; note supports no_pad or pad= / pad_before= / pad_after= as fallback).

step_7_generate_drive_checksum.py
    - Creates a SHA3-256 checksum manifest for the full drive/archive.

step_8_verify_drive_checksum.py
    - Verifies the drive checksum manifest created in step_7 (SHA3-256 by default,
      supports legacy BLAKE3 manifests).

step_14_make_original_chapter_comparisons.py
    - Creates side-by-side chapter comparison videos.
    - Left side is the original chapter section, sourced from `<archive>_proxy.mp4` for speed.
    - Right side is the processed chapter MP4 from step_6 output.
    - Outputs to `Clips/chapter_comparisons/<archive>/` by default.

Usage Notes
-----------

1. All commands/scripts rely on the paths defined in common.py. Adjust paths if moving the project.
2. On Linux/macOS, step_6_make_videos.py now uses an FFmpeg `bwdif` fallback when AviSynth/QTGMC is unavailable. AviSynth `.avs` filter scripts (including chapter-specific QTGMC tuning) are Windows-only and are skipped on Linux/macOS.
3. Virtual environments are platform-specific: venv-win/ for Windows, venv-mac/ for macOS, venv-linux/ for Linux.
4. Make sure all software dependencies are installed (VirtualDub, UT Video codec, FFmpeg-QTGMC Easy, drivers for capture cards).
5. Linux FFmpeg archives in `bin/` are kept compressed for Git compatibility; `setup.py` extracts `bin/ffmpeg` and `bin/ffprobe` on setup. MediaInfo is expected from the system package manager (e.g. `apt-get install mediainfo`) or `MEDIAINFO_BIN`.
6. `vhs.py` is the preferred interface. `step_*.py` files are compatibility wrappers.

General Workflow
----------------

1. Capture tapes to AVI following step_1_capture_vhs_to_avi.txt.
2. Convert captured files into the archive:
   - AVI: `python vhs.py convert avi <files...>`
   - U-matic/ProRes MOV: `python vhs.py convert umatic <files...>`
3. Generate archive metadata: `python vhs.py metadata build`
4. Verify archive integrity: `python vhs.py verify archive`
5. Optionally generate proxy videos: `python vhs.py proxy`
6. Process archive into final clips with subtitles: `python vhs.py render [step_6 args]`
7. Optionally generate/verify drive checksums:
   - `python vhs.py checksum drive`
   - `python vhs.py verify drive`
