VHS Digital Archive Project
===========================

This project contains tools and scripts for capturing, processing, and archiving VHS and U-matic tapes.
Most scripts assume a Windows environment for deinterlacing and filter application. On macOS/Linux, only basic processing works.

Directory Structure
-------------------

bin/                - Helper binaries used by scripts (e.g., b3sum, mediainfo)
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

Top-Level Scripts
-----------------

step_1_capture_vhs_to_avi.txt
    - Instructions for capturing VHS tapes to AVI using VirtualDub.
    - Only captures; does not process or deinterlace.
    - Requires Windows and an analog capture card.

step_2_convert_umatic_prores_mov_to_archive.py
    - Converts U-matic or ProRes MOV files to the archive folder as MKV.
    - Copies metadata if available.

step_2_convert_virtualdub_to_archive.py
    - Converts VirtualDub AVI captures into the archive folder as MKV.
    - Ensures proper naming and storage.

step_3_generate_archive_metadata.py
    - Reads archive files and generates chapter metadata in metadata/ directory.
    - Populates chapters.ffmetadata files.

step_4_verify_archive.py
    - Verifies that all archive files exist and are not corrupted using BLAKE3 checksums.

step_5_make_proxies.py
    - Generates lower-resolution proxy files (_proxy.mp4) from archive MKVs for easier playback.

step_6_make_videos.py
    - Main script to produce final video clips from archive.
    - Performs deinterlacing and applies filters (Windows-only).
    - Extracts audio, transcribes with Whisper, generates subtitles (.srt, .vtt, .ass), and encodes final videos.

step_7_generate_drive_checksum.py
    - Creates a BLAKE3 checksum manifest for the full drive/archive for backup verification.

step_8_verify_drive_checksum.py
    - Verifies the drive checksum manifest created in step_7.

Usage Notes
-----------

1. All scripts rely on the paths defined in common.py. Adjust paths if moving the project.
2. On macOS/Linux, deinterlacing and AVS filters in step_6_make_videos.py will be skipped; Windows is required for full processing.
3. Virtual environments are platform-specific: venv-win/ for Windows, venv-mac/ for macOS.
4. Make sure all software dependencies are installed (VirtualDub, UT Video codec, FFmpeg-QTGMC Easy, drivers for capture cards).

General Workflow
----------------

1. Capture tapes to AVI following step_1_capture_vhs_to_avi.txt.
2. Convert captured files into the archive using step_2 scripts.
3. Generate archive metadata (step_3).
4. Verify archive integrity (step_4).
5. Optionally, generate proxy videos for quick review (step_5).
6. Process archive into final clips with subtitles (step_6).
7. Optionally, generate and verify drive-level checksums (steps 7–8).
