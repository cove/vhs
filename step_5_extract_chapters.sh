#!/usr/bin/env bash
# Extract chapters to separate files, preserving creation_time
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./extract_chapters.sh file1.mkv [file2.mkv ...] [chapter_name]"
    exit 1
fi

# Last arg may be a chapter filter (optional)
CHAPTER_FILTER=""
last="${@: -1}"
if [[ $# -gt 1 && ! -f "$last" ]]; then
    CHAPTER_FILTER="$last"
    set -- "${@:1:$(($#-1))}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG="${SCRIPT_DIR}/bin/ffmpeg"

[[ ! -x "$FFMPEG" ]] && echo "ERROR: ffmpeg not found or not executable: $FFMPEG" && exit 1

filters_video="$SCRIPT_DIR/filters_video.cfg"
filters_audio="$SCRIPT_DIR/filters_audio.cfg"
[[ ! -f "$filters_video" ]] && echo "Error: video filter file not found: $filters_video" && exit 1
[[ ! -f "$filters_audio" ]] && echo "Error: audio filter file not found: $filters_audio" && exit 1

VIDEO_FILTER_CHAIN=$(grep -v '^\s*#' "$filters_video" | sed '/^\s*$/d' | paste -sd, - | tr -s ',')
AUDIO_FILTER_CHAIN=$(grep -v '^\s*#' "$filters_audio" | sed '/^\s*$/d' | paste -sd, - | tr -s ',')

echo "Using video filter chain: $VIDEO_FILTER_CHAIN"
echo "Using audio filter chain: $AUDIO_FILTER_CHAIN"
echo ""

process_chapter() {
    local in="$1"
    local start_ns="$2"
    local end_ns="$3"
    local title="$4"
    local creation_time="$5"

    [[ -z "$start_ns" || -z "$end_ns" || -z "$title" ]] && return
    [[ -n "$CHAPTER_FILTER" && "$title" != "$CHAPTER_FILTER" ]] && return

    local start_sec end_sec
    start_sec=$(awk "BEGIN{printf \"%.3f\", $start_ns/1e9}")
    end_sec=$(awk "BEGIN{printf \"%.3f\", $end_ns/1e9}")

    local safe_title out_file
    safe_title=$(echo "$title" | tr '/\\:*?"<>|' '_')
    out_file="${safe_title}.mp4"

    echo "  -> $out_file"

# From Grok AI:
# =============================================================================
# FINAL x265 SETTINGS – GOLD STANDARD FOR VHS / VHS-C AFTER QTGMC (2025)
#
# These parameters are ONLY safe and ideal AFTER a proper QTGMC pass.
# Never use them on raw interlaced captures – they will preserve combing and jitter!
#
#   -x265-params "no-sao=1:psy-rd=2.0:psy-rq=2.0:aq-mode=3:deblock=-2,-2"
#
# Parameter       What it does                                          Why it belongs AFTER QTGMC (and nowhere else)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# no-sao=1        Disables Sample Adaptive Offset (SAO)                 QTGMC already removed all block/combing artefacts.
#                                                               SAO would now only attack real tape grain → turns snow into plastic.
#                                                               Mandatory OFF for analog sources.
#
# psy-rd=2.0      Psychovisual RD optimisation                          QTGMC recovered extremely fine luma detail and stabilised grain.
#                                                               High psy-rd forces x265 to preserve that detail instead of flattening it.
#
# psy-rq=2.0      Psychovisual quantisation tuning                      Protects the natural “breathing” motion of tape grain that QTGMC restored.
#
# aq-mode=3       Variance + edge-aware Adaptive Quantisation           QTGMC output is full of correct high-frequency noise (grain).
#                                                               Only mode 3 prevents ugly pooling/worms in grainy areas.
#
# deblock=-2,-2   Very light positive deblocking                        QTGMC + SourceMatch/Lossless can leave <0.1 % residual ringing.
#                                                               Tiny deblock cleans the last traces without harming real grain.
#
# This exact string is used on every single top-tier VHS/VHS-C release in 2025.
# It is the current universal best practice for final archival encodes.
# =============================================================================

    "$FFMPEG" -nostdin -v error -i "$in" \
        -ss "$start_sec" -to "$end_sec" \
        -pix_fmt yuv422p \
        -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 \
        -tag:v hvc1 \
        -vf "$VIDEO_FILTER_CHAIN" \
        -c:v libx265 -preset slower -crf 16 \
        -x265-params "no-sao=1:psy-rd=2.0:psy-rq=2.0:aq-mode=3:deblock=-2,-2" \
        -af "$AUDIO_FILTER_CHAIN" \
        -c:a aac -b:a 48k -ac 1 -ar 48000 \
        -movflags +faststart \
        -metadata "title=$title" \
        -metadata "creation_time=$creation_time" \
        -metadata "comment=Extracted chapter from $in (video_filter_chain=$VIDEO_FILTER_CHAIN, audio_filter_chain=$AUDIO_FILTER_CHAIN)" \
        -y "$out_file"
}

process_file() {
    local in="$1"

    if [[ ! -f "$in" ]]; then
        echo "ERROR: File not found: $in (skipping)"
        echo ""
        return
    fi

    echo "Extracting chapters from: $in"

    tmp_meta=$(mktemp)
    "$FFMPEG" -nostdin -v error -i "$in" -f ffmetadata -y "$tmp_meta"

    START=""; END=""; TITLE=""; CREATION_TIME=""

    while IFS= read -r line; do
        case "$line" in
            "[CHAPTER]"*)
                process_chapter "$in" "$START" "$END" "$TITLE" "$CREATION_TIME"
                START=""; END=""; TITLE=""; CREATION_TIME=""
                ;;
            "START="*)
                START="${line#START=}"
                ;;
            "END="*)
                END="${line#END=}"
                ;;
            "title="*)
                TITLE="${line#title=}"
                ;;
            "creation_time="*)
                CREATION_TIME="${line#creation_time=}"
                ;;
        esac
    done < "$tmp_meta"

    process_chapter "$in" "$START" "$END" "$TITLE" "$CREATION_TIME"

    rm -f "$tmp_meta"
    echo "Done."
    echo ""
}

for in in "$@"; do
    process_file "$in"
done

echo "All files processed."
