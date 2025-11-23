#!/usr/bin/env bash
# step_2_convert_avi_to_mkv.sh
# Usage: ./step_2_convert_avi_to_mkv.sh captured.avi [another.avi ...]

set -euo pipefail

# Path to ffmpeg (adjust if not in PATH, or use absolute path)
FFMPEG="$(command -v ffmpeg)" || { echo "Error: ffmpeg not found in PATH"; exit 1; }

# Directory where this script lives (so it works when called from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If you keep ffmpeg in ./bin/ffmpeg like in Windows version, uncomment this:
# FFMPEG="$SCRIPT_DIR/bin/ffmpeg"
# [[ -x "$FFMPEG" ]] || { echo "Error: ffmpeg not found at $FFMPEG"; exit 1; }

# Check if at least one argument was given
(( $# == 0 )) && {
    echo "Usage: ${0##*/} file1.avi [file2.avi ...]"
    exit 1
}

for INPUT in "$@"; do
    [[ -f "$INPUT" ]] || { echo "Error: File not found: $INPUT"; continue; }

    BASENAME="$(basename "$INPUT" .avi)"
    OUTPUT="${BASENAME}_archive.mkv"

    echo "Creating MKV encoded as FFv1 archive: $OUTPUT ..."

    "$FFMPEG" -nostdin -v error -i "$INPUT" \
        -pix_fmt yuv422p \
        -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 \
        -map 0:v:0 -c:v ffv1 \
            -level 3 \
            -g 1 \
            -coder 1 \
            -context 1 \
            -slices 24 \
            -slicecrc 1 \
        -map 0:a:0 -c:a pcm_s16le \
        -y "$OUTPUT"

    echo "Done: $INPUT → $OUTPUT"
    echo
done

echo "All files processed."