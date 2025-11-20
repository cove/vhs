#!/usr/bin/env bash
set -euo pipefail

# Usage: truncate_video.sh input.mp4 output.mp4 01:04:28

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 input_file output_file duration"
    echo "Example: $0 input.mp4 output.mp4 01:04:28"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
DURATION="$3"

ffmpeg -nostdin -v error -i "$INPUT" -t "$DURATION" -c copy "$OUTPUT"

echo "Trim complete: $OUTPUT"
