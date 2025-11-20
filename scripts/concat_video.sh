#!/usr/bin/env bash
# Usage: ./concat_videos.sh part1.mp4 part2.mp4 ... output.mp4

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 part1.mp4 part2.mp4 ... output.mp4"
    exit 1
fi

# Last argument is the output
OUTPUT="${!#}"

# All other arguments are input parts
INPUTS=("${@:1:$#-1}")

# Verify inputs exist and get full paths
for i in "${INPUTS[@]}"; do
    if [[ ! -f "$i" ]]; then
        echo "Error: file not found: $i"
        exit 1
    fi
done

# Create temporary list file
LISTFILE=$(mktemp)
for f in "${INPUTS[@]}"; do
    # use absolute path
    FULLPATH="$(pwd)/$f"
    echo "file '$FULLPATH'" >> "$LISTFILE"
done

# Run FFmpeg concat
ffmpeg -f concat -safe 0 -i "$LISTFILE" -c copy "$OUTPUT"

# Clean up
rm "$LISTFILE"

echo "Merge complete: $OUTPUT"
