#!/usr/bin/env bash
# Usage: ./delete_time_range.sh input.mp4 start_time end_time output.mp4
# Example: ./delete_time_range.sh video.mkv 00:07:20 00:08:24 output.mkv

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 input.mp4 start_time end_time output.mp4"
    exit 1
fi

INPUT="$1"
START_TIME="$2"
END_TIME="$3"
OUTPUT="$4"

# Temporary files
PART1=$(mktemp).mkv
PART2=$(mktemp).mkv
LISTFILE=$(mktemp)

# Get start and end times in seconds using FFmpeg-friendly format
# Extract first part (before the section to remove)
ffmpeg -nostdin -v error -i "$INPUT" -ss 0 -to "$START_TIME" -c copy "$PART1"

# Extract second part (after the section to remove)
ffmpeg -nostdin -v error -i "$INPUT" -ss "$END_TIME" -c copy "$PART2"

# Create concat list
echo "file '$PART1'" > "$LISTFILE"
echo "file '$PART2'" >> "$LISTFILE"

# Concatenate the two parts
ffmpeg -nostdin -v error -f concat -safe 0 -i "$LISTFILE" -c copy "$OUTPUT"

# Clean up temporary files
rm "$PART1" "$PART2" "$LISTFILE"

echo "Section removed: $OUTPUT"
