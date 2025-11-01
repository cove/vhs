#!/bin/bash

# Check for input argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 input_file.mkv"
    exit 1
fi

INPUT="$1"

# Strip directory path and extension
FILENAME=$(basename "$INPUT")
BASENAME="${FILENAME%.*}"

# Remove "Master" (case-insensitive) from start of filename
BASENAME_NO_MASTER=$(echo "$BASENAME" | sed -E 's/Master//')

# Trim leading and ending spaces if any
BASENAME_NO_MASTER=$(echo "$BASENAME_NO_MASTER" | sed 's/^ *//')
BASENAME_NO_MASTER="${BASENAME_NO_MASTER%"${BASENAME_NO_MASTER##*[![:space:]]}"}"

# Write out video info
vhs_info() {
    local input="$1"
    local BASENAME="$(basename "$input")"
    BASENAME="${BASENAME%.*}"

    ffprobe -v quiet -print_format json -show_format -show_streams "$input" > "${BASENAME} info.json"
    mediainfo "$input" > "${BASENAME} info.txt"

    b3sum "$input" > "${BASENAME} blake3.txt"
}

# Write out master info
vhs_info "$INPUT"

# Normal, good quality
ffmpeg -i "$INPUT" -vf "yadif=1" -c:v libx264 -profile:v high -preset slow -crf 16 -c:a aac -movflags +faststart "${BASENAME_NO_MASTER} Normal.mp4"
vhs_info "${BASENAME_NO_MASTER}.mp4"

# Small, fast downloading
ffmpeg -i "$INPUT" -c:v libx264 -profile:v baseline -preset slow -crf 20 -vf scale=640:-2 -c:a aac -b:a 48k "${BASENAME_NO_MASTER} Small.mp4"
vhs_info "${BASENAME_NO_MASTER} Small.mp4"


