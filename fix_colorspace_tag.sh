#!/bin/zsh

# Usage: ./update_colorspace_tag.sh input_file.mkv

if [ -z "$1" ]; then
    echo "Usage: $(basename "$0") input_file.mkv"
    exit 1
fi

INPUT="$1"
FILENAME="${INPUT##*/}"
BASENAME="${FILENAME%.*}"

echo "Updating colorspace metadata for \"$INPUT\"..."

# Update MKV metadata (no re-encoding)
# For standard-definition BT.601 / SMPTE 170M, the correct numeric IDs are:
#  - color_primaries: 6 (SMPTE 170M)
#  - color_trc: 6 (SMPTE 170M)
#  - colorspace: 5 (SMPTE 170M)
# Reference: https://en.wikipedia.org/wiki/Color_space#List_of_color_spaces
ffmpeg -i "$INPUT" \
  -c copy \
  -color_primaries:v 6 \
  -color_trc:v 6 \
  -colorspace:v 5 \
  "${BASENAME}_fixed.mkv"

echo "Done! Output: ${BASENAME}_fixed.mkv"
