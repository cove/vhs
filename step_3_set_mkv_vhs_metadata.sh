#!/bin/zsh

# Usage: ./step_3_set_mkv_vhs_metadata.sh input_file.mkv

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
#  - interlaced_frame: 1 (indicates interlaced video)
#  - aspect ratio: 4:3
#  - Remove encoder metadata from video and audio streams so they can be compared fairly
#  - --flags +ilme+ildct -vf "setfield=bff" Bottom Field First for interlaced video (BFF)
ffmpeg -i "$INPUT" \
  -c copy \
  -color_primaries:v 6 \
  -color_trc:v 6 \
  -colorspace:v 5 \
  -interlaced_frame 1 \
  -aspect 4:3 \
  -metadata:s:v:0 encoder="" \
  -metadata:s:a:0 encoder="" \
  -flags +ilme+ildct -vf "setfield=bff" \
  "${BASENAME}_metadata.mkv"

echo "Done! Output: ${BASENAME}_metadata.mkv"
