#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: process_mkv_all.sh video.mkv"
  exit 1
fi

INPUT="$1"
[[ ! -f "$INPUT" ]] && echo "ERROR: $INPUT not found." && exit 1

FILENAME="$(basename "$INPUT")"
BASENAME="${FILENAME%.*}"

# Extract video name prefix up to first number
VIDEO_NAME="$(printf '%s' "$BASENAME" | sed -E 's/^([^0-9]*[0-9]+).*/\1/')"

META_DIR="$SCRIPT_DIR/media_metadata/$VIDEO_NAME"

COVER="$META_DIR/cover.jpg"
LABEL="$META_DIR/label.jpg"
TITLE_FILE="$META_DIR/title.txt"
COMMENT_FILE="$META_DIR/comment.txt"
CHAPTERS="$META_DIR/chapters.ffmetadata"

# Validate required metadata files
for f in "$COVER" "$LABEL" "$TITLE_FILE" "$COMMENT_FILE" "$CHAPTERS"; do
  [[ ! -f "$f" ]] && echo "ERROR: Missing expected metadata file: $f" && exit 1
done

TITLE=$(<"$TITLE_FILE")
COMMENT=$(<"$COMMENT_FILE")

echo "Processing \"$INPUT\" ..."
echo "Cleaning metadata, removing attachments, applying VHS tags, then adding extras..."

TMP1=$(mktemp -t mkv_clean.XXXXXX.mkv)
TMP2=$(mktemp -t mkv_final.XXXXXX.mkv)

cleanup() {
  [[ -f "$TMP1" ]] && rm -f "$TMP1"
  [[ -f "$TMP2" ]] && rm -f "$TMP2"
}
trap cleanup EXIT

#
# STEP 1 — Clean attachments + reset metadata + set VHS color info
#
ffmpeg -nostdin -v error -i "$INPUT" \
    -map 0 \
    -map -0:t \
    -c copy \
    -metadata title="" \
    -metadata comment="" \
    -metadata:s:v:0 encoder="" \
    -metadata:s:a:0 encoder="" \
    -metadata:s:v:0 field_order="BFF" \
    -color_primaries:v 6 \
    -color_trc:v 6 \
    -colorspace:v 5 \
    -aspect 4:3 \
    "$TMP1"

#
# STEP 2 — Add chapters, title/comment, and new attachments
#

# Helper to lowercase extension
ext_lower() {
  local f="$1"
  local e="${f##*.}"
  printf '%s' "$e" | tr '[:upper:]' '[:lower:]'
}

args=()
args+=(-i "$TMP1")
args+=(-i "$CHAPTERS")
args+=(-map 0 -c copy -map_metadata 1)
args+=(-metadata "title=$TITLE" -metadata "comment=$COMMENT")

idx=0
cover_ext=$(ext_lower "$COVER")
args+=(-attach "$COVER" \
       -metadata:s:t:$idx mimetype=image/jpeg \
       -metadata:s:t:$idx filename="cover.$cover_ext")
idx=$((idx+1))

label_ext=$(ext_lower "$LABEL")
args+=(-attach "$LABEL" \
       -metadata:s:t:$idx mimetype=image/jpeg \
       -metadata:s:t:$idx filename="label.$label_ext")
idx=$((idx+1))

# Perform final mux
if ffmpeg -nostdin -v error "${args[@]}" -f matroska "$TMP2" -y; then
  OUT="${BASENAME}_metadata_extras.mkv"
  mv -f "$TMP2" "$OUT"
  trap - EXIT
  echo "Done."
  echo "Output: $OUT"
else
  echo "ERROR: ffmpeg failed."
  exit 1
fi
