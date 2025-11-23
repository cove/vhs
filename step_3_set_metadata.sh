#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: process_mkv_all.sh *.mkv"
  exit 1
fi

# get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG="${SCRIPT_DIR}/bin/ffmpeg"

# Helper to lowercase extension
ext_lower() {
  local f="$1"
  local e="${f##*.}"
  printf '%s' "$e" | tr '[:upper:]' '[:lower:]'
}

for INPUT in "$@"; do
  if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: $INPUT not found. Skipping."
    echo
    continue
  fi

  FILENAME="$(basename "$INPUT")"
  BASENAME="${FILENAME%.*}"
  OUTPUT="${BASENAME}_metadata.mkv"

  # Extract video name prefix up to first number
  VIDEO_NAME="$(printf '%s' "$BASENAME" | sed -E 's/^([^0-9]*[0-9]+).*/\1/')"

  META_DIR="$SCRIPT_DIR/media_metadata/$VIDEO_NAME"

  COVER="$META_DIR/cover.jpg"
  TITLE_FILE="$META_DIR/title.txt"
  COMMENT_FILE="$META_DIR/comment.txt"
  CHAPTERS="$META_DIR/chapters.ffmetadata"

  # Validate required metadata files
  missing=false
  for f in "$COVER" "$TITLE_FILE" "$COMMENT_FILE" "$CHAPTERS"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: Missing expected metadata file: $f"
      missing=true
    fi
  done
  if [[ "$missing" == true ]]; then
    echo "Skipping $INPUT"
    echo
    continue
  fi

  TITLE=$(<"$TITLE_FILE")
  COMMENT=$(<"$COMMENT_FILE")

  echo "Processing \"$INPUT\" -> \"$OUTPUT\"..."
  echo "Applying VHS tags and attachments..."

  "$FFMPEG" -nostdin -v error -i "$INPUT" \
    -f ffmetadata -i "$CHAPTERS" \
    -map 0:v:0 -map 0:a \
    -map_metadata 0 \
    -map_chapters -1 \
    -map_chapters 1 \
    -c copy \
    -metadata title="$TITLE" \
    -metadata comment="$COMMENT" \
    -attach "$COVER" \
    -metadata:s:t:0 mimetype=image/jpeg \
    -metadata:s:t:0 filename="cover.$(ext_lower "$COVER")" \
    -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -aspect 4:3 \
    -f matroska "$OUTPUT" -y

  echo "Done."
  echo "Output: $OUTPUT"
  echo
done

echo "All files processed."
