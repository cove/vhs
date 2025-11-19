#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: attach_extras.sh video.mkv"
  exit 1
fi

MKV="$1"
[[ ! -f "$MKV" ]] && echo "ERROR: $MKV not found." && exit 1

VIDEO_BASE="$(basename "$MKV")"
VIDEO_BASE="${VIDEO_BASE%.*}"

VIDEO_NAME="$(printf '%s' "$VIDEO_BASE" | sed -E 's/^([^0-9]*[0-9]+).*/\1/')"

META_DIR="$SCRIPT_DIR/media_metadata/$VIDEO_NAME"

COVER="$META_DIR/cover.jpg"
LABEL="$META_DIR/label.jpg"
TITLE_FILE="$META_DIR/title.txt"
COMMENT_FILE="$META_DIR/comment.txt"
CHAPTERS="$META_DIR/chapters.ffmetadata"

# validate metadata dir + required files
for f in "$COVER" "$LABEL" "$TITLE_FILE" "$COMMENT_FILE" "$CHAPTERS"; do
  [[ ! -f "$f" ]] && echo "ERROR: Missing expected metadata file: $f" && exit 1
done

TITLE=$(<"$TITLE_FILE")
COMMENT=$(<"$COMMENT_FILE")

echo "Updating \"$MKV\" ..."

TMP=$(mktemp -t attach_extras.XXXXXX.mkv)
cleanup() {
  [[ -f "$TMP" ]] && rm -f "$TMP"
}
trap cleanup EXIT

ext_lower() {
  local f="$1"
  local e="${f##*.}"
  printf '%s' "$e" | tr '[:upper:]' '[:lower:]'
}

args=()
args+=(-i "$MKV")
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

if ffmpeg -nostdin -v error "${args[@]}" -f matroska "$TMP" -y; then
  OUT="${MKV%.*}_extras.mkv"
  mv -f "$TMP" "$OUT"
  trap - EXIT
  echo "Done."
  echo "Output: $OUT"
else
  echo "ERROR: ffmpeg failed."
  exit 1
fi
