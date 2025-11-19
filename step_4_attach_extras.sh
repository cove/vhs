#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: attach_extras.sh \"video.mkv\" \"cover.jpg\" \"label.jpg\" \"notes.txt\" \"title.txt\" \"comment.txt\" \"chapters.ffmetadata\""
  exit 1
fi

MKV="$1"
COVER="$2"
LABEL="$3"
NOTES="$4"
TITLE_FILE="$5"
COMMENT_FILE="$6"
CHAPTERS="$7"

# validate inputs
for f in "$MKV" "$COVER" "$LABEL" "$NOTES" "$TITLE_FILE" "$COMMENT_FILE" "$CHAPTERS"; do
  [[ ! -f "$f" ]] && echo "ERROR: $f not found." && exit 1
done

# read title/comment from files
TITLE=$(<"$TITLE_FILE")
COMMENT=$(<"$COMMENT_FILE")

echo "Updating \"$MKV\" ..."

TMP=$(mktemp -t attach_extras.XXXXXX.mkv)
cleanup() {
  [[ -f "$TMP" ]] && rm -f "$TMP"
}
trap cleanup EXIT

# helper to get lowercase extension (without dot)
ext_lower() {
  local f="$1"
  local e="${f##*.}"
  printf '%s' "${e}" | tr '[:upper:]' '[:lower:]'
}

# Build ffmpeg args
args=()
args+=(-i "$MKV")
args+=(-i "$CHAPTERS")  # always use chapters

# map streams
args+=(-map 0 -c copy -map_metadata 1)

# global metadata
args+=(-metadata "title=$TITLE" -metadata "comment=$COMMENT")

# attachments
idx=0
cover_ext=$(ext_lower "$COVER")
cover_stored="cover.${cover_ext}"
args+=(-attach "$COVER" -metadata:s:t:$idx mimetype=image/jpeg -metadata:s:t:$idx filename="$cover_stored")
idx=$((idx+1))

label_ext=$(ext_lower "$LABEL")
label_stored="label.${label_ext}"
args+=(-attach "$LABEL" -metadata:s:t:$idx mimetype=image/jpeg -metadata:s:t:$idx filename="$label_stored")
idx=$((idx+1))

notes_ext=$(ext_lower "$NOTES")
[[ -z "$notes_ext" ]] && notes_ext="txt"
notes_stored="notes.${notes_ext}"
args+=(-attach "$NOTES" -metadata:s:t:$idx mimetype=text/plain -metadata:s:t:$idx filename="$notes_stored")
idx=$((idx+1))

# run ffmpeg
if ffmpeg "${args[@]}" -f matroska "$TMP" -y; then
  mv -f "$TMP" "${MKV%.*}_extras.mkv"
  trap - EXIT
  echo "Done."
else
  echo "ERROR: ffmpeg failed."
  exit 1
fi

echo "Output: ${MKV%.*}_extras.mkv"