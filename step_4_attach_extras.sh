#!/usr/bin/env zsh
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: attach_extras.sh \"video.mkv\" \"cover.jpg\" \"label.jpg\" [\"notes.txt\"] [\"title\"] [\"comment\"] [\"chapters.ffmetadata\"]"
  exit 1
fi

MKV="$1"
COVER="$2"
LABEL="$3"
NOTES="${4:-}"
TITLE="${5:-}"
COMMENT="${6:-}"
CHAPTERS="${7:-}"

if [[ ! -f "$MKV" ]]; then
  echo "ERROR: $MKV not found."
  exit 1
fi

if [[ ! -f "$COVER" ]]; then
  echo "ERROR: $COVER not found."
  exit 1
fi

if [[ -n "$LABEL" && ! -f "$LABEL" ]]; then
  echo "WARNING: $LABEL not found. Skipping label."
  LABEL=""
fi

if [[ -n "$NOTES" && ! -f "$NOTES" ]]; then
  echo "WARNING: $NOTES not found. Skipping notes."
  NOTES=""
fi

if [[ -n "$CHAPTERS" && ! -f "$CHAPTERS" ]]; then
  echo "WARNING: $CHAPTERS not found. Skipping chapters."
  CHAPTERS=""
fi

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
# input(s)
args+=(-i "$MKV")
if [[ -n "$CHAPTERS" ]]; then
  args+=(-i "$CHAPTERS")
fi

# map input streams from main file
args+=(-map 0 -c copy)

# if chapters provided as ffmetadata, map it as metadata source (input index 1)
if [[ -n "$CHAPTERS" ]]; then
  args+=(-map_metadata 1)
fi

# global metadata: title/comment (if provided)
if [[ -n "$TITLE" ]]; then
  args+=(-metadata "title=$TITLE")
fi
if [[ -n "$COMMENT" ]]; then
  args+=(-metadata "comment=$COMMENT")
fi

# attachments with fixed stored filenames (cover/label/notes) regardless of source filename
idx=0
cover_ext=$(ext_lower "$COVER")
cover_stored="cover.${cover_ext}"
cover_name=$(basename -- "$COVER")
args+=(-attach "$COVER" -metadata:s:t:$idx mimetype=image/jpeg -metadata:s:t:$idx filename="$cover_stored")
idx=$((idx+1))

if [[ -n "$LABEL" ]]; then
  label_ext=$(ext_lower "$LABEL")
  label_stored="label.${label_ext}"
  label_name=$(basename -- "$LABEL")
  args+=(-attach "$LABEL" -metadata:s:t:$idx mimetype=image/jpeg -metadata:s:t:$idx filename="$label_stored")
  idx=$((idx+1))
fi

if [[ -n "$NOTES" ]]; then
  notes_ext=$(ext_lower "$NOTES")
  # default to txt if no extension
  if [[ -z "$notes_ext" ]]; then
    notes_ext="txt"
  fi
  notes_stored="notes.${notes_ext}"
  notes_name=$(basename -- "$NOTES")
  args+=(-attach "$NOTES" -metadata:s:t:$idx mimetype=text/plain -metadata:s:t:$idx filename="$notes_stored")
  idx=$((idx+1))
fi

# run ffmpeg and replace original on success
if ffmpeg "${args[@]}" -f matroska "$TMP" -y; then
  mv -f "$TMP" "${MKV%.*}_extras.mkv"
  trap - EXIT
  echo "Done."
else
  echo "ERROR: ffmpeg failed."
  exit 1
fi
