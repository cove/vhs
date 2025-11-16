#!/usr/bin/env zsh
set -euo pipefail
setopt NULL_GLOB

# script directory
SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"

# find b3 tool (prefer b3sum, then blake3, then bundled binary in script dir)
if command -v b3sum >/dev/null 2>&1; then
  B3="$(command -v b3sum)"
elif command -v blake3 >/dev/null 2>&1; then
  B3="$(command -v blake3)"
elif [[ -x "$SCRIPT_DIR/b3sum" ]]; then
  B3="$SCRIPT_DIR/b3sum"
elif [[ -x "$SCRIPT_DIR/b3sum_macos" ]]; then
  B3="$SCRIPT_DIR/b3sum_macos"
else
  echo "Error: b3sum or blake3 not found in PATH or script directory."
  exit 1
fi

# ffprobe required
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "Error: ffprobe not found in PATH. Please install ffmpeg."
  exit 1
fi

OUTBLAKE="00-manifest-blake3sums.txt"
rm -f -- "$OUTBLAKE"

found=false
for f in *.mp4 *.mkv; do
  if [[ -f "$f" ]]; then
    found=true
    echo
    echo "Processing: $f"

    # append blake3 line for the media file
    "$B3" "$f" >> "$OUTBLAKE"
    if [[ $? -ne 0 ]]; then
      echo "Error generating blake3 for \"$f\""
      exit 1
    fi

    # generate ffprobe-based media mediainfo (text)
    bn="${f%.*}"
    mediainfo_txt="${bn} mediainfo.txt"
    echo "Generating mediainfo for \"$f\" -> \"$mediainfo_txt\""
    ffprobe -v quiet -show_format -show_streams "$f" > "$mediainfo_txt"
    if [[ $? -ne 0 ]]; then
      echo "Error generating ffprobe mediainfo for \"$f\""
      exit 1
    fi

    # append blake3 line for the generated media mediainfo file
    "$B3" "$mediainfo_txt" >> "$OUTBLAKE"
    if [[ $? -ne 0 ]]; then
      echo "Error generating blake3 for \"$mediainfo_txt\""
      exit 1
    fi
  fi
done

if ! $found; then
  echo "No .mp4 or .mkv files found in $PWD"
  exit 0
fi

echo
echo "Combined blake3 file: $OUTBLAKE"
echo "All files processed successfully."
exit 0