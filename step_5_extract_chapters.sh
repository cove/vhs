#!/usr/bin/env bash
# Extract chapters to separate files, skipping "Start Capture" / "End Capture"
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./extract_chapters.sh input.mkv [chapter_name]"
    exit 1
fi

IN="$1"
BASE="${IN%.*}"
CHAPTER_FILTER="${2:-}"  # Optional: only export this chapter

# get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FFMPEG=${SCRIPT_DIR}/bin/ffmpeg

[[ ! -f "$IN" ]] && echo "ERROR: Input file not found: $IN" && exit 1
[[ ! -x "$FFMPEG" ]] && echo "ERROR: ffmpeg not found or not executable: $FFMPEG" && exit 1

# load filter file from script dir
filters_video="$SCRIPT_DIR/filters_video.cfg"
if [[ ! -f "$filters_video" ]]; then
    echo "Error: video filter file not found: $filters_video"
    exit 1
fi
VIDEO_FILTER_CHAIN=$(grep -v '^\s*#' "$filters_video" | sed '/^\s*$/d' | paste -sd, -| tr -s ',')
echo "Using video filter chain: $VIDEO_FILTER_CHAIN"

filters_audio="$SCRIPT_DIR/filters_audio.cfg"
if [[ ! -f "$filters_audio" ]]; then
    echo "Error: audio filter file not found: $filters_audio"
    exit 1
fi
AUDIO_FILTER_CHAIN=$(grep -v '^\s*#' "$filters_audio" | sed '/^\s*$/d' | paste -sd, -| tr -s ',')
echo "Using audio filter chain: $AUDIO_FILTER_CHAIN"

echo "Extracting chapters from ${IN}..."

# Export chapters to ffmetadata format
$FFMPEG -nostdin -v error -i "$IN" -f ffmetadata -y /tmp/chapters_ffmeta.txt

process_chapter() {
    local start_ns="$1"
    local end_ns="$2"
    local title="$3"

    [[ ! -n "$start_ns" || ! -n "$end_ns" || ! -n "$title" ]] && return
    [[ "$title" == *"Capture Start"* ]] || [[ "$title" == *"Capture End"* ]] && return
    [[ -n "$CHAPTER_FILTER" && "$title" != "$CHAPTER_FILTER" ]] && return

    # Convert nanoseconds → seconds
    local start_sec
    local end_sec
    start_sec=$(awk "BEGIN{printf \"%.3f\", $start_ns/1000000000}")
    end_sec=$(awk "BEGIN{printf \"%.3f\", $end_ns/1000000000}")

    # Safe filename
    local safe_title
    safe_title=$(echo "$title" | tr '/\\:*?"<>|' '_')

    local out_file="${safe_title}.mp4"

    echo "Extracting chapter to file: $out_file"
    $FFMPEG -nostdin -v error -i "$IN" \
      -ss "$start_sec" -to "$end_sec" \
      -pix_fmt yuv420p \
      -color_primaries:v 6 -color_trc:v 6 -colorspace:v 5 -color_range:v 1 \
      -vf "$VIDEO_FILTER_CHAIN" \
      -c:v libx265 -preset slow -crf 20 -profile:v main \
      -af "$AUDIO_FILTER_CHAIN" \
      -c:a aac -b:a 41.1k -ac 1 -ar 44100 \
      -movflags +faststart \
      -metadata "title=$title" \
      -metadata "comment=Extracted chapter from $IN (video_filter_chain=$VIDEO_FILTER_CHAIN, audio_filter_chain=$AUDIO_FILTER_CHAIN)" -y \
      "$out_file"
}

START=""
END=""
TITLE=""

while IFS= read -r line; do
    case "$line" in

        "[CHAPTER]"*)
            # process previous chapter if complete
            process_chapter "$START" "$END" "$TITLE"
            START=""
            END=""
            TITLE=""
            ;;

        "START="*)
            START="${line#START=}"
            ;;

        "END="*)
            END="${line#END=}"
            ;;

        "title="*)
            TITLE="${line#title=}"
            ;;
    esac
done < /tmp/chapters_ffmeta.txt

# Process last chapter
process_chapter "$START" "$END" "$TITLE"

echo "Chapter extraction complete."
