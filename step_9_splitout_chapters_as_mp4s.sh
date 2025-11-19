#!/usr/bin/env bash
# Extract chapters from an MP4 to separate files, skipping "Start Capture" / "End Capture"
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./extract_chapters.sh input.mp4"
    exit 1
fi

IN="$1"
BASE="${IN%.*}"

echo "Extracting chapters from ${IN}..."

# Export chapters to ffmetadata format
ffmpeg -nostdin -v error -i "$IN" -f ffmetadata -y /tmp/chapters_ffmeta.txt

process_chapter() {
    local start_ns="$1"
    local end_ns="$2"
    local title="$3"

    [[ ! -n "$start_ns" || ! -n "$end_ns" || ! -n "$title" ]] && return
    [[ "$title" == *"Capture Start"* ]] || [[ "$title" == *"Capture End"* ]] && return

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

    ffmpeg -nostdin -v error -i "$IN" \
      -ss "$start_sec" -to "$end_sec" \
      -vf "setfield=bff,yadif=0,zscale=matrixin=6:matrix=1:transferin=6:transfer=1:primariesin=6:primaries=1:rangein=0:range=0,eq=saturation=0.90:gamma=1.0,crop=in_w-2:in_h-6:0:0,scale=640:480,setsar=1" \
      -pix_fmt yuv420p \
      -metadata "title=$title" \
      -c:v libx264 -preset slow -crf 20 -profile:v main \
      -c:a aac -b:a 41.1k -ac 1 -ar 44100 \
      -movflags +faststart \
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

echo "All chapters extracted."

