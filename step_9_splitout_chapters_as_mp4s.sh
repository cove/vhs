#!/usr/bin/env bash
# Extract chapters from an MP4 to separate files, skipping "Start Capture" / "End Capture"
set -euo pipefail
set -x

if [[ $# -lt 1 ]]; then
    echo "Usage: ./extract_chapters.sh input.mp4"
    exit 1
fi

IN="$1"
BASE="${IN%.*}"  # filename without extension

echo "Extracting chapters from ${IN}..."

# Export chapters to ffmetadata format
ffmpeg -i "$IN" -f ffmetadata -y /tmp/chapters_ffmeta.txt

# Extract global title if it exists
GLOBAL_TITLE=$(awk -F= '/^title=/ {print $2; exit}' /tmp/chapters_ffmeta.txt || echo "$BASE")

chapter_index=1

process_chapter() {
    local start_ns="$1"
    local end_ns="$2"
    local title="$3"

    local start_sec
    local end_sec
    start_sec=$(awk "BEGIN{printf \"%.3f\", $start_ns/1000000000}")
    end_sec=$(awk "BEGIN{printf \"%.3f\", $end_ns/1000000000}")

    local safe_title
    safe_title=$(echo "$title" | tr '/\\:*?"<>|' '_')
    local chapter_meta_title="${GLOBAL_TITLE} - ${title}"
    local out_file="${chapter_meta_title}.mp4"

    echo "Extracting chapter: $title -> $out_file"
    ffmpeg -i "$IN" \
      -ss "$start_sec" -to "$end_sec" \
      -vf "setfield=bff,yadif=0,colorspace=all=5:iall=5,eq=saturation=0.90:gamma=1.0,crop=in_w-2:in_h-6:0:0,scale=640:480,setsar=1" \
      -pix_fmt yuv420p \
      -flags +ilme+ildct \
      -map_metadata 0:g -metadata encoder="" \
      -c:v libx264 -preset slow -crf 20 -profile:v baseline \
      -c:a aac -b:a 41.1k -ac 1 -ar 44100 \
      -movflags +faststart \
      -metadata title="$chapter_meta_title" \
      "$out_file"

    ((chapter_index++))
}

START=""
END=""
TITLE=""

while IFS= read -r line; do
    case "$line" in
        "[CHAPTER]"*)
            # process previous chapter if any
            if [[ -n "$START" && -n "$END" && -n "$TITLE" ]]; then
                # skip unwanted chapters
                if [[ "$TITLE" != "Capture Start" ]] && [[ "$TITLE" != "Capture End" ]]; then
                    process_chapter "$START" "$END" "$TITLE"
                fi
            fi
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
