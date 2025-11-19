#!/usr/bin/env bash
# Extract chapters from an MP4 to separate files, skipping "Start Capture" / "End Capture"
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: ./extract_chapters.sh input.mp4"
    exit 1
fi

IN="$1"
BASE="${IN%.*}"  # filename without extension

echo "Extracting chapters from ${IN}..."

# Export chapters to ffmetadata format
ffmpeg -i "$IN" -f ffmetadata -y /tmp/chapters_ffmeta.txt

chapter_index=1
START=""
END=""
TITLE=""

process_chapter() {
    [[ -z "$TITLE" || -z "$START" || -z "$END" ]] && return

    # skip unwanted chapters
    if [[ "$TITLE" == *"Capture Start"* ]] || [[ "$TITLE" == *"Capture End"* ]]; then
        echo "Skipping chapter: $TITLE"
        return
    fi

    START_SEC=$(awk "BEGIN{printf \"%.3f\", $START/1000}")
    END_SEC=$(awk "BEGIN{printf \"%.3f\", $END/1000}")

    SAFE_TITLE=$(echo "$TITLE" | tr '/\\:*?"<>|' '_')
    OUT_FILE="${BASE} ${SAFE_TITLE}.mp4"

    echo "Extracting chapter: $TITLE -> $OUT_FILE"
    ffmpeg -i "$IN" -ss "$START_SEC" -to "$END_SEC" -c copy "$OUT_FILE"

    ((chapter_index++))
}

while IFS= read -r line; do
    case "$line" in
        "[CHAPTER]"*)
            # process previous chapter if any
            process_chapter
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

# process last chapter
process_chapter

echo "All chapters extracted."
