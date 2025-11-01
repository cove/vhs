#!/bin/bash

shopt -s nullglob

for FILE in *.mp4 *.mkv; do
    BASENAME="${FILE%.*}"   # Remove extension
    BLAKE3_FILE="$BASENAME blake3.txt"

    # Check BLAKE3 hash and output
    b3sum -c "$BLAKE3_FILE"
done

