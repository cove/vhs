#!/bin/bash
# Verifies all files in the archive against the BLAKE3 checksum manifest

# --- Set paths (adjust as needed) ---
ARCHIVE_DIR="../"
ARCHIVE_CHECKSUM_FILE="00-archive-manifest-blake3sums.txt"
B3SUM_BIN="scripts/bin/b3sum"

echo "Verifying: $ARCHIVE_CHECKSUM_FILE"
echo ""

# --- Run b3sum verification ---
cd "$ARCHIVE_DIR" || { echo "Failed to enter directory $ARCHIVE_DIR"; exit 1; }

$B3SUM_BIN -c "$ARCHIVE_CHECKSUM_FILE"
RETURN_CODE=$?

if [ $RETURN_CODE -eq 0 ]; then
    echo "ALL FILES VERIFIED — CHECKSUMS MATCH!"
else
    echo "SOME FILES FAILED VERIFICATION!"
fi

exit $RETURN_CODE

