#!/usr/bin/env bash
# Verifies all files on the drive against the BLAKE3 checksum manifest

# --- Set paths ---
DRIVE_DIR="../../"
DRIVE_CHECKSUM_FILE="Archive/00-drive-manifest-blake3sums.txt"
B3SUM_BIN="Archive/scripts/bin/b3sum"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "ERROR: This script must be run on macOS."
    exit 1
fi

echo "Verifying: $DRIVE_CHECKSUM_FILE"
echo ""

# --- Run b3sum verification ---
cd "$DRIVE_DIR" || { echo "Failed to enter directory $DRIVE_DIR"; exit 1; }

$B3SUM_BIN -c "$DRIVE_CHECKSUM_FILE"
RETURN_CODE=$?

echo

if [ $RETURN_CODE -eq 0 ]; then
    echo "ALL FILES VERIFIED — CHECKSUMS MATCH!"
else
    echo "SOME FILES FAILED VERIFICATION!"
fi

echo "Verify manifest: $DRIVE_CHECKSUM_FILE"
echo "All done."

exit $RETURN_CODE
