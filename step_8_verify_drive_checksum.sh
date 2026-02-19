#!/usr/bin/env bash
# Verifies all files on the drive against the checksum manifest (SHA3-256 or legacy BLAKE3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/step_8_verify_drive_checksum.py" "$@"
exit $?
