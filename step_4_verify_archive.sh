#!/bin/bash
# Verifies all files in the archive against the checksum manifest (SHA3-256 or legacy BLAKE3)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/step_4_verify_archive.py" "$@"
exit $?
