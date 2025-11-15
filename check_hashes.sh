#!/bin/bash

shopt -s nullglob

BLAKE3_FILE="00-manifest-blake3sums.txt"

# Check BLAKE3 hash and output
b3sum -c "$BLAKE3_FILE"
