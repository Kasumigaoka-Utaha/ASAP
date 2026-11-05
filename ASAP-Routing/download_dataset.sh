#!/usr/bin/env bash

# Download the Google Drive-hosted dataset with wget, handling the confirmation token Google adds for large files.

set -euo pipefail

FILE_ID="1meYCOULaX_ckosg46Bv1rK8f5sbxMHHV"
OUTPUT="${1:-dataset.zip}"

tmpdir=$(mktemp -d)
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

cookie_file="$tmpdir/cookies.txt"

# First request grabs the confirmation token (if Google requires one) and saves cookies.
confirm_page=$(wget \
  --quiet \
  --save-cookies "$cookie_file" \
  --keep-session-cookies \
  --no-check-certificate \
  "https://docs.google.com/uc?export=download&id=${FILE_ID}" \
  -O-)

# Extract confirm token if present.
confirm_token=$(printf '%s' "$confirm_page" | sed -n 's/.*confirm=\\([^&]*\\).*/\\1/p' | head -n1)

download_url="https://docs.google.com/uc?export=download&id=${FILE_ID}"
if [ -n "$confirm_token" ]; then
  download_url="${download_url}&confirm=${confirm_token}"
fi

wget \
  --load-cookies "$cookie_file" \
  --no-check-certificate \
  "$download_url" \
  -O "$OUTPUT"

echo "Saved dataset to $OUTPUT"
