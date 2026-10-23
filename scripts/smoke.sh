#!/usr/bin/env bash
# Fast end-to-end check on the 40-row fixture: no network, under a minute.
set -euo pipefail

TMP_DIR="${TMPDIR:-/tmp}/fakenews_smoke"
mkdir -p "$TMP_DIR"

python -m fakenews.data.clean \
  --input tests/fixtures/tiny.csv \
  --output "$TMP_DIR/tiny_clean.parquet" \
  --min-len 20

python -m pytest -q
