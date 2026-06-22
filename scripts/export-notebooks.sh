#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="notebooks"
OUTPUT_DIR="public/notebooks"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "No notebooks directory found; skipping notebook export."
  exit 0
fi

if ! find "$SOURCE_DIR" -name "*.ipynb" -print -quit | grep -q .; then
  echo "No notebooks found; skipping notebook export."
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
jupyter nbconvert --to html "$SOURCE_DIR"/*.ipynb --output-dir "$OUTPUT_DIR"
