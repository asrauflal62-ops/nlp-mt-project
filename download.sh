#!/usr/bin/env bash
set -e

echo "[1/3] Prepare folders..."
mkdir -p Data

echo "[2/3] Download data package..."
# TODO: replace with your real direct download link
# Example:
# wget -O data.zip "https://your-direct-link/data.zip"
echo "Please edit download.sh and replace the URL with your data.zip direct link."
exit 1

echo "[3/3] Unzip..."
unzip -o data.zip -d Data

echo "Done."

