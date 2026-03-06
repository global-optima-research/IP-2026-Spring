#!/bin/bash
# download_openvid.sh — Download OpenVid-1M dataset (subset) to shared /data/datasets/
# Uses hf-mirror.com since huggingface.co is blocked on this server.
#
# Usage:
#   bash download_openvid.sh          # Download default 7 parts (~50K videos)
#   bash download_openvid.sh 3        # Download first 3 parts
#   bash download_openvid.sh all_csv  # Download only the CSV metadata

set -euo pipefail

MIRROR="https://hf-mirror.com"
REPO="datasets/nkp37/OpenVid-1M"
BASE_URL="${MIRROR}/${REPO}/resolve/main"
DEST="/data/datasets/OpenVid-1M"
VIDEO_DIR="${DEST}/videos"
NUM_PARTS="${1:-7}"

echo "=== OpenVid-1M Download Script ==="
echo "Mirror: ${MIRROR}"
echo "Destination: ${DEST}"

mkdir -p "${DEST}" "${VIDEO_DIR}"

# Step 1: Download CSV metadata (always)
# NOTE: CSV is under data/train/ subdirectory in the HF repo
echo ""
echo "=== Step 1: Downloading CSV metadata ==="
if [ ! -f "${DEST}/OpenVid-1M.csv" ]; then
    wget -c -O "${DEST}/OpenVid-1M.csv" "${BASE_URL}/data/train/OpenVid-1M.csv"
    echo "CSV downloaded: $(wc -l < "${DEST}/OpenVid-1M.csv") lines"
else
    echo "CSV already exists, skipping. $(wc -l < "${DEST}/OpenVid-1M.csv") lines"
fi

if [ "${NUM_PARTS}" = "all_csv" ]; then
    echo "CSV-only mode, done."
    exit 0
fi

# Step 2: Download video zip parts
# Each part is ~30-50GB, contains ~7-10K videos
# 7 parts ≈ 50-70K videos, enough for Phase 0
PARTS=(0 1 2 3 4 5 6)
if [ "${NUM_PARTS}" -gt 0 ] 2>/dev/null; then
    PARTS=()
    for i in $(seq 0 $((NUM_PARTS - 1))); do
        PARTS+=($i)
    done
fi

echo ""
echo "=== Step 2: Downloading ${#PARTS[@]} video zip parts ==="

for i in "${PARTS[@]}"; do
    ZIP_NAME="OpenVid_part${i}.zip"
    ZIP_PATH="${DEST}/${ZIP_NAME}"

    if [ -f "${ZIP_PATH}.done" ]; then
        echo "[${i}/${#PARTS[@]}] ${ZIP_NAME} already downloaded, skipping."
        continue
    fi

    echo "[${i}/${#PARTS[@]}] Downloading ${ZIP_NAME}..."
    wget -c -O "${ZIP_PATH}" "${BASE_URL}/${ZIP_NAME}"

    # Mark as complete
    touch "${ZIP_PATH}.done"
    echo "[${i}/${#PARTS[@]}] ${ZIP_NAME} done."
done

# Step 3: Extract videos
echo ""
echo "=== Step 3: Extracting videos ==="

for i in "${PARTS[@]}"; do
    ZIP_NAME="OpenVid_part${i}.zip"
    ZIP_PATH="${DEST}/${ZIP_NAME}"

    if [ -f "${ZIP_PATH}.extracted" ]; then
        echo "[${i}] ${ZIP_NAME} already extracted, skipping."
        continue
    fi

    if [ -f "${ZIP_PATH}" ]; then
        echo "[${i}] Extracting ${ZIP_NAME}..."
        unzip -joq "${ZIP_PATH}" -d "${VIDEO_DIR}"
        touch "${ZIP_PATH}.extracted"
        echo "[${i}] Extracted."
    fi
done

echo ""
echo "=== Download complete ==="
echo "Videos in: ${VIDEO_DIR}"
echo "Total videos: $(ls "${VIDEO_DIR}"/*.mp4 2>/dev/null | wc -l)"
echo "CSV: ${DEST}/OpenVid-1M.csv"
echo ""
echo "Next step: run convert_to_webdataset.py to prepare training data."
