#!/bin/bash
# prepare_training_data.sh — Full pipeline: download OpenVid-1M subset + convert to WebDataset
#
# Usage: bash prepare_training_data.sh
# Run on server (111.17.197.107) with fastgen conda env activated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/data/chenqingzhan/miniconda3/envs/fastgen/bin/python"
DATASET_ROOT="/data/datasets/OpenVid-1M"
WEBDATASET_DIR="${DATASET_ROOT}/webdataset"
NUM_PARTS=7
MAX_SAMPLES=50000

echo "=== Phase 0: Training Data Preparation ==="
echo "Dataset root: ${DATASET_ROOT}"
echo "Target: ${MAX_SAMPLES} samples from ${NUM_PARTS} zip parts"
echo ""

# Step 1: Download
echo ">>> Step 1: Download OpenVid-1M (${NUM_PARTS} parts)"
bash "${SCRIPT_DIR}/download_openvid.sh" "${NUM_PARTS}"

# Step 2: Convert to WebDataset
echo ""
echo ">>> Step 2: Convert to WebDataset format"
${PYTHON} "${SCRIPT_DIR}/convert_to_webdataset.py" \
    --csv "${DATASET_ROOT}/OpenVid-1M.csv" \
    --video_dir "${DATASET_ROOT}/videos" \
    --output_dir "${WEBDATASET_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --shard_size 1000 \
    --min_frames 81 \
    --min_aesthetic 5.0 \
    --min_seconds 2.0

# Step 3: Set permissions for team sharing
echo ""
echo ">>> Step 3: Setting shared permissions"
chmod -R 755 "${DATASET_ROOT}"

echo ""
echo "=== All done ==="
echo "WebDataset shards: ${WEBDATASET_DIR}/"
echo "Shared at: ${DATASET_ROOT} (readable by all team members)"
echo ""
echo "To use in FastGen training configs, set:"
echo "  dataloader_train.datatags=[\"WDS:${WEBDATASET_DIR}/shard-{000000..000049}.tar\"]"
