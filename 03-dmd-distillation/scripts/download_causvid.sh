#!/bin/bash
# download_causvid.sh — Download CausVid checkpoint and setup inference environment
# CausVid: DMD-based distillation for Wan2.1-1.3B (4-step, CVPR 2025)
# Only downloads bidirectional_checkpoint2 (11.4GB) — best quality variant
#
# Usage: bash download_causvid.sh

set -euo pipefail

export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

CAUSVID_DIR="/data/chenqingzhan/CausVid"
CKPT_DIR="/data/chenqingzhan/causvid_checkpoints"

source /data/chenqingzhan/miniconda3/bin/activate fastgen

echo "=== Step 1: Clone CausVid repo ==="
if [ -d "$CAUSVID_DIR" ]; then
    echo "CausVid repo already exists at $CAUSVID_DIR, skipping clone"
else
    cd /data/chenqingzhan
    git clone https://github.com/tianweiy/CausVid.git
    echo "CausVid repo cloned"
fi

echo ""
echo "=== Step 2: Download bidirectional_checkpoint2 (11.4GB) ==="
mkdir -p "$CKPT_DIR/bidirectional_checkpoint2"

# Use huggingface-cli to download only the specific checkpoint folder
huggingface-cli download tianweiy/CausVid \
    bidirectional_checkpoint2/model.pt \
    --local-dir "$CKPT_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "=== Step 3: Verify download ==="
if [ -f "$CKPT_DIR/bidirectional_checkpoint2/model.pt" ]; then
    SIZE=$(du -sh "$CKPT_DIR/bidirectional_checkpoint2/model.pt" | cut -f1)
    echo "SUCCESS: bidirectional_checkpoint2/model.pt downloaded ($SIZE)"
else
    echo "ERROR: model.pt not found!"
    exit 1
fi

echo ""
echo "=== Step 4: Install CausVid dependencies ==="
cd "$CAUSVID_DIR"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt 2>&1 | tail -5
    echo "Dependencies installed"
fi
if [ -f "setup.py" ]; then
    pip install -e . 2>&1 | tail -3
    echo "CausVid package installed"
fi

echo ""
echo "=== Download complete ==="
echo "Checkpoint: $CKPT_DIR/bidirectional_checkpoint2/model.pt"
echo "CausVid code: $CAUSVID_DIR"
echo ""
echo "To run inference:"
echo "  cd $CAUSVID_DIR"
echo "  python minimal_inference/bidirectional_inference.py \\"
echo "    --config_path configs/wan_bidirectional_dmd_from_scratch.yaml \\"
echo "    --checkpoint_folder $CKPT_DIR/bidirectional_checkpoint2 \\"
echo "    --output_folder /data/chenqingzhan/fastgen_output/causvid_samples \\"
echo "    --prompt_file_path <prompt_file>"
