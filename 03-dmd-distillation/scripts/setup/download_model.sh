#!/bin/bash
# download_model.sh — Download Wan2.1-T2V-1.3B model weights
# Usage: bash download_model.sh

set -euo pipefail

PYTHON="/data/chenqingzhan/miniconda3/envs/fastgen/bin/python"
SAVE_DIR="/data/chenqingzhan/.cache/huggingface"
HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN environment variable}"

echo "=== Downloading Wan2.1-T2V-1.3B-Diffusers ==="
$PYTHON /data/chenqingzhan/hf-download.py \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --token "$HF_TOKEN" \
    --save_dir "$SAVE_DIR"

echo "=== Download complete ==="
echo "Model saved to: $SAVE_DIR/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
