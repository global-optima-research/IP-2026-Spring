#!/bin/bash
# run_causvid_inference.sh — Run CausVid (DMD-distilled) inference on Wan2.1-1.3B
#
# CausVid: 3-step DMD distillation (CVPR 2025), bidirectional checkpoint2
# Uses GPU 0 (single GPU, ~15-20GB VRAM expected)
#
# Usage: bash run_causvid_inference.sh

set -euo pipefail

source /data/chenqingzhan/miniconda3/bin/activate fastgen

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

CAUSVID_DIR="/data/chenqingzhan/CausVid"
CKPT_DIR="/data/chenqingzhan/causvid_checkpoints/bidirectional_checkpoint2"
OUTPUT_DIR="/data/chenqingzhan/fastgen_output/causvid_samples"
PROMPT_FILE="/data/chenqingzhan/scripts/eval_prompts.txt"

cd $CAUSVID_DIR
export PYTHONPATH=$CAUSVID_DIR:${PYTHONPATH:-}

echo "=== CausVid Inference ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Prompts: $PROMPT_FILE"
echo "Start: $(date)"

python minimal_inference/bidirectional_inference.py \
    --config_path configs/wan_bidirectional_dmd_from_scratch.yaml \
    --checkpoint_folder "$CKPT_DIR" \
    --output_folder "$OUTPUT_DIR" \
    --prompt_file_path "$PROMPT_FILE"

echo "=== Inference complete ==="
echo "End: $(date)"
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -10
