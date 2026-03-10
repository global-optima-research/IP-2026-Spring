#!/bin/bash
# run_ect_inference.sh — Run ECT student inference using trained checkpoint
# Uses the final checkpoint (iter 6000) from ECT training
#
# Usage: bash run_ect_inference.sh

set -euo pipefail

source /data/chenqingzhan/miniconda3/bin/activate fastgen

export CUDA_VISIBLE_DEVICES=5
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
CKPT_PATH="/data/chenqingzhan/fastgen_output/fastgen/wan_cm_ct/ect_wan1.3b_4gpu/checkpoints/0006000"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

echo "=== ECT Student Inference ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT_PATH"
echo "Start: $(date)"

torchrun --nproc_per_node=1 --standalone \
    scripts/inference/video_model_inference.py \
    --do_student_sampling True \
    --ckpt_path "$CKPT_PATH" \
    --config fastgen/configs/experiments/WanT2V/config_cm_ct.py \
    - trainer.seed=42 \
      trainer.ddp=False \
      model.guidance_scale=5.0 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      log_config.name=ect_inference_6000 \
      log_config.wandb_mode=disabled

echo "=== Inference complete ==="
echo "End: $(date)"
