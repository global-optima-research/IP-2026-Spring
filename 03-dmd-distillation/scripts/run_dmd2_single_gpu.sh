#!/bin/bash
# run_dmd2_single_gpu.sh — Run DMD2 distillation on Wan2.1-1.3B with single GPU
# Usage: bash run_dmd2_single_gpu.sh

set -euo pipefail

# Environment
export CUDA_VISIBLE_DEVICES=0
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

CONDA_DIR="/data/chenqingzhan/miniconda3"
FASTGEN_DIR="/data/chenqingzhan/FastGen"
PYTHON="$CONDA_DIR/envs/fastgen/bin/python"

cd $FASTGEN_DIR

# Single GPU training: use python directly (not torchrun with nproc>1)
# Key adjustments for single GPU:
#   - trainer.ddp=False (no distributed data parallel)
#   - trainer.fsdp=False (no model sharding)
#   - trainer.batch_size_global=8 (reduced from 64, auto grad_accum=8)
#   - max_iter=6000 (same as default)
$PYTHON train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=8 \
      trainer.max_iter=6000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=500 \
      log_config.name=dmd2_wan1.3b_single_gpu \
      2>&1 | tee /data/chenqingzhan/fastgen_output/dmd2_train.log
