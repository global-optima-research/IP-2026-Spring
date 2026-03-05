#!/bin/bash
# run_fdistill_single_gpu.sh — Run f-distill (f-divergence DMD2) on Wan2.1-1.3B with single GPU
# Usage: bash run_fdistill_single_gpu.sh
#
# f-distill: Improved DMD2 with f-divergence weighting (JS divergence)
# Same data format as DMD2 (VideoLoaderConfig: mp4+txt WebDataset)
# Reference: Xu et al., 2025 (https://arxiv.org/abs/2502.15681)

set -euo pipefail

# Environment
export CUDA_VISIBLE_DEVICES=0
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

CONDA_DIR="/data/chenqingzhan/miniconda3"
FASTGEN_DIR="/data/chenqingzhan/FastGen"
PYTHON="$CONDA_DIR/envs/fastgen/bin/python"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="${1:-/data/chenqingzhan/training_data/video_shards}"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

$PYTHON train.py \
    --config=fastgen/configs/experiments/WanT2V/config_fdistill.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=8 \
      trainer.max_iter=6000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=500 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"WDS:$DATA_DIR\"]" \
      log_config.name=fdistill_wan1.3b_single_gpu \
      2>&1 | tee /data/chenqingzhan/fastgen_output/fdistill_train.log
