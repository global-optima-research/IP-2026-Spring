#!/bin/bash
# run_ladd_single_gpu.sh — Run LADD (Latent Adversarial Diffusion Distillation) on Wan2.1-1.3B with single GPU
# Usage: bash run_ladd_single_gpu.sh
#
# LADD: Pure adversarial distillation (no score distillation loss, GAN-only)
# Same data format as DMD2 (VideoLoaderConfig: mp4+txt WebDataset)
# Reference: Sauer et al., 2024 (https://arxiv.org/abs/2403.12015)

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
    --config=fastgen/configs/experiments/WanT2V/config_ladd.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=8 \
      trainer.max_iter=6000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=500 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"WDS:$DATA_DIR\"]" \
      log_config.name=ladd_wan1.3b_single_gpu \
      2>&1 | tee /data/chenqingzhan/fastgen_output/ladd_train.log
