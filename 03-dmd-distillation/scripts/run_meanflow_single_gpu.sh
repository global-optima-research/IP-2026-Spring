#!/bin/bash
# run_meanflow_single_gpu.sh — Run MeanFlow on Wan2.1-1.3B with single GPU
# Usage: bash run_meanflow_single_gpu.sh
#
# MeanFlow: Consistency model family, learns mean velocity between trajectory points
# Data format: VideoLatentLoaderConfig (pre-computed latent.pth + txt_emb.pth)
# NOTE: Different data format from DMD2/f-distill/LADD — needs pre-computed VAE latents
# Reference: Geng et al., 2025 (https://arxiv.org/abs/2505.13447)

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
LATENT_DATA_DIR="${1:-/data/chenqingzhan/training_data/latent_shards}"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

$PYTHON train.py \
    --config=fastgen/configs/experiments/WanT2V/config_mf.py \
    - trainer.ddp=False \
      trainer.fsdp=False \
      trainer.batch_size_global=8 \
      trainer.max_iter=6000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=500 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"WDS:$LATENT_DATA_DIR\"]" \
      log_config.name=meanflow_wan1.3b_single_gpu \
      2>&1 | tee /data/chenqingzhan/fastgen_output/meanflow_train.log
