#!/bin/bash
# run_fdistill.sh — Run f-distill (f-divergence DMD2) on Wan2.1-1.3B with 4-GPU FSDP
#
# f-distill: Improved DMD2 with f-divergence weighting (JS divergence)
# Reference: Xu et al., 2025 (https://arxiv.org/abs/2502.15681)
#
# Usage: bash run_fdistill.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2,3,4,5
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=4 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_fdistill.py \
    - trainer.ddp=False \
      trainer.fsdp=True \
      trainer.batch_size_global=8 \
      trainer.max_iter=6000 \
      trainer.logging_iter=50 \
      trainer.save_ckpt_iter=500 \
      trainer.validation_iter=500 \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"$DATA_SHARDS\"]" \
      log_config.wandb_mode=disabled \
      log_config.name=fdistill_wan1.3b_4gpu
