#!/bin/bash
# run_ladd.sh — Run LADD (Latent Adversarial Diffusion Distillation) on Wan2.1-1.3B with 4-GPU FSDP
#
# LADD: Pure adversarial distillation (GAN-only, no score distillation loss)
# Reference: Sauer et al., 2024 (https://arxiv.org/abs/2403.12015)
#
# Usage: bash run_ladd.sh

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
    --config=fastgen/configs/experiments/WanT2V/config_ladd.py \
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
      log_config.name=ladd_wan1.3b_4gpu
