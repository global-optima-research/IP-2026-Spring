#!/bin/bash
# run_dmd2.sh — Run DMD2 distillation on Wan2.1-1.3B with 4-GPU FSDP
#
# DMD2: VSD loss + GAN adversarial training, alternates student/discriminator updates
# Reference: Yin et al., 2024 (https://arxiv.org/abs/2405.14867)
#
# Usage: bash run_dmd2.sh

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

# 4-GPU FSDP: model sharded across GPUs, grad_accum = 8/(1*4) = 2
torchrun --nproc_per_node=4 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
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
      log_config.name=dmd2_wan1.3b_4gpu
