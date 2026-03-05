#!/bin/bash
# run_cd_single_gpu.sh — Run CD (Consistency Distillation) on Wan2.1-1.3B with single GPU
# Usage: bash run_cd_single_gpu.sh [data_dir]
#
# CD: Consistency Distillation with teacher (use_cd=True)
# Same data format as DMD2 (VideoLoaderConfig: mp4+txt WebDataset)
# Config: custom config_cm_cd.py (adapted from EDM2 CM + WanT2V DMD2)
# Reference: Song et al., 2023 (https://arxiv.org/abs/2303.01469)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
PYTHON="/data/chenqingzhan/miniconda3/envs/fastgen/bin/python"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
CONFIG_PATH="/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_cm_cd.py"
DATA_DIR="${1:-/data/chenqingzhan/training_data/video_shards}"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

$PYTHON train.py \
    --config=$CONFIG_PATH \
    - trainer.ddp=False \
      trainer.fsdp=False \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"WDS:$DATA_DIR\"]" \
      log_config.name=cd_wan1.3b_single_gpu \
      2>&1 | tee /data/chenqingzhan/fastgen_output/cd_train.log
