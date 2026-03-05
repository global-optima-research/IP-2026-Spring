#!/bin/bash
# run_ect_single_gpu.sh — Run ECT (Enhanced Consistency Training) on Wan2.1-1.3B with single GPU
# Usage: bash run_ect_single_gpu.sh [data_dir]
#
# ECT: Consistency Training without teacher (use_cd=False)
# Same data format as DMD2 (VideoLoaderConfig: mp4+txt WebDataset)
# Config: custom config_cm_ct.py (adapted from EDM2 CM + WanT2V MeanFlow)
# Reference: Geng et al., 2024 (https://arxiv.org/abs/2406.14548)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
PYTHON="/data/chenqingzhan/miniconda3/envs/fastgen/bin/python"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
CONFIG_PATH="/data/chenqingzhan/FastGen/fastgen/configs/experiments/WanT2V/config_cm_ct.py"
DATA_DIR="${1:-/data/chenqingzhan/training_data/video_shards}"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

$PYTHON train.py \
    --config=$CONFIG_PATH \
    - trainer.ddp=False \
      trainer.fsdp=False \
      model.net.model_id_or_local_path=$MODEL_PATH \
      dataloader_train.datatags="[\"WDS:$DATA_DIR\"]" \
      log_config.name=ect_wan1.3b_single_gpu \
      2>&1 | tee /data/chenqingzhan/fastgen_output/ect_train.log
