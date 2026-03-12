#!/bin/bash
# run_cd_train.sh — Run CD (Consistency Distillation) on Wan2.1-1.3B with 2-GPU FSDP
# CD: Consistency Distillation with teacher (use_cd=True)
# Uses GPU 5,7 (2 free GPUs available)
#
# Usage: nohup bash run_cd_train.sh > /data/chenqingzhan/fastgen_output/cd_train.log 2>&1 &

set -euo pipefail

source /data/chenqingzhan/miniconda3/bin/activate fastgen

export CUDA_VISIBLE_DEVICES=5,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

echo "=== CD Training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

torchrun --nproc_per_node=2 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_cm_cd.py \
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
      log_config.name=cd_wan1.3b_2gpu

echo "=== CD Training complete ==="
echo "End: $(date)"
