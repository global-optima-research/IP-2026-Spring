#!/bin/bash
# launch_ect_now.sh — Launch ECT training on 4 free GPUs
# ECT only needs Student network (no teacher/discriminator) → no OOM risk

set -euo pipefail

source /data/chenqingzhan/miniconda3/bin/activate fastgen

export CUDA_VISIBLE_DEVICES=2,3,5,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

FASTGEN_DIR="/data/chenqingzhan/FastGen"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"
LOG="/data/chenqingzhan/fastgen_output/ect_train.log"

cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

echo "=== Launching ECT training ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG"
echo "Start: $(date)"

nohup torchrun --nproc_per_node=4 --standalone train.py \
    --config=fastgen/configs/experiments/WanT2V/config_cm_ct.py \
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
      log_config.name=ect_wan1.3b_4gpu \
      > "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo "$PID" > /data/chenqingzhan/fastgen_output/ect_train.pid
sleep 3
if ps -p $PID > /dev/null 2>&1; then
    echo "ECT training process is running"
else
    echo "WARNING: Process not found!"
    tail -20 "$LOG"
fi
