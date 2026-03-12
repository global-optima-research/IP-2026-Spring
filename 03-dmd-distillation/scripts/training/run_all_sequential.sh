#!/bin/bash
# run_all_sequential.sh — Run all 5 distillation methods sequentially
# Order: DMD2 → ECT → LADD → f-distill → CD
#
# Usage: nohup bash run_all_sequential.sh > /data/chenqingzhan/fastgen_output/all_train.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="/data/chenqingzhan/scripts"
FASTGEN_DIR="/data/chenqingzhan/FastGen"
LOG_DIR="/data/chenqingzhan/fastgen_output"
MODEL_PATH="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
DATA_SHARDS="WDS:/data/datasets/OpenVid-1M/webdataset"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3,4,5
export FASTGEN_OUTPUT_ROOT="$LOG_DIR"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

source /data/chenqingzhan/miniconda3/bin/activate fastgen
cd $FASTGEN_DIR
export PYTHONPATH=$(pwd)

echo "=========================================="
echo "=== Phase 0: Sequential 5-Method Training"
echo "=== Start: $(date)"
echo "=== GPUs: 2,3,4,5 (4x RTX 5090, FSDP)"
echo "=== PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "=========================================="

run_method() {
    local name="$1"
    local config="$2"
    local log_name="$3"
    local extra_args="${4:-}"
    local log="$LOG_DIR/${name}_train.log"

    echo ""
    echo "=========================================="
    echo "=== [$name] Starting at $(date)"
    echo "=== Config: $config"
    echo "=== Log: $log"
    echo "=========================================="

    torchrun --nproc_per_node=4 --standalone train.py \
        --config=$config \
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
          log_config.name=$log_name \
          $extra_args \
          > "$log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "=== [$name] COMPLETED at $(date) (exit code: 0)"
    else
        echo "=== [$name] FAILED at $(date) (exit code: $exit_code)"
        echo "=== Last error:"
        grep -E 'Error|OOM' "$log" | tail -3
    fi
    return $exit_code
}

# ============================================================
# Method 1: DMD2 (with CPU offload for discriminator memory)
# ============================================================
run_method "DMD2" \
    "fastgen/configs/experiments/WanT2V/config_dmd2.py" \
    "dmd2_wan1.3b_4gpu" \
    "trainer.fsdp_cpu_offload=True" \
    || true

# ============================================================
# Method 2: ECT (lightest — no teacher, no discriminator)
# ============================================================
run_method "ECT" \
    "fastgen/configs/experiments/WanT2V/config_cm_ct.py" \
    "ect_wan1.3b_4gpu" \
    || true

# ============================================================
# Method 3: LADD (with CPU offload for discriminator)
# ============================================================
run_method "LADD" \
    "fastgen/configs/experiments/WanT2V/config_ladd.py" \
    "ladd_wan1.3b_4gpu" \
    "trainer.fsdp_cpu_offload=True" \
    || true

# ============================================================
# Method 4: f-distill (with CPU offload)
# ============================================================
run_method "f-distill" \
    "fastgen/configs/experiments/WanT2V/config_fdistill.py" \
    "fdistill_wan1.3b_4gpu" \
    "trainer.fsdp_cpu_offload=True" \
    || true

# ============================================================
# Method 5: CD (with teacher, consistency distillation)
# ============================================================
run_method "CD" \
    "fastgen/configs/experiments/WanT2V/config_cm_cd.py" \
    "cd_wan1.3b_4gpu" \
    || true

echo ""
echo "=========================================="
echo "=== All methods finished at $(date)"
echo "=========================================="
echo "Logs in: $LOG_DIR/*_train.log"
