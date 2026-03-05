#!/bin/bash
# run_inference.sh — Run FastGen inference (Teacher or Student) on single GPU
# Usage:
#   bash run_inference.sh teacher     # Run 50-step teacher inference
#   bash run_inference.sh student     # Run 4-step student inference
#   bash run_inference.sh both        # Compare teacher vs student

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

CONDA_DIR="/data/chenqingzhan/miniconda3"
FASTGEN_DIR="/data/chenqingzhan/FastGen"
PYTHON="$CONDA_DIR/envs/fastgen/bin/python"

MODE="${1:-teacher}"
CKPT_PATH="${2:-}"

cd $FASTGEN_DIR

case "$MODE" in
    teacher)
        echo "Running Teacher (50-step) inference..."
        PYTHONPATH=$(pwd) $PYTHON scripts/inference/video_model_inference.py \
            --do_student_sampling False \
            --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
            - trainer.seed=42 trainer.ddp=False model.guidance_scale=5.0
        ;;
    student)
        if [ -z "$CKPT_PATH" ]; then
            echo "Error: Student mode requires checkpoint path as second argument"
            echo "Usage: bash run_inference.sh student /path/to/checkpoint.pth"
            exit 1
        fi
        echo "Running Student (4-step) inference..."
        PYTHONPATH=$(pwd) $PYTHON scripts/inference/video_model_inference.py \
            --ckpt_path "$CKPT_PATH" \
            --do_student_sampling True \
            --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
            - trainer.seed=42 trainer.ddp=False model.guidance_scale=5.0
        ;;
    both)
        if [ -z "$CKPT_PATH" ]; then
            echo "Error: Both mode requires checkpoint path as second argument"
            exit 1
        fi
        echo "Running Teacher + Student comparison..."
        PYTHONPATH=$(pwd) $PYTHON scripts/inference/video_model_inference.py \
            --ckpt_path "$CKPT_PATH" \
            --do_student_sampling True --do_teacher_sampling True \
            --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
            - trainer.seed=42 trainer.ddp=False model.guidance_scale=5.0
        ;;
    *)
        echo "Usage: bash run_inference.sh {teacher|student|both} [checkpoint_path]"
        exit 1
        ;;
esac
