#!/bin/bash
# run_teacher_inference.sh — Run Teacher (50-step) inference with custom prompts
# Usage: nohup bash run_teacher_inference.sh > teacher_inference.log 2>&1 &

set -uo pipefail

export CUDA_VISIBLE_DEVICES=5
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

OUTPUT_ROOT="/data/chenqingzhan/fastgen_output/comparison_2026_03_12"
TIMING_FILE="$OUTPUT_ROOT/timing_results.txt"
CONDA_DIR="/data/chenqingzhan/miniconda3"
PYTHON="$CONDA_DIR/envs/fastgen/bin/python"
FASTGEN_DIR="/data/chenqingzhan/FastGen"
WAN_DIFFUSERS="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate fastgen

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Create prompt file for FastGen (one prompt per line)
PROMPT_FILE="$OUTPUT_ROOT/teacher_prompts.txt"
cat > "$PROMPT_FILE" << 'EOF'
A golden retriever puppy playing joyfully in a sunny garden with colorful flowers blooming around it
A futuristic city skyline at sunset with flying cars and brilliant neon lights reflecting off glass towers
Ocean waves crashing dramatically on rocky cliffs during a powerful storm with dark clouds overhead
An astronaut in a white spacesuit riding a brown cow through a lush green meadow under blue skies
A red sports car driving fast through a winding mountain road with autumn foliage on both sides
EOF

cd "$FASTGEN_DIR"
export PYTHONPATH="$FASTGEN_DIR"

log "=== Teacher (50-step) Inference Start ==="
log "GPU: $CUDA_VISIBLE_DEVICES"

START_ALL=$(date +%s)

$PYTHON scripts/inference/video_model_inference.py \
    --do_student_sampling False \
    --prompt_file "$PROMPT_FILE" \
    --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.seed=42 trainer.ddp=False model.guidance_scale=5.0 \
      model.net.model_id_or_local_path="$WAN_DIFFUSERS" \
      log_config.name="teacher_comparison" \
      log_config.wandb_mode=disabled \
    2>&1 | tee /tmp/teacher_output.log

END_ALL=$(date +%s)
TOTAL_ELAPSED=$((END_ALL - START_ALL))

log "Teacher inference done in ${TOTAL_ELAPSED}s total"

# Find generated videos and copy to output
TEACHER_OUT="$OUTPUT_ROOT/teacher"
mkdir -p "$TEACHER_OUT"

log "Looking for generated videos..."
find /data/chenqingzhan/fastgen_output/fastgen/ -name "teacher_cfg5.0_steps50_*_seed42.mp4" -newer "$PROMPT_FILE" 2>/dev/null | sort | while read -r f; do
    BASENAME=$(basename "$f")
    cp "$f" "$TEACHER_OUT/$BASENAME"
    log "Copied: $BASENAME"
done

NUM_VIDEOS=$(ls "$TEACHER_OUT"/*.mp4 2>/dev/null | wc -l)
if [ "$NUM_VIDEOS" -gt 0 ]; then
    PER_VIDEO=$((TOTAL_ELAPSED / NUM_VIDEOS))
    log "SUCCESS: $NUM_VIDEOS videos, total=${TOTAL_ELAPSED}s, per-video=${PER_VIDEO}s"
    # Append to timing file
    echo "" >> "$TIMING_FILE"
    echo "=== Teacher (fixed) ===" >> "$TIMING_FILE"
    echo "Teacher total ($NUM_VIDEOS videos): ${TOTAL_ELAPSED}s (per-video: ~${PER_VIDEO}s) [OK]" >> "$TIMING_FILE"
else
    log "FAILED: No videos found"
    # Try alternative search path
    log "Checking alternative paths..."
    find /data/chenqingzhan/fastgen_output/ -name "*.mp4" -newer "$PROMPT_FILE" 2>/dev/null | head -10
    echo "Teacher: ${TOTAL_ELAPSED}s [FAILED]" >> "$TIMING_FILE"
fi

log "=== Teacher Inference Complete ==="
