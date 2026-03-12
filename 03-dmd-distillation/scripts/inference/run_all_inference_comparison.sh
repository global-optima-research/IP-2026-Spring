#!/bin/bash
# run_all_inference_comparison.sh
# Master script: Download pretrained models + Run inference for 4 models
# Designed to run with nohup in background on GPU server
#
# Models compared:
#   1. Teacher (Wan2.1-1.3B, 50-step) — FastGen baseline
#   2. CausVid (DMD, 3-step) — already deployed
#   3. rCM (Consistency Model, 4-step) — NVlabs/rcm
#   4. TurboDiffusion (rCM + quant, 4-step) — thu-ml
#
# Usage: nohup bash run_all_inference_comparison.sh > comparison.log 2>&1 &

set -uo pipefail

# ============================================================
# Configuration
# ============================================================
OUTPUT_ROOT="/data/chenqingzhan/fastgen_output/comparison_2026_03_10"
LOG="$OUTPUT_ROOT/comparison.log"
PROMPT_FILE="$OUTPUT_ROOT/prompts.txt"
TIMING_FILE="$OUTPUT_ROOT/timing_results.txt"

# Shared weights (already on server from CausVid setup)
WAN_VAE="/data/chenqingzhan/CausVid/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
WAN_T5="/data/chenqingzhan/CausVid/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
WAN_DIFFUSERS="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
CAUSVID_CKPT="/data/chenqingzhan/causvid_checkpoints/bidirectional_checkpoint2"

# Mirror for China
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data/chenqingzhan/.cache/huggingface"

# Use GPU 0 for all inference (single GPU)
export CUDA_VISIBLE_DEVICES=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

time_cmd() {
    local label="$1"
    shift
    local start=$(date +%s%N)
    "$@"
    local status=$?
    local end=$(date +%s%N)
    local elapsed=$(( (end - start) / 1000000 ))  # milliseconds
    local elapsed_s=$(echo "scale=2; $elapsed / 1000" | bc)
    echo "$label: ${elapsed_s}s (exit=$status)" | tee -a "$TIMING_FILE"
    return $status
}

# ============================================================
# Step 0: Setup directories and prompts
# ============================================================
mkdir -p "$OUTPUT_ROOT"/{teacher,causvid,rcm,turbodiffusion}
echo "============================================================" > "$LOG"
log "=== Phase 0 Inference Comparison — Start ==="
log "Output root: $OUTPUT_ROOT"
echo "" > "$TIMING_FILE"
echo "=== Timing Results ===" >> "$TIMING_FILE"
echo "Date: $(date)" >> "$TIMING_FILE"
echo "" >> "$TIMING_FILE"

# Unified prompts for fair comparison (5 diverse scenarios)
cat > "$PROMPT_FILE" << 'PROMPTS'
A golden retriever puppy playing joyfully in a sunny garden with green grass and colorful flowers
A futuristic city skyline at sunset with flying cars and bright neon lights reflecting on glass buildings
Ocean waves crashing dramatically on rocky cliffs during a powerful storm with dark clouds
An astronaut in a white spacesuit riding a brown cow through a green meadow under clear blue sky
A red sports car driving fast through a winding mountain road surrounded by autumn foliage
PROMPTS

log "Prompt file created with 5 prompts"

# ============================================================
# Step 1: Setup rCM (NVlabs/rcm)
# ============================================================
log "=== Step 1: Setup rCM ==="

RCM_DIR="/data/chenqingzhan/rcm"
RCM_CKPT="$RCM_DIR/assets/checkpoints/rCM_Wan2.1_T2V_1.3B_480p.pt"

if [ ! -d "$RCM_DIR" ]; then
    log "Cloning NVlabs/rcm..."
    cd /data/chenqingzhan
    git clone https://github.com/NVlabs/rcm.git 2>&1 | tail -5 | tee -a "$LOG"
    log "rCM cloned"
else
    log "rCM directory already exists, skipping clone"
fi

# Create checkpoints dir and symlink shared weights
mkdir -p "$RCM_DIR/assets/checkpoints"
ln -sf "$WAN_VAE" "$RCM_DIR/assets/checkpoints/Wan2.1_VAE.pth"
ln -sf "$WAN_T5" "$RCM_DIR/assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth"

# Download rCM checkpoint if not exists
if [ ! -f "$RCM_CKPT" ]; then
    log "Downloading rCM checkpoint (2.84 GB)..."
    wget -q --show-progress -O "$RCM_CKPT" \
        "${HF_ENDPOINT}/worstcoder/rcm-Wan/resolve/main/rCM_Wan2.1_T2V_1.3B_480p.pt" 2>&1 | tee -a "$LOG"
    log "rCM checkpoint downloaded: $(du -h "$RCM_CKPT" | cut -f1)"
else
    log "rCM checkpoint already exists: $(du -h "$RCM_CKPT" | cut -f1)"
fi

# Install rCM dependencies (in existing fastgen env)
log "Installing rCM dependencies..."
source /data/chenqingzhan/miniconda3/bin/activate fastgen
pip install -q loguru attrs fvcore imageio[ffmpeg] ftfy 2>&1 | tail -3 | tee -a "$LOG"

# Patch rCM for RTX 5090 (flash_attn rotary embedding fallback)
RCM_WAN_FILE="$RCM_DIR/rcm/networks/wan2pt1.py"
if [ -f "$RCM_WAN_FILE" ] && grep -q "flash_apply_rotary_emb" "$RCM_WAN_FILE"; then
    log "Patching rCM for RTX 5090 (rotary embedding fallback)..."
    cp "$RCM_WAN_FILE" "${RCM_WAN_FILE}.bak"

    # Add a pure-PyTorch rotary embedding fallback
    python3 << 'PATCH_SCRIPT'
import re

filepath = "/data/chenqingzhan/rcm/rcm/networks/wan2pt1.py"
with open(filepath, 'r') as f:
    content = f.read()

# Add fallback rotary embedding function after the flash_attn import block
fallback_code = '''
# === RTX 5090 Patch: Pure-PyTorch rotary embedding fallback ===
def _pytorch_apply_rotary_emb(x, freqs):
    """Apply rotary position embedding using pure PyTorch (no flash_attn needed)."""
    cos = freqs.cos().to(x.dtype)
    sin = freqs.sin().to(x.dtype)
    # x shape: (batch, seqlen, nheads, headdim) or (batch, nheads, seqlen, headdim)
    if x.dim() == 4:
        # Handle both layouts
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        # Expand cos/sin to match x dimensions
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if cos.shape[-1] != d:
            cos = cos[..., :d]
            sin = sin[..., :d]
        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.to(x.dtype)
    return x

if flash_apply_rotary_emb is None:
    flash_apply_rotary_emb = _pytorch_apply_rotary_emb
# === End RTX 5090 Patch ===
'''

# Find the flash_attn import try/except block and add fallback after it
# Pattern: after the try/except that sets flash_apply_rotary_emb
import_pattern = r'(except.*?:\s*\n\s*flash_apply_rotary_emb\s*=\s*None)'
match = re.search(import_pattern, content, re.DOTALL)
if match:
    insert_pos = match.end()
    content = content[:insert_pos] + '\n' + fallback_code + content[insert_pos:]
    # Need to add torch import if not present
    if 'import torch' not in content.split('flash_apply_rotary_emb')[0]:
        content = 'import torch\n' + content
    with open(filepath, 'w') as f:
        f.write(content)
    print("SUCCESS: Patched wan2pt1.py with rotary embedding fallback")
else:
    # Try alternative: just replace flash_apply_rotary_emb usage with inline code
    print("WARNING: Could not find expected import pattern, trying alternative patch")
    # Check if flash_apply_rotary_emb is already None and used directly
    if 'flash_apply_rotary_emb' in content:
        content = content.replace(
            'flash_apply_rotary_emb = None',
            'flash_apply_rotary_emb = None\n' + fallback_code.replace('if flash_apply_rotary_emb is None:\n    flash_apply_rotary_emb = _pytorch_apply_rotary_emb\n# === End RTX 5090 Patch ===', '').strip()
        )
        with open(filepath, 'w') as f:
            f.write(content)
        print("SUCCESS: Applied alternative patch")
    else:
        print("SKIP: No flash_apply_rotary_emb found in file")
PATCH_SCRIPT
    log "rCM patch applied"
else
    log "rCM already patched or file not found"
fi

log "rCM setup complete"

# ============================================================
# Step 2: Setup TurboDiffusion (thu-ml)
# ============================================================
log "=== Step 2: Setup TurboDiffusion ==="

TURBO_DIR="/data/chenqingzhan/TurboDiffusion"
TURBO_CKPT_FULL="$TURBO_DIR/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"
TURBO_CKPT_QUANT="$TURBO_DIR/checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth"

if [ ! -d "$TURBO_DIR" ]; then
    log "Cloning thu-ml/TurboDiffusion..."
    cd /data/chenqingzhan
    git clone https://github.com/thu-ml/TurboDiffusion.git 2>&1 | tail -5 | tee -a "$LOG"
    cd "$TURBO_DIR"
    git submodule update --init --recursive 2>&1 | tail -5 | tee -a "$LOG"
    log "TurboDiffusion cloned"
else
    log "TurboDiffusion directory already exists, skipping clone"
fi

# Create checkpoints dir and symlink shared weights
mkdir -p "$TURBO_DIR/checkpoints"
ln -sf "$WAN_VAE" "$TURBO_DIR/checkpoints/Wan2.1_VAE.pth"
ln -sf "$WAN_T5" "$TURBO_DIR/checkpoints/models_t5_umt5-xxl-enc-bf16.pth"

# Download TurboDiffusion checkpoints
if [ ! -f "$TURBO_CKPT_FULL" ]; then
    log "Downloading TurboDiffusion full checkpoint (2.84 GB)..."
    wget -q --show-progress -O "$TURBO_CKPT_FULL" \
        "${HF_ENDPOINT}/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P.pth" 2>&1 | tee -a "$LOG"
    log "TurboDiffusion full checkpoint: $(du -h "$TURBO_CKPT_FULL" | cut -f1)"
else
    log "TurboDiffusion full checkpoint already exists"
fi

if [ ! -f "$TURBO_CKPT_QUANT" ]; then
    log "Downloading TurboDiffusion quantized checkpoint (1.45 GB)..."
    wget -q --show-progress -O "$TURBO_CKPT_QUANT" \
        "${HF_ENDPOINT}/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth" 2>&1 | tee -a "$LOG"
    log "TurboDiffusion quant checkpoint: $(du -h "$TURBO_CKPT_QUANT" | cut -f1)"
else
    log "TurboDiffusion quant checkpoint already exists"
fi

# Install TurboDiffusion (try pip install, fallback to manual)
log "Installing TurboDiffusion dependencies..."
cd "$TURBO_DIR"
pip install -q -e . --no-build-isolation 2>&1 | tail -5 | tee -a "$LOG" || {
    log "WARNING: pip install failed, trying manual dependency install..."
    pip install -q loguru imageio[ffmpeg] ftfy einops 2>&1 | tail -3 | tee -a "$LOG"
}
log "TurboDiffusion setup complete"

# ============================================================
# Step 3: Run Teacher Inference (50-step baseline)
# ============================================================
log "=== Step 3: Teacher Inference (50-step) ==="

TEACHER_OUT="$OUTPUT_ROOT/teacher"
FASTGEN_DIR="/data/chenqingzhan/FastGen"

cd "$FASTGEN_DIR"
export PYTHONPATH=$(pwd)

# Run teacher with each prompt
PROMPT_IDX=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    PROMPT_IDX=$((PROMPT_IDX + 1))
    OUT_FILE="$TEACHER_OUT/teacher_50step_$(printf '%03d' $PROMPT_IDX).mp4"

    if [ -f "$OUT_FILE" ]; then
        log "Teacher prompt $PROMPT_IDX already exists, skipping"
        continue
    fi

    log "Teacher prompt $PROMPT_IDX: ${prompt:0:60}..."

    START_T=$(date +%s%N)
    torchrun --nproc_per_node=1 --standalone \
        scripts/inference/video_model_inference.py \
        --do_student_sampling False \
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
        - trainer.seed=42 trainer.ddp=False model.guidance_scale=5.0 \
          model.net.model_id_or_local_path="$WAN_DIFFUSERS" \
          log_config.name="teacher_comparison_p${PROMPT_IDX}" \
          log_config.wandb_mode=disabled \
          model.prompts="[\"$prompt\"]" \
        2>&1 | tail -5 | tee -a "$LOG"
    END_T=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)
    echo "Teacher prompt $PROMPT_IDX: ${ELAPSED}s" >> "$TIMING_FILE"

    # Find and move the generated video
    LATEST=$(find "$FASTGEN_DIR/../fastgen_output/fastgen/" -name "teacher_cfg5.0_steps50_0000_seed42.mp4" -newer "$LOG" 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST" "$OUT_FILE"
        log "Teacher saved: $OUT_FILE (${ELAPSED}s)"
    else
        log "WARNING: Teacher output not found for prompt $PROMPT_IDX"
    fi
done < "$PROMPT_FILE"

log "Teacher inference complete"

# ============================================================
# Step 4: Run CausVid Inference (3-step DMD)
# ============================================================
log "=== Step 4: CausVid Inference (3-step DMD) ==="

CAUSVID_OUT="$OUTPUT_ROOT/causvid"
CAUSVID_DIR="/data/chenqingzhan/CausVid"

cd "$CAUSVID_DIR"

# CausVid takes a prompt file
cp "$PROMPT_FILE" "$CAUSVID_DIR/comparison_prompts.txt"

START_T=$(date +%s%N)
python minimal_inference/bidirectional_inference.py \
    --config_path configs/wan_bidirectional_dmd_from_scratch.yaml \
    --checkpoint_folder "$CAUSVID_CKPT" \
    --output_folder "$CAUSVID_OUT" \
    --prompt_file_path "$CAUSVID_DIR/comparison_prompts.txt" \
    2>&1 | tail -10 | tee -a "$LOG"
END_T=$(date +%s%N)
ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)
echo "CausVid total (5 videos): ${ELAPSED}s" >> "$TIMING_FILE"
NUM_VIDEOS=$(ls "$CAUSVID_OUT"/output_*.mp4 2>/dev/null | wc -l)
if [ "$NUM_VIDEOS" -gt 0 ]; then
    PER_VIDEO=$(echo "scale=2; $ELAPSED / $NUM_VIDEOS" | bc)
    echo "CausVid per-video: ${PER_VIDEO}s" >> "$TIMING_FILE"
fi
log "CausVid inference complete: $NUM_VIDEOS videos (${ELAPSED}s total)"

# ============================================================
# Step 5: Run rCM Inference (4-step Consistency Model)
# ============================================================
log "=== Step 5: rCM Inference (4-step) ==="

RCM_OUT="$OUTPUT_ROOT/rcm"
cd "$RCM_DIR"
export PYTHONPATH="$RCM_DIR:$PYTHONPATH"

PROMPT_IDX=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    PROMPT_IDX=$((PROMPT_IDX + 1))
    OUT_FILE="$RCM_OUT/rcm_4step_$(printf '%03d' $PROMPT_IDX).mp4"

    if [ -f "$OUT_FILE" ]; then
        log "rCM prompt $PROMPT_IDX already exists, skipping"
        continue
    fi

    log "rCM prompt $PROMPT_IDX: ${prompt:0:60}..."

    START_T=$(date +%s%N)
    python rcm/inference/wan2pt1_t2v_rcm_infer.py \
        --dit_path "$RCM_CKPT" \
        --vae_path "$RCM_DIR/assets/checkpoints/Wan2.1_VAE.pth" \
        --text_encoder_path "$RCM_DIR/assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth" \
        --model_size 1.3B \
        --num_samples 1 \
        --num_steps 4 \
        --sigma_max 80 \
        --resolution 480p \
        --aspect_ratio 16:9 \
        --num_frames 81 \
        --seed 42 \
        --save_path "$OUT_FILE" \
        --prompt "$prompt" \
        2>&1 | tail -10 | tee -a "$LOG"
    STATUS=$?
    END_T=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)
    echo "rCM prompt $PROMPT_IDX: ${ELAPSED}s (exit=$STATUS)" >> "$TIMING_FILE"

    if [ $STATUS -ne 0 ]; then
        log "ERROR: rCM failed on prompt $PROMPT_IDX (exit=$STATUS)"
        # Try with 2 steps as fallback
        log "Retrying with 2 steps..."
        python rcm/inference/wan2pt1_t2v_rcm_infer.py \
            --dit_path "$RCM_CKPT" \
            --vae_path "$RCM_DIR/assets/checkpoints/Wan2.1_VAE.pth" \
            --text_encoder_path "$RCM_DIR/assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth" \
            --model_size 1.3B \
            --num_samples 1 \
            --num_steps 2 \
            --sigma_max 80 \
            --resolution 480p \
            --num_frames 81 \
            --seed 42 \
            --save_path "$OUT_FILE" \
            --prompt "$prompt" \
            2>&1 | tail -10 | tee -a "$LOG"
    fi

    log "rCM prompt $PROMPT_IDX: ${ELAPSED}s"
done < "$PROMPT_FILE"

log "rCM inference complete"

# ============================================================
# Step 6: Run TurboDiffusion Inference (4-step rCM + optimizations)
# ============================================================
log "=== Step 6: TurboDiffusion Inference (4-step) ==="

TURBO_OUT="$OUTPUT_ROOT/turbodiffusion"
cd "$TURBO_DIR"
export PYTHONPATH="$TURBO_DIR:$PYTHONPATH"

# Try full checkpoint with original attention first (most compatible)
PROMPT_IDX=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    PROMPT_IDX=$((PROMPT_IDX + 1))
    OUT_FILE="$TURBO_OUT/turbo_4step_$(printf '%03d' $PROMPT_IDX).mp4"

    if [ -f "$OUT_FILE" ]; then
        log "TurboDiffusion prompt $PROMPT_IDX already exists, skipping"
        continue
    fi

    log "TurboDiffusion prompt $PROMPT_IDX: ${prompt:0:60}..."

    START_T=$(date +%s%N)
    python turbodiffusion/inference/wan2.1_t2v_infer.py \
        --model Wan2.1-1.3B \
        --dit_path "$TURBO_CKPT_FULL" \
        --vae_path "$TURBO_DIR/checkpoints/Wan2.1_VAE.pth" \
        --text_encoder_path "$TURBO_DIR/checkpoints/models_t5_umt5-xxl-enc-bf16.pth" \
        --resolution 480p \
        --num_samples 1 \
        --num_steps 4 \
        --num_frames 81 \
        --seed 42 \
        --attention_type original \
        --save_path "$OUT_FILE" \
        --prompt "$prompt" \
        2>&1 | tail -10 | tee -a "$LOG"
    STATUS=$?
    END_T=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)
    echo "TurboDiffusion prompt $PROMPT_IDX: ${ELAPSED}s (exit=$STATUS)" >> "$TIMING_FILE"

    if [ $STATUS -ne 0 ]; then
        log "ERROR: TurboDiffusion failed on prompt $PROMPT_IDX (exit=$STATUS)"
    else
        log "TurboDiffusion prompt $PROMPT_IDX: ${ELAPSED}s"
    fi
done < "$PROMPT_FILE"

log "TurboDiffusion inference complete"

# ============================================================
# Step 7: Summary
# ============================================================
log "=== Summary ==="
log ""
log "Output directory: $OUTPUT_ROOT"
log ""
log "Files generated:"
ls -la "$TEACHER_OUT"/*.mp4 2>/dev/null | tee -a "$LOG" || log "  Teacher: no files"
ls -la "$CAUSVID_OUT"/*.mp4 2>/dev/null | tee -a "$LOG" || log "  CausVid: no files"
ls -la "$RCM_OUT"/*.mp4 2>/dev/null | tee -a "$LOG" || log "  rCM: no files"
ls -la "$TURBO_OUT"/*.mp4 2>/dev/null | tee -a "$LOG" || log "  TurboDiffusion: no files"
log ""
log "Timing results:"
cat "$TIMING_FILE" | tee -a "$LOG"
log ""
log "=== Phase 0 Inference Comparison — Complete ==="
log "Total elapsed: $(date)"
