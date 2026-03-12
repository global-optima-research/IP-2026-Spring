#!/bin/bash
# fix_and_run_inference.sh — Fixed inference comparison for Phase 0 Report
# Fixes applied:
#   1. Teacher: use direct Python instead of torchrun
#   2. CausVid: pip install -e . to fix 'causvid' module import
#   3. rCM: rewrite rope_apply to bypass flash_attn rotary emb
#   4. TurboDiffusion: make 'ops' import conditional in modify_model.py
#
# Usage: nohup bash fix_and_run_inference.sh > fix_comparison.log 2>&1 &

set -uo pipefail

# ============================================================
# Configuration
# ============================================================
export CUDA_VISIBLE_DEVICES=5
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

OUTPUT_ROOT="/data/chenqingzhan/fastgen_output/comparison_2026_03_12"
TIMING_FILE="$OUTPUT_ROOT/timing_results.txt"
PROMPT_FILE="$OUTPUT_ROOT/prompts.txt"
CONDA_DIR="/data/chenqingzhan/miniconda3"
PYTHON="$CONDA_DIR/envs/fastgen/bin/python"
PIP="$CONDA_DIR/envs/fastgen/bin/pip"

# Model paths
WAN_DIFFUSERS="/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
CAUSVID_DIR="/data/chenqingzhan/CausVid"
CAUSVID_CKPT="/data/chenqingzhan/causvid_checkpoints/bidirectional_checkpoint2"
RCM_DIR="/data/chenqingzhan/rcm"
RCM_CKPT="$RCM_DIR/assets/checkpoints/rCM_Wan2.1_T2V_1.3B_480p.pt"
TURBO_DIR="/data/chenqingzhan/TurboDiffusion"
TURBO_CKPT="$TURBO_DIR/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"
FASTGEN_DIR="/data/chenqingzhan/FastGen"

# Activate conda
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate fastgen

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Setup output dirs
mkdir -p "$OUTPUT_ROOT"/{teacher,causvid,rcm,turbodiffusion}

# Create prompt file
cat > "$PROMPT_FILE" << 'PROMPTS'
A golden retriever puppy playing joyfully in a sunny garden with colorful flowers blooming around it
A futuristic city skyline at sunset with flying cars and brilliant neon lights reflecting off glass towers
Ocean waves crashing dramatically on rocky cliffs during a powerful storm with dark clouds overhead
An astronaut in a white spacesuit riding a brown cow through a lush green meadow under blue skies
A red sports car driving fast through a winding mountain road with autumn foliage on both sides
PROMPTS

# Init timing file
echo "=== Timing Results ===" > "$TIMING_FILE"
echo "Date: $(date)" >> "$TIMING_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" >> "$TIMING_FILE"
echo "" >> "$TIMING_FILE"

log "============================================================"
log "=== Phase 0 Inference Comparison (Fixed) — Start ==="
log "============================================================"

# ============================================================
# Fix 1: Teacher — use direct Python, not torchrun
# ============================================================
log "=== Model 1: Teacher (50-step baseline) ==="

cd "$FASTGEN_DIR"
export PYTHONPATH="$FASTGEN_DIR"

PROMPT_IDX=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    PROMPT_IDX=$((PROMPT_IDX + 1))
    OUT_FILE="$OUTPUT_ROOT/teacher/teacher_50step_$(printf '%03d' $PROMPT_IDX).mp4"

    [ -f "$OUT_FILE" ] && { log "Teacher prompt $PROMPT_IDX: SKIP (exists)"; continue; }

    log "Teacher prompt $PROMPT_IDX: ${prompt:0:60}..."
    START_T=$(date +%s%N)

    $PYTHON scripts/inference/video_model_inference.py \
        --do_student_sampling False \
        --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
        - trainer.seed=42 trainer.ddp=False model.guidance_scale=5.0 \
          model.net.model_id_or_local_path="$WAN_DIFFUSERS" \
          log_config.name="teacher_fix_p${PROMPT_IDX}" \
          log_config.wandb_mode=disabled \
          "model.prompts=[\"$prompt\"]" \
        2>&1 | tail -10

    END_T=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)

    # Find the generated video
    LATEST=$(find "$FASTGEN_DIR/../fastgen_output/" -name "teacher_cfg5.0_steps50_0000_seed42.mp4" -newer "$TIMING_FILE" 2>/dev/null | sort -t/ -k9 | tail -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST" "$OUT_FILE"
        log "Teacher prompt $PROMPT_IDX: OK (${ELAPSED}s) -> $OUT_FILE"
        echo "Teacher prompt $PROMPT_IDX: ${ELAPSED}s [OK]" >> "$TIMING_FILE"
    else
        log "Teacher prompt $PROMPT_IDX: FAILED (${ELAPSED}s) — output not found"
        echo "Teacher prompt $PROMPT_IDX: ${ELAPSED}s [FAILED]" >> "$TIMING_FILE"
    fi
done < "$PROMPT_FILE"

log "Teacher inference complete"

# ============================================================
# Fix 2: CausVid — install causvid package
# ============================================================
log "=== Model 2: CausVid (3-step DMD) ==="

cd "$CAUSVID_DIR"
log "Installing causvid package..."
$PIP install -q -e . 2>&1 | tail -3
export PYTHONPATH="$CAUSVID_DIR:$FASTGEN_DIR"

cp "$PROMPT_FILE" "$CAUSVID_DIR/comparison_prompts.txt"

START_T=$(date +%s%N)
$PYTHON minimal_inference/bidirectional_inference.py \
    --config_path configs/wan_bidirectional_dmd_from_scratch.yaml \
    --checkpoint_folder "$CAUSVID_CKPT" \
    --output_folder "$OUTPUT_ROOT/causvid" \
    --prompt_file_path "$CAUSVID_DIR/comparison_prompts.txt" \
    2>&1 | tail -20
STATUS=$?
END_T=$(date +%s%N)
ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)

NUM_VIDEOS=$(ls "$OUTPUT_ROOT/causvid/"*.mp4 2>/dev/null | wc -l)
if [ "$NUM_VIDEOS" -gt 0 ]; then
    PER_VIDEO=$(echo "scale=2; $ELAPSED / $NUM_VIDEOS" | bc)
    echo "CausVid total ($NUM_VIDEOS videos): ${ELAPSED}s (per-video: ${PER_VIDEO}s) [OK]" >> "$TIMING_FILE"
    log "CausVid: OK ($NUM_VIDEOS videos, ${ELAPSED}s total)"
else
    echo "CausVid: ${ELAPSED}s [FAILED, exit=$STATUS]" >> "$TIMING_FILE"
    log "CausVid: FAILED (exit=$STATUS, ${ELAPSED}s)"
fi

# ============================================================
# Fix 3: rCM — patch rope_apply to not use flash_attn rotary
# ============================================================
log "=== Model 3: rCM (4-step Consistency Model) ==="

RCM_WAN_FILE="$RCM_DIR/rcm/networks/wan2pt1.py"
log "Patching rCM rope_apply..."

# Back up original
cp "$RCM_WAN_FILE" "${RCM_WAN_FILE}.bak_fix2"

# Apply proper patch: replace rope_apply function
$PYTHON << 'PATCH_SCRIPT'
import re

filepath = "/data/chenqingzhan/rcm/rcm/networks/wan2pt1.py"
with open(filepath, 'r') as f:
    content = f.read()

# Replace the rope_apply function entirely
old_rope_apply = '''def rope_apply(x, freqs):
    """
    Optimized version of rope_apply using flash_attention's rotary embedding implementation.
    This version processes the entire batch at once for efficiency.

    Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads, head_dim]
        freqs (Tensor): Complex frequencies with shape [max_seq_len, head_dim // 2]

    Returns:
        Tensor: Rotary-embedded tensor with same shape as input
    """
    batch_size, seq_len, n_heads, head_dim = x.shape

    # freqs is already sharded to local seq_len under flattened CP
    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    # Apply the rotation
    rotated = flash_apply_rotary_emb(x.to(torch.float32), cos, sin, interleaved=True, inplace=False)

    return rotated.to(x.dtype)'''

new_rope_apply = '''def rope_apply(x, freqs):
    """
    Rotary position embedding application.
    Uses flash_attn when available, falls back to pure PyTorch implementation.

    Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads, head_dim]
        freqs (Tensor): Frequencies with shape [max_seq_len, head_dim // 2]

    Returns:
        Tensor: Rotary-embedded tensor with same shape as input
    """
    batch_size, seq_len, n_heads, head_dim = x.shape

    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    try:
        from flash_attn.layers.rotary import apply_rotary_emb as _flash_rope
        rotated = _flash_rope(x.to(torch.float32), cos, sin, interleaved=True, inplace=False)
    except (ImportError, TypeError):
        # Pure PyTorch fallback for interleaved rotary embedding
        x_fp32 = x.to(torch.float32)
        # Interleaved layout: pairs are (x[...,0], x[...,1]), (x[...,2], x[...,3]), ...
        x_even = x_fp32[..., 0::2]  # [B, S, H, D/2]
        x_odd = x_fp32[..., 1::2]   # [B, S, H, D/2]
        cos_b = cos.view(1, seq_len, 1, head_dim // 2)
        sin_b = sin.view(1, seq_len, 1, head_dim // 2)
        out_even = x_even * cos_b - x_odd * sin_b
        out_odd = x_even * sin_b + x_odd * cos_b
        rotated = torch.stack([out_even, out_odd], dim=-1).flatten(-2)

    return rotated.to(x.dtype)'''

if old_rope_apply in content:
    content = content.replace(old_rope_apply, new_rope_apply)
    with open(filepath, 'w') as f:
        f.write(content)
    print("SUCCESS: Patched rope_apply with PyTorch fallback")
else:
    # Try a more flexible match
    pattern = r'def rope_apply\(x, freqs\):.*?return rotated\.to\(x\.dtype\)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + new_rope_apply + content[match.end():]
        with open(filepath, 'w') as f:
            f.write(content)
        print("SUCCESS: Patched rope_apply (flexible match)")
    else:
        print("ERROR: Could not find rope_apply function to patch")
PATCH_SCRIPT

cd "$RCM_DIR"
export PYTHONPATH="$RCM_DIR"

PROMPT_IDX=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    PROMPT_IDX=$((PROMPT_IDX + 1))
    OUT_FILE="$OUTPUT_ROOT/rcm/rcm_4step_$(printf '%03d' $PROMPT_IDX).mp4"

    [ -f "$OUT_FILE" ] && { log "rCM prompt $PROMPT_IDX: SKIP (exists)"; continue; }

    log "rCM prompt $PROMPT_IDX: ${prompt:0:60}..."
    START_T=$(date +%s%N)

    $PYTHON rcm/inference/wan2pt1_t2v_rcm_infer.py \
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
        2>&1 | tail -10
    STATUS=$?

    END_T=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)

    if [ $STATUS -eq 0 ] && [ -f "$OUT_FILE" ]; then
        log "rCM prompt $PROMPT_IDX: OK (${ELAPSED}s)"
        echo "rCM prompt $PROMPT_IDX: ${ELAPSED}s [OK]" >> "$TIMING_FILE"
    else
        log "rCM prompt $PROMPT_IDX: FAILED (exit=$STATUS, ${ELAPSED}s)"
        echo "rCM prompt $PROMPT_IDX: ${ELAPSED}s [FAILED, exit=$STATUS]" >> "$TIMING_FILE"
    fi
done < "$PROMPT_FILE"

log "rCM inference complete"

# ============================================================
# Fix 4: TurboDiffusion — make ops import conditional
# ============================================================
log "=== Model 4: TurboDiffusion (4-step rCM + optimizations) ==="

MODIFY_FILE="$TURBO_DIR/turbodiffusion/inference/modify_model.py"
log "Patching TurboDiffusion modify_model.py (lazy ops import)..."

cp "$MODIFY_FILE" "${MODIFY_FILE}.bak_fix"

$PYTHON << 'PATCH_TURBO'
filepath = "/data/chenqingzhan/TurboDiffusion/turbodiffusion/inference/modify_model.py"
with open(filepath, 'r') as f:
    content = f.read()

# Make ops and SLA imports conditional
old_import1 = "from ops import FastLayerNorm, FastRMSNorm, Int8Linear"
new_import1 = """try:
    from ops import FastLayerNorm, FastRMSNorm, Int8Linear
    HAS_OPS = True
except ImportError:
    print("WARNING: 'ops' module not available, quantization features disabled")
    FastLayerNorm = FastRMSNorm = Int8Linear = None
    HAS_OPS = False"""

old_import2 = """from SLA import (
    SparseLinearAttention as SLA,
    SageSparseLinearAttention as SageSLA
)"""
new_import2 = """try:
    from SLA import (
        SparseLinearAttention as SLA,
        SageSparseLinearAttention as SageSLA
    )
    HAS_SLA = True
except ImportError:
    print("WARNING: 'SLA' module not available, sparse attention disabled")
    SLA = SageSLA = None
    HAS_SLA = False"""

if old_import1 in content:
    content = content.replace(old_import1, new_import1)
    print("SUCCESS: Patched ops import")
else:
    print("SKIP: ops import already patched or not found")

if old_import2 in content:
    content = content.replace(old_import2, new_import2)
    print("SUCCESS: Patched SLA import")
else:
    print("SKIP: SLA import already patched or not found")

with open(filepath, 'w') as f:
    f.write(content)
PATCH_TURBO

cd "$TURBO_DIR"
export PYTHONPATH="$TURBO_DIR:$RCM_DIR"

PROMPT_IDX=0
while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    PROMPT_IDX=$((PROMPT_IDX + 1))
    OUT_FILE="$OUTPUT_ROOT/turbodiffusion/turbo_4step_$(printf '%03d' $PROMPT_IDX).mp4"

    [ -f "$OUT_FILE" ] && { log "TurboDiffusion prompt $PROMPT_IDX: SKIP (exists)"; continue; }

    log "TurboDiffusion prompt $PROMPT_IDX: ${prompt:0:60}..."
    START_T=$(date +%s%N)

    $PYTHON turbodiffusion/inference/wan2.1_t2v_infer.py \
        --model Wan2.1-1.3B \
        --dit_path "$TURBO_CKPT" \
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
        2>&1 | tail -10
    STATUS=$?

    END_T=$(date +%s%N)
    ELAPSED=$(echo "scale=2; ($END_T - $START_T) / 1000000000" | bc)

    if [ $STATUS -eq 0 ] && [ -f "$OUT_FILE" ]; then
        log "TurboDiffusion prompt $PROMPT_IDX: OK (${ELAPSED}s)"
        echo "TurboDiffusion prompt $PROMPT_IDX: ${ELAPSED}s [OK]" >> "$TIMING_FILE"
    else
        log "TurboDiffusion prompt $PROMPT_IDX: FAILED (exit=$STATUS, ${ELAPSED}s)"
        echo "TurboDiffusion prompt $PROMPT_IDX: ${ELAPSED}s [FAILED, exit=$STATUS]" >> "$TIMING_FILE"
    fi
done < "$PROMPT_FILE"

log "TurboDiffusion inference complete"

# ============================================================
# Summary
# ============================================================
log "============================================================"
log "=== Summary ==="
echo "" >> "$TIMING_FILE"
echo "=== File Counts ===" >> "$TIMING_FILE"
echo "Teacher: $(ls "$OUTPUT_ROOT/teacher/"*.mp4 2>/dev/null | wc -l) videos" >> "$TIMING_FILE"
echo "CausVid: $(ls "$OUTPUT_ROOT/causvid/"*.mp4 2>/dev/null | wc -l) videos" >> "$TIMING_FILE"
echo "rCM: $(ls "$OUTPUT_ROOT/rcm/"*.mp4 2>/dev/null | wc -l) videos" >> "$TIMING_FILE"
echo "TurboDiffusion: $(ls "$OUTPUT_ROOT/turbodiffusion/"*.mp4 2>/dev/null | wc -l) videos" >> "$TIMING_FILE"

log "Output: $OUTPUT_ROOT"
log ""
cat "$TIMING_FILE"
log ""
log "=== Phase 0 Inference Comparison (Fixed) — Complete ==="
log "End time: $(date)"
