# Scripts — Task 3: DMD Distillation & Acceleration

Scripts for Phase 0 (FastGen familiarization & pretrained inference comparison) and Phase 1+ (progressive distillation training).

All scripts are designed to run on the GPU server (`111.17.197.107`) inside the `fastgen` conda environment unless noted otherwise.

---

## Directory Structure

```
scripts/
├── README.md               ← This file
├── setup/                   # Environment & model setup
├── data/                    # Training data preparation
├── configs/                 # Custom FastGen experiment configs
├── inference/               # Inference & evaluation scripts
└── training/                # Distillation training scripts
```

---

## setup/ — Environment & Model Setup

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_model.sh` | Download Wan2.1-T2V-1.3B model weights from HuggingFace (via hf-mirror) | `bash download_model.sh` |
| `download_causvid.sh` | Clone CausVid repo, download pretrained DMD checkpoint, patch attention for RTX 5090 | `bash download_causvid.sh` |

**Notes:**
- `download_model.sh` downloads to `/data/chenqingzhan/.cache/huggingface/`
- `download_causvid.sh` also patches `attention.py` for SDPA fallback (RTX 5090 has no flash_attn support)

---

## data/ — Training Data Preparation

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_openvid.sh` | Download OpenVid-1M dataset subset (filtered by quality/duration) | `bash download_openvid.sh` |
| `convert_to_webdataset.py` | Convert downloaded videos + captions into WebDataset tar shards for FastGen training | `python convert_to_webdataset.py --input_dir /path/to/videos --output_dir /path/to/shards` |
| `prepare_training_data.sh` | Full pipeline: download OpenVid-1M subset + convert to WebDataset | `bash prepare_training_data.sh` |

**Notes:**
- WebDataset format is required by FastGen's `wds_dataloaders.py`
- Filters: frames >= 81, aesthetic score >= 5.0, duration >= 2.0s
- Output: ~22 tar shards, ~22GB total

---

## configs/ — Custom FastGen Experiment Configs

| File | Purpose |
|------|---------|
| `config_cm_ct.py` | ECT (Enhanced Consistency Training) config for Wan2.1-T2V-1.3B — custom-created, not shipped with FastGen |
| `config_cm_cd.py` | CD (Consistency Distillation) config for Wan2.1-T2V-1.3B — custom-created |

**Usage:** Copy to `FastGen/fastgen/configs/experiments/WanT2V/` on the server, then reference via `--config`:
```bash
torchrun ... train.py --config fastgen/configs/experiments/WanT2V/config_cm_ct.py - ...
```

**Key difference from built-in configs:**
- FastGen ships DMD2/f-distill/LADD/MeanFlow configs for WanT2V, but NOT ECT/CD
- These were created by adapting `EDM2` consistency model configs to work with the Wan2.1 model

---

## inference/ — Inference & Evaluation Scripts

### Core Inference Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_inference.sh` | Run FastGen Teacher (50-step) or Student inference on single GPU | `bash run_inference.sh teacher` or `bash run_inference.sh student /path/to/ckpt` |
| `run_teacher_inference.sh` | Run Teacher inference with custom 5-prompt set (Phase 0 comparison) | `nohup bash run_teacher_inference.sh > teacher.log 2>&1 &` |
| `run_causvid_inference.sh` | Run CausVid (DMD-distilled, 3-step) inference | `bash run_causvid_inference.sh` |
| `run_ect_inference.sh` | Run ECT student inference using self-trained checkpoint | `bash run_ect_inference.sh` |

### Phase 0 Comparison Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `fix_and_run_inference.sh` | **Main comparison script** — fixes & runs Teacher + CausVid + rCM + TurboDiffusion inference | **Used for final Phase 0 results** |
| `run_all_inference_comparison.sh` | Original batch comparison script (all 4 models) | Superseded by `fix_and_run_inference.sh` (had config/import bugs) |

**`fix_and_run_inference.sh` details:**
- Fixes applied: Teacher uses direct Python (not torchrun), CausVid `pip install -e .`, rCM rope_apply patched, TurboDiffusion ops import made conditional
- Output: `/data/chenqingzhan/fastgen_output/comparison_2026_03_12/`
- Usage: `nohup bash fix_and_run_inference.sh > fix_comparison.log 2>&1 &`

### Utility Files

| File | Purpose |
|------|---------|
| `eval_prompts.txt` | 5 standardized evaluation prompts used across all inference comparisons |
| `patch_attention.py` | Python script to patch CausVid `attention.py` for PyTorch SDPA fallback (RTX 5090 compatibility) |

---

## training/ — Distillation Training Scripts

### Method-Specific Training

| Script | Method | GPUs | Notes |
|--------|--------|------|-------|
| `run_ect_single_gpu.sh` | ECT (Enhanced Consistency Training) | 4-GPU FSDP | Student only, lowest VRAM |
| `run_cd_single_gpu.sh` | CD (Consistency Distillation) | 4-GPU FSDP | Student + Teacher |
| `run_cd_train.sh` | CD (variant) | 2-GPU FSDP | Reduced GPU count version |
| `run_dmd2_single_gpu.sh` | DMD2 | 4-GPU FSDP | OOM on 32GB GPUs (4 networks) |
| `run_fdistill_single_gpu.sh` | f-distill | 4-GPU FSDP | OOM on 32GB GPUs |
| `run_ladd_single_gpu.sh` | LADD | 4-GPU FSDP | OOM on 32GB GPUs |
| `run_meanflow_single_gpu.sh` | MeanFlow | 1-GPU | Requires pre-computed latents |

### Launch Helpers

| Script | Purpose | Usage |
|--------|---------|-------|
| `launch_ect.sh` | Launch ECT training in background with nohup | `bash launch_ect.sh` |
| `launch_ect_now.sh` | Launch ECT on 4 specific free GPUs | `bash launch_ect_now.sh` |
| `launch_dmd2.sh` | Launch DMD2 training in background | `bash launch_dmd2.sh` |
| `run_all_sequential.sh` | Run all 5 methods sequentially (for comparison) | `bash run_all_sequential.sh` |

### VRAM Feasibility (RTX 5090 32GB, Wan2.1-1.3B)

| Method | Feasible? | Networks in Memory |
|--------|-----------|-------------------|
| ECT | Yes (25.8 GB/GPU) | Student only |
| CD | Yes (marginal, ~28-30 GB/GPU) | Student + Teacher |
| DMD2 | No (OOM) | Student + Teacher + FakeScore + Discriminator |
| f-distill | No (OOM) | Same as DMD2 |
| LADD | No (OOM) | Student + Teacher + Discriminator |
| MeanFlow | Yes (untested) | Student only |

---

## Common Environment Variables

All scripts expect these environment variables (set within each script):

```bash
export CUDA_VISIBLE_DEVICES=0        # GPU index (0-7)
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"   # China mirror for HuggingFace
```

## Server Paths

| Path | Content |
|------|---------|
| `/data/chenqingzhan/FastGen/` | NVIDIA FastGen framework |
| `/data/chenqingzhan/CausVid/` | CausVid (DMD) codebase |
| `/data/chenqingzhan/rcm/` | NVlabs rCM codebase |
| `/data/chenqingzhan/TurboDiffusion/` | TurboDiffusion codebase |
| `/data/chenqingzhan/fastgen_output/` | All training & inference outputs |
| `/data/chenqingzhan/miniconda3/envs/fastgen/` | Conda environment |
