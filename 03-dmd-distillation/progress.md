# Phase 0 Progress — DMD Distillation Reproduction

> **Author:** Chen Hing Chin (陈庆展)
> **Branch:** `Task3_dev_ChenHingChin`
> **Start Date:** 2026-03-02
> **Server:** 8x RTX 5090 32GB @ 111.17.197.107 (single GPU mode)

---

## Task Overview

| # | Task | Status | Date |
|---|------|--------|------|
| 1 | FastGen 环境搭建 | ✅ Done | 2026-03-05 |
| 2 | DMD2 复现 (Wan2.1 1.3B) | 🔄 In Progress | - |
| 3 | ECT 复现 | ⬜ Pending | - |
| 4 | Consistency Distillation 复现 | ⬜ Pending | - |
| 5 | 多方法对比 & 复现报告 | ⬜ Pending | - |

---

## Task 1: FastGen 环境搭建

### 1.1 Server Specs

| Item | Value |
|------|-------|
| GPU | 8x NVIDIA RTX 5090 32GB (use single GPU only) |
| CPU | 384 cores |
| RAM | 1TB |
| Disk | 21TB total, ~10TB free |
| OS | Ubuntu, Linux 5.15.0 |
| Python (system) | 3.10.12 |
| Python (fastgen env) | 3.12.12 |
| CUDA Driver | 580.65.06 |
| PyTorch | 2.10.0+cu128 |
| CUDA Toolkit | 12.8 |

### 1.2 Setup Steps

- [x] Install Miniconda on server
- [x] Create conda env `fastgen` (Python 3.12.12)
- [x] Clone FastGen repo
- [x] Install FastGen + dependencies (PyTorch 2.10.0+cu128, fastgen 0.1.0)
- [x] Upload `hf-download.py` to server
- [x] Download Wan2.1-T2V-1.3B model weights
- [x] Verify: run FastGen inference (single GPU)

### 1.3 Execution Log

**2026-03-05: Environment setup complete**
- PyTorch 2.10.0+cu128 installed (CUDA 12.8, RTX 5090 supported)
- FastGen 0.1.0 installed with all dependencies (diffusers, transformers, accelerate, wandb, etc.)
- Wan2.1-T2V-1.3B-Diffusers model downloaded to `/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/`
- Teacher inference (50-step, cfg=5.0) verified: 5 videos generated successfully
- Performance: ~167s per video (3.35s/step), single GPU RTX 5090
- Output: `/data/chenqingzhan/fastgen_output/fastgen/wan_dmd2/wan_t2v_test/inference_validation/`
- Note: text_encoder (UMT5) auto-downloaded from hf-mirror.com during first run
- Model path override needed: `model.net.model_id_or_local_path=/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers`

---

## Task 2: DMD2 Reproduction

### Goal
Reproduce DMD2 distillation on Wan2.1-1.3B using FastGen (50→4 step, single GPU).

### Key Config
- Model: Wan2.1-T2V-1.3B
- Config: `fastgen/configs/experiments/WanT2V/config_dmd2.py`
- Student steps: 4
- Single GPU, batch_size=1, gradient accumulation
- GAN loss weight: 0.03, guidance scale: 5.0

### Training Data Requirements

**Format:** WebDataset (.tar shards), each shard containing (mp4, txt) pairs.

```
shard_00000.tar
├── sample_000000.mp4    # Video file (832x480, ≥81 frames)
├── sample_000000.txt    # Text caption
├── sample_000001.mp4
├── sample_000001.txt
└── ...
```

**Video specs for Wan2.1 1.3B DMD2:**
| Parameter | Value |
|-----------|-------|
| Resolution | 832x480 (480p) |
| Frame count | ≥81 frames |
| Latent shape | [16, 21, 60, 104] (C, T, H, W) |
| VAE compression | Temporal 4x, Spatial 8x8 |

**Key finding: NVIDIA used synthetic data for DMD2 training.**
- They used [VidProM](https://vidprom.github.io/) prompt set for text captions
- Generated synthetic training videos using Wan2.1 14B Teacher model (50-step)
- Packed (generated_video.mp4, prompt.txt) pairs into WebDataset shards
- DMD2 adversarial distillation does NOT require real video data
- Their result: VBench Total Score = 84.72 with 4-step student

**Data preparation script (from FastGen docs):**
```python
import webdataset as wds

with wds.ShardWriter("/path/to/shards/%05d.tar", maxcount=1000) as sink:
    for idx, (video_path, caption) in enumerate(dataset):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        sink.write({
            "__key__": f"sample_{idx:06d}",
            "mp4": video_bytes,
            "txt": caption.encode("utf-8"),
        })
```

**Data path config:**
```python
config.dataloader_train.datatags = ["WDS:/path/to/your/shards"]
```

### Data Strategy Options

| Option | Pros | Cons |
|--------|------|------|
| A: Use VidProM prompts + 1.3B Teacher to generate data | Simple, follows NVIDIA's approach | 1.3B quality < 14B, takes time to generate |
| B: Use project's own product video data + captions | Domain-specific, better for PVTT | Need to prepare captions, resize to 480p |
| C: Download public video dataset (Panda-70M, OpenVid) | Large scale, diverse | Huge download, not product-domain |

### Steps
- [ ] Decide data strategy (A / B / C)
- [ ] Prepare training data (WebDataset shards)
- [ ] Adapt config for single GPU (nproc=1, adjust batch/accumulation)
- [ ] Launch DMD2 training
- [ ] Monitor loss curves (W&B or local logs)
- [ ] Run student inference and evaluate quality

---

## Task 3: ECT Reproduction

### Goal
Reproduce ECT (Enhanced Consistency Training) on Wan2.1-1.3B.

### Approach: Custom WanT2V CM Config (Route 1)

**Problem:** FastGen has no WanT2V config for CM/ECT. Only EDM/EDM2 configs exist.

**Solution:** Created custom `config_cm_ct.py` by combining:
- **Method logic:** `fastgen/methods/consistency_model/CM.py` (use_cd=False for ECT)
- **EDM2 CM reference:** `EDM2/config_cm_s.py` (CTSchedule, EMA, loss params)
- **Wan adaptation:** `WanT2V/config_mf.py` (same consistency family, time sampling params)

### How CM/ECT/CD Works in FastGen

```python
# CM.py — single_train_step() core logic:
#
# 1. Sample t from time distribution
# 2. Add noise: y_t = noise_scheduler.forward_process(data, eps, t)
# 3. Compute r from t using sigmoid mapping (ECT paper):
#    r = t - t * (1 - ratio) * (1 + 8 * sigmoid(-t))
# 4. Target:
#    - ECT (use_cd=False): y_r = noise_scheduler.forward_process(data, eps, r)
#    - CD  (use_cd=True):  y_r = ode_solver(teacher, y_t, t, r)  # Teacher ODE step
# 5. Loss = ||net(y_t, t) - net(y_r, r)||^2  (consistency constraint)
# 6. CTScheduleCallback controls `ratio` curriculum (increases over training)
```

### Key Config Design Decisions

| Parameter | Value | Source / Reasoning |
|-----------|-------|--------------------|
| `use_cd` | False | ECT = Consistency Training (no teacher) |
| `time_dist_type` | logitnormal | From MeanFlow WanT2V (flow matching) |
| `train_p_mean` / `train_p_std` | -0.8 / 1.6 | From MeanFlow WanT2V |
| `huber_const` | 0.06 | From EDM2 CM config |
| `weighting_ct_loss` | "default" (1/(t-r)) | Standard CM loss weighting |
| `EMA type` | power (gamma=96.99) | From MeanFlow WanT2V |
| `optimizer` | AdamW, lr=1e-5 | From MeanFlow WanT2V |
| `kimg_per_stage` | 50 | Scaled for batch_size=8, max_iter=6000 |
| `q` | 4 | From EDM2 CM config (controls curriculum speed) |
| Data format | mp4+txt (VideoLoader) | **Same as DMD2** — shared dataset |

### Validation Strategy
1. Run 100-200 step sanity check — verify loss decreases, no NaN
2. Cross-reference with distill_wan2.1 (azuresky03) if results seem off
3. Compare loss curve shape with published ECT results on EDM2

### Setup Status
- [x] Config created: `config_cm_ct.py` → uploaded to `FastGen/fastgen/configs/experiments/WanT2V/`
- [x] Training script: `run_ect_single_gpu.sh` → uploaded to server
- [ ] Prepare training data (WebDataset shards, shared with DMD2)
- [ ] Run sanity check (100-200 steps)
- [ ] Launch full ECT training
- [ ] Record convergence speed & quality

---

## Task 4: Consistency Distillation (CD) Reproduction

### Goal
Reproduce Consistency Distillation on Wan2.1-1.3B.

### Approach
Same as Task 3 but with `use_cd=True` — uses Teacher model for ODE-based target.

### Key Differences from ECT
| | ECT (config_cm_ct.py) | CD (config_cm_cd.py) |
|--|---|---|
| `use_cd` | False | True |
| Teacher | Not needed | Required (Wan2.1-1.3B loaded as teacher) |
| Target y_r | `forward_process(data, eps, r)` | `ode_solver(teacher, y_t, t, r)` |
| `guidance_scale` | N/A | 5.0 (CFG for teacher) |
| VRAM usage | Lower (no teacher) | Higher (teacher + student) |

### Setup Status
- [x] Config created: `config_cm_cd.py` → uploaded to `FastGen/fastgen/configs/experiments/WanT2V/`
- [x] Training script: `run_cd_single_gpu.sh` → uploaded to server
- [ ] Prepare training data (WebDataset shards, shared with DMD2 and ECT)
- [ ] Run sanity check (100-200 steps)
- [ ] Launch full CD training
- [ ] Record convergence speed & quality

---

## Additional Methods (Bonus Comparison)

### f-distill & LADD (Ready-made WanT2V configs)

These are additional distribution matching methods with **existing** WanT2V configs.
They share the same mp4+txt data format as DMD2/ECT/CD.

| Method | Config | Description | Script |
|--------|--------|-------------|--------|
| f-distill | `config_fdistill.py` (built-in) | f-divergence weighted DMD2 | `run_fdistill_single_gpu.sh` |
| LADD | `config_ladd.py` (built-in) | Pure adversarial distillation | `run_ladd_single_gpu.sh` |
| MeanFlow | `config_mf.py` (built-in) | Consistency family, pre-computed latents | `run_meanflow_single_gpu.sh` |

### Data Format Groups

```
Group A: VideoLoaderConfig (mp4 + txt WebDataset shards)
  → DMD2, ECT, CD, f-distill, LADD
  → One dataset serves ALL five methods ← KEY ADVANTAGE

Group B: VideoLatentLoaderConfig (pre-computed latent.pth + txt_emb.pth)
  → MeanFlow
  → Requires extra pre-processing step
```

### Setup Status
- [x] f-distill training script created and uploaded
- [x] LADD training script created and uploaded
- [x] MeanFlow training script created and uploaded
- [ ] Pre-compute VAE latents + text embeddings from training videos
- [ ] Prepare pre-computed negative prompt embedding (.npy)
- [ ] Launch MeanFlow training
- [ ] Record convergence speed & quality

---

## Task 5: Method Comparison & Report

### Comparison Axes
| Metric | DMD2 | ECT | CD | f-distill | LADD |
|--------|------|-----|-----|-----------|------|
| Convergence speed (iterations) | - | - | - | - | - |
| Final FVD | - | - | - | - | - |
| Final CLIP-I | - | - | - | - | - |
| VRAM usage (GB) | - | - | - | - | - |
| Training time (hours) | - | - | - | - | - |
| Inference steps | - | - | - | - | - |
| Generation quality (subjective) | - | - | - | - | - |

### Deliverables
- [ ] Comparison table filled
- [ ] Loss curve plots
- [ ] Sample video comparisons
- [ ] Written reproduction report
- [ ] Framework evaluation & recommendation

---

## Appendix A: Server File Paths

```
/data/chenqingzhan/
├── miniconda3/                          # Miniconda installation
│   └── envs/fastgen/                    # fastgen conda env (Python 3.12.12)
│       └── bin/
│           ├── python                   # Python interpreter
│           ├── pip                       # pip
│           └── huggingface-cli          # HF CLI tool
├── FastGen/                             # NVIDIA FastGen repo (cloned)
│   ├── train.py                         # Training entry point
│   ├── scripts/inference/               # Inference scripts
│   │   └── video_model_inference.py     # Video inference script
│   ├── fastgen/
│   │   ├── configs/
│   │   │   ├── data.py                  # VideoLoaderConfig / VideoLatentLoaderConfig
│   │   │   ├── net.py                   # Wan_1_3B_Config (model_id_or_local_path)
│   │   │   └── experiments/WanT2V/
│   │   │       ├── config_dmd2.py       # DMD2 config (1.3B, built-in)
│   │   │       ├── config_fdistill.py   # f-distill config (built-in)
│   │   │       ├── config_ladd.py       # LADD config (built-in)
│   │   │       ├── config_mf.py         # MeanFlow config (built-in)
│   │   │       ├── config_cm_ct.py      # ECT config (CUSTOM, we created)
│   │   │       ├── config_cm_cd.py      # CD config (CUSTOM, we created)
│   │   │       └── ...                  # KD, SF, SFT, etc.
│   │   ├── datasets/
│   │   │   ├── README.md                # Data preparation guide
│   │   │   ├── wds_dataloaders.py       # WebDataset loader
│   │   │   └── decoders.py              # Video decoder
│   │   └── methods/
│   │       ├── distribution_matching/
│   │       │   ├── dmd2.py              # DMD2 method
│   │       │   ├── f_distill.py         # f-distill method
│   │       │   ├── ladd.py              # LADD method
│   │       │   └── README.md            # Expected results
│   │       └── consistency_model/
│   │           ├── CM.py                # ECT/CD method (use_cd=False/True)
│   │           ├── mean_flow.py         # MeanFlow method
│   │           └── README.md            # Expected results
│   └── scripts/inference/prompts/       # Prompt files for inference
├── .cache/huggingface/
│   └── models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/   # Model weights
│       ├── model_index.json
│       ├── transformer/                 # DiT transformer weights (2 shards)
│       ├── text_encoder/                # UMT5-XXL text encoder (5 shards)
│       ├── vae/                         # VAE weights
│       ├── tokenizer/                   # Tokenizer
│       └── scheduler/                   # Noise scheduler config
├── fastgen_output/                      # Training & inference output root
│   └── fastgen/wan_dmd2/wan_t2v_test/   # Teacher inference results
│       └── inference_validation/
│           └── validation_aug_qwen_2_5_14b_seed42/
│               ├── teacher_cfg5.0_steps50_0000_seed42.mp4
│               ├── teacher_cfg5.0_steps50_0001_seed42.mp4
│               ├── teacher_cfg5.0_steps50_0002_seed42.mp4
│               ├── teacher_cfg5.0_steps50_0003_seed42.mp4
│               └── teacher_cfg5.0_steps50_0004_seed42.mp4
├── hf-download.py                       # HuggingFace mirror download script
├── setup_server.sh                      # Server setup script
├── scripts/
│   ├── download_model.sh               # Model download script
│   ├── run_inference.sh                 # Inference script
│   ├── run_dmd2_single_gpu.sh          # DMD2 training script
│   ├── run_ect_single_gpu.sh           # ECT training script (custom config)
│   ├── run_cd_single_gpu.sh            # CD training script (custom config)
│   ├── run_fdistill_single_gpu.sh      # f-distill training script
│   ├── run_ladd_single_gpu.sh          # LADD training script
│   └── run_meanflow_single_gpu.sh      # MeanFlow training script
├── setup.log                            # Setup execution log
├── download.log                         # Model download log
└── inference.log                        # Inference execution log
```

## Appendix B: Key Commands Reference

```bash
# SSH to server
ssh chenqingzhan@111.17.197.107

# Activate conda env
source /data/chenqingzhan/miniconda3/bin/activate fastgen

# Environment variables (also in ~/.bashrc)
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

# Teacher inference (50-step, single GPU)
cd /data/chenqingzhan/FastGen
PYTHONPATH=$(pwd) python scripts/inference/video_model_inference.py \
    --do_student_sampling False \
    --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.seed=42 trainer.ddp=False model.guidance_scale=5.0 \
      model.net.model_id_or_local_path=/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers \
      log_config.name=wan_t2v_test

# DMD2 training (single GPU, to be configured)
python train.py \
    --config=fastgen/configs/experiments/WanT2V/config_dmd2.py \
    - trainer.ddp=False trainer.fsdp=False \
      trainer.batch_size_global=8 trainer.max_iter=6000 \
      model.net.model_id_or_local_path=/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers \
      dataloader_train.datatags='["WDS:/path/to/training/shards"]' \
      log_config.name=dmd2_wan1.3b_single_gpu
```

## Appendix C: Installed Package Versions

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| fastgen | 0.1.0 |
| diffusers | 0.35.1 |
| transformers | 4.49.0 |
| accelerate | 1.13.0 |
| huggingface-hub | 0.36.2 |
| wandb | 0.25.0 |
| webdataset | 1.0.2 |
| safetensors | 0.7.0 |
| triton | 3.6.0 |
