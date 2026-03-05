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
Reproduce Enhanced Consistency Training on Wan2.1-1.3B.

### Steps
- [ ] Locate ECT config in FastGen
- [ ] Adapt for single GPU
- [ ] Launch ECT training
- [ ] Record convergence speed & quality

---

## Task 4: Consistency Distillation Reproduction

### Goal
Reproduce Consistency Distillation as baseline.

### Steps
- [ ] Locate CD config in FastGen
- [ ] Adapt for single GPU
- [ ] Launch CD training
- [ ] Record convergence speed & quality

---

## Task 5: Method Comparison & Report

### Comparison Axes
| Metric | DMD2 | ECT | Consistency |
|--------|------|-----|-------------|
| Convergence speed (iterations) | - | - | - |
| Final FVD | - | - | - |
| Final CLIP-I | - | - | - |
| VRAM usage (GB) | - | - | - |
| Training time (hours) | - | - | - |
| Inference steps | - | - | - |
| Generation quality (subjective) | - | - | - |

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
│   │   │       ├── config_dmd2.py       # DMD2 config (1.3B)
│   │   │       ├── config_dmd2_14b.py   # DMD2 config (14B)
│   │   │       └── ...                  # ECT, MF, SF, SFT, etc.
│   │   ├── datasets/
│   │   │   ├── README.md                # Data preparation guide
│   │   │   ├── wds_dataloaders.py       # WebDataset loader
│   │   │   └── decoders.py              # Video decoder
│   │   └── methods/
│   │       └── distribution_matching/
│   │           ├── dmd2.py              # DMD2 method implementation
│   │           └── README.md            # DMD2 expected results
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
│   └── run_dmd2_single_gpu.sh          # DMD2 single-GPU training script
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
