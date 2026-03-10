# Phase 0 Report — FastGen Video Networks: Architecture, Inference & Capability Boundaries

> **Author:** Chen Hing Chin (陈庆展)
> **Date:** 2026-03-10 (Draft, in progress)
> **Task:** Task 3 — DMD Distillation & Acceleration
> **Branch:** `Task3_dev_ChenHingChin`
> **Status:** Draft — sections marked `[TODO]` are pending completion

---

## 0. Learning Plan & Roadmap

### 0.1 Task Objective

> "找一个公开数据集，先熟悉一下 FastGen 的 Video Networks，熟悉训练以及推理，形成 report，探索 FastGen 架构的能力边界。"

The goal of Phase 0 is **not** to reproduce full training from scratch, but to:

1. **Understand** FastGen's architecture and code structure
2. **Run inference** with pretrained checkpoints (DMD, ECT, CD) to compare distillation methods
3. **Explore** what FastGen can and cannot do — its capability boundaries
4. **Prepare** for Phase 1 progressive distillation with informed architectural understanding

### 0.2 Learning Path

```
Step 1: FastGen Architecture Analysis                    ← Section 1-2
  ├── Codebase structure & module organization
  ├── Video Networks: Wan2.1 DiT architecture
  ├── VAE, Text Encoder, Noise Scheduler
  └── How distillation methods plug into the framework

Step 2: Distillation Methods Survey                      ← Section 3
  ├── Taxonomy: Consistency-based vs Distribution Matching vs Adversarial
  ├── What each method does conceptually
  └── Tradeoffs: quality, speed, VRAM, complexity

Step 3: Inference Experiments with Pretrained Models     ← Section 4
  ├── Teacher baseline (50-step Wan2.1-1.3B)
  ├── DMD distilled (CausVid pretrained, 3-step)
  ├── ECT distilled (self-trained checkpoint, 1-step)
  ├── CD distilled (partial checkpoint, 4-step)
  └── Unified prompt set, speed & quality comparison

Step 4: Training Pipeline Walkthrough                    ← Section 5
  ├── Code-level training flow (train.py → Method → Trainer)
  ├── Data pipeline: WebDataset → VideoLoader → Latent
  ├── Key configurations and hyperparameters
  └── Short sanity training (100-iter) to verify understanding

Step 5: Capability Boundary Exploration                  ← Section 6
  ├── VRAM limits per method (measured)
  ├── Resolution / frame count scalability
  ├── Supported models & methods matrix
  ├── What is NOT supported or requires custom work
  └── Implications for Phase 1 (14B model, product video editing)

Step 6: Summary & Phase 1 Recommendations               ← Section 7
```

### 0.3 Resources Used

| Resource | Purpose |
|----------|---------|
| [NVIDIA FastGen](https://github.com/NVlabs/FastGen) | Primary framework under study |
| [CausVid](https://github.com/tianweiy/CausVid) (CVPR 2025) | DMD pretrained checkpoint |
| OpenVid-1M | Public video-text dataset for data pipeline testing |
| Wan2.1-T2V-1.3B | Base video generation model (DiT architecture) |

---

## 1. FastGen Framework Architecture

### 1.1 Overview

`[TODO]` High-level description of FastGen's purpose and design philosophy.

- What is FastGen? NVIDIA's unified framework for diffusion model distillation
- Target use case: accelerating video/image diffusion models from many-step to few-step
- Key value proposition: multiple distillation methods under one API

### 1.2 Codebase Structure

`[TODO]` Annotated directory tree of FastGen with key file descriptions.

```
FastGen/
├── train.py                            # Training entry point
├── scripts/inference/
│   └── video_model_inference.py        # Inference entry point
├── fastgen/
│   ├── configs/
│   │   ├── base.py                     # [TODO] Base config classes
│   │   ├── data.py                     # VideoLoaderConfig / VideoLatentLoaderConfig
│   │   ├── net.py                      # Model configs (Wan_1_3B_Config, etc.)
│   │   └── experiments/WanT2V/         # Wan2.1 experiment configs
│   │       ├── config_dmd2.py          # DMD2
│   │       ├── config_fdistill.py      # f-divergence distillation
│   │       ├── config_ladd.py          # Latent Adversarial Distillation
│   │       ├── config_mf.py            # MeanFlow
│   │       ├── config_cm_ct.py         # ECT (custom)
│   │       └── config_cm_cd.py         # CD (custom)
│   ├── datasets/
│   │   ├── wds_dataloaders.py          # [TODO] WebDataset loading
│   │   └── decoders.py                 # [TODO] Video decoding
│   ├── methods/
│   │   ├── distribution_matching/
│   │   │   ├── dmd2.py                 # [TODO] DMD2 method
│   │   │   ├── f_distill.py            # [TODO] f-distill method
│   │   │   └── ladd.py                 # [TODO] LADD method
│   │   └── consistency_model/
│   │       ├── CM.py                   # [TODO] ECT/CD method
│   │       └── mean_flow.py            # [TODO] MeanFlow method
│   ├── trainer.py                      # [TODO] Training loop
│   ├── callbacks/                      # [TODO] Logging, profiling, scheduling
│   └── utils/                          # [TODO] Checkpointing, FSDP, etc.
```

### 1.3 Core Abstractions

`[TODO]` Describe the key abstractions in FastGen:

- **Config system:** How experiment configs compose (base → model → method → data)
- **Method:** Base class for distillation methods (`single_train_step()`)
- **Trainer:** Training loop with FSDP/DDP support
- **Callbacks:** Modular hooks for logging, checkpointing, scheduling
- **Net:** Model wrapper that handles teacher/student loading

---

## 2. Video Networks: Wan2.1 Architecture

### 2.1 Wan2.1 DiT (Diffusion Transformer)

`[TODO]` Detailed analysis of Wan2.1's architecture:

- DiT (Diffusion Transformer) design
- 3D attention: spatial + temporal
- Model variants: 1.3B vs 14B parameter count
- Flow matching noise schedule

### 2.2 Components

| Component | Architecture | Parameters | Purpose |
|-----------|-------------|------------|---------|
| **Transformer (DiT)** | Diffusion Transformer | 1.3B | Core denoising network |
| **VAE** | 3D Causal VAE | ~100M | Video ↔ Latent encoding (4x temporal, 8x8 spatial) |
| **Text Encoder** | UMT5-XXL | ~4.7B | Text prompt → embedding |
| **Noise Scheduler** | Flow Matching | - | Controls diffusion process |

### 2.3 Latent Space

| Parameter | Value |
|-----------|-------|
| Input video | 832 x 480, 81 frames |
| Latent shape | `[16, 21, 60, 104]` (C, T, H, W) |
| Temporal compression | 4x (81 frames → 21 latent frames) |
| Spatial compression | 8x8 (832x480 → 104x60) |

`[TODO]` How the latent shape affects VRAM usage and method feasibility.

---

## 3. Distillation Methods in FastGen

### 3.1 Method Taxonomy

`[TODO]` Conceptual explanation of each distillation family.

```
Distillation Methods in FastGen
│
├── Consistency-Based (learn self-consistency along ODE trajectories)
│   ├── ECT (Enhanced Consistency Training) — no teacher needed
│   ├── CD  (Consistency Distillation)      — teacher provides ODE targets
│   └── MeanFlow                            — pre-computed latents, mean flow matching
│
└── Distribution Matching (adversarial / score matching)
    ├── DMD2      — VSD loss + GAN loss + FakeScore network
    ├── f-distill — f-divergence weighted DMD2 variant
    └── LADD      — Pure adversarial (GAN only, no VSD/FakeScore)
```

### 3.2 Method Comparison (Conceptual)

`[TODO]` For each method, explain:
- Core idea in 2-3 sentences
- Loss function
- Required networks (Student, Teacher, Discriminator, FakeScore)
- Expected tradeoffs (quality vs speed vs VRAM)

### 3.3 How Methods Plug Into FastGen

`[TODO]` Code-level walkthrough of how a method implements `single_train_step()`.

Example: ECT in `CM.py`:
```python
# Pseudocode — ECT single_train_step():
# 1. Sample timestep t from logitnormal distribution
# 2. Add noise: y_t = forward_process(data, eps, t)
# 3. Compute target timestep r using sigmoid mapping
# 4. ECT target: y_r = forward_process(data, eps, r)
# 5. Loss = ||student(y_t, t) - student(y_r, r)||^2
# 6. CTSchedule controls ratio curriculum (increases over training)
```

---

## 4. Inference Experiments

### 4.1 Experimental Setup

| Item | Value |
|------|-------|
| GPU | Single RTX 5090 32GB |
| Evaluation Prompts | 5 standardized prompts (see Section 4.2) |
| Precision | bfloat16 |
| Seed | 42 |
| Models Compared | Teacher (50-step), CausVid DMD (3-step), ECT (1-step), CD (4-step) |

### 4.2 Evaluation Prompts

| # | Prompt |
|---|--------|
| 1 | A joyful child swinging on a colorful swing set in a sunny park |
| 2 | A yellow rubber duck floating gently on a calm pond |
| 3 | Cyclists crossing a busy intersection in a modern city |
| 4 | An astronaut riding a cow through a green meadow |
| 5 | A bird flying gracefully toward a medieval church tower |

### 4.3 Results: Teacher Baseline (50-step)

| Metric | Value |
|--------|-------|
| Denoising Steps | 50 |
| CFG Scale | 5.0 |
| Per-Video Time | **167s** |
| Quality | Reference standard |

### 4.4 Results: CausVid DMD (3-step, pretrained)

| Metric | Value |
|--------|-------|
| Source | Pretrained checkpoint (`tianweiy/CausVid`) |
| Denoising Steps | 3 (timesteps: 1000 → 757 → 522) |
| CFG Scale | 3.5 |
| Per-Video Time | **9.26s** |
| Speedup vs Teacher | **18x** |

Note: Required patching `attention.py` for SDPA fallback (no `flash_attn` on RTX 5090 Blackwell).

### 4.5 Results: ECT (1-step, self-trained 6000 iter)

| Metric | Value |
|--------|-------|
| Source | Self-trained, 6000 iter, 4-GPU FSDP |
| Denoising Steps | **1** |
| Per-Video Time | **1.7s** |
| Speedup vs Teacher | **94x** |
| Training Time | 36.5 hours |

### 4.6 Results: CD (4-step, partial checkpoint iter 1500)

`[TODO]` Run inference with iter 1500 checkpoint and record results.

| Metric | Value |
|--------|-------|
| Source | Self-trained, stopped at iter 1500 / 6000 |
| Denoising Steps | 4 |
| Per-Video Time | TBD |
| Speedup vs Teacher | TBD |

### 4.7 Speed Comparison Summary

| Model | Steps | Time/Video | Speedup | Quality |
|-------|-------|-----------|---------|---------|
| Teacher (Wan2.1-1.3B) | 50 | 167s | 1x | Reference |
| CausVid (DMD pretrained) | 3 | 9.26s | 18x | `[TODO]` |
| ECT (self-trained) | 1 | 1.7s | 94x | `[TODO]` |
| CD (partial, iter 1500) | 4 | TBD | TBD | `[TODO]` |

### 4.8 Qualitative Comparison

`[TODO]` Side-by-side screenshots or video quality notes for each method.

### 4.9 Quantitative Quality Metrics

`[TODO]` Compute and fill in:
- CLIP Score (text-video alignment)
- LPIPS (student vs teacher perceptual similarity)
- VBench (if setup ready)

---

## 5. Training Pipeline Analysis

### 5.1 Training Flow (Code Walkthrough)

`[TODO]` Trace the execution path of a training run:

```
train.py
  → parse config (Hydra-like custom parser)
  → build model (student, [teacher], [discriminator], [fake_score])
  → build dataloader (WebDataset → VideoLoader → VAE encode)
  → build optimizer (AdamW)
  → Trainer.train()
    → for each iteration:
      → method.single_train_step(batch)
      → callbacks: logging, checkpointing, validation, GPU stats
      → EMA update
```

### 5.2 Data Pipeline

`[TODO]` Trace how training data flows through FastGen:

```
WebDataset (.tar shards)
  → wds_dataloaders.py: load mp4 + txt pairs
  → decoders.py: decode video frames
  → VideoLoaderConfig: resize, crop, normalize
  → VAE encoder: video → latent [16, 21, 60, 104]
  → Text encoder (UMT5): prompt → text embedding
  → Method: compute loss on latent space
```

Data prepared for this study:
- **Source:** OpenVid-1M (public dataset, 1M+ video-text pairs)
- **Filtered:** 21,133 samples (frames >= 81, aesthetic >= 5.0, duration >= 2s)
- **Format:** 22 WebDataset tar shards, 22GB total

### 5.3 Key Configuration Parameters

`[TODO]` Document the most important hyperparameters and what they control.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `trainer.batch_size_global` | Total batch size across GPUs | 8 |
| `trainer.max_iter` | Total training iterations | 6000 |
| `trainer.fsdp` | Enable Fully Sharded Data Parallel | True |
| `model.guidance_scale` | Classifier-Free Guidance scale | 5.0 |
| `model.num_steps_student` | Student denoising steps | 4 |
| ... | ... | ... |

### 5.4 Short Training Experiment (Sanity Check)

`[TODO]` Run 100-200 iterations of ECT or CD to demonstrate understanding of the training pipeline. Document:
- How to launch training
- How to monitor progress (logs, W&B)
- How to interpret loss values
- How checkpoints are saved and loaded

---

## 6. Capability Boundaries of FastGen

### 6.1 Supported Models

`[TODO]` List all video/image models supported by FastGen.

| Model | Type | Configs Available |
|-------|------|------------------|
| Wan2.1-T2V-1.3B | Text-to-Video | DMD2, ECT, CD, f-distill, LADD, MeanFlow |
| Wan2.1-T2V-14B | Text-to-Video | `[TODO]` check |
| ... | ... | ... |

### 6.2 Supported Distillation Methods

| Method | Image | Video | Wan2.1 Config? | Tested? |
|--------|-------|-------|---------------|---------|
| DMD2 | Yes | Yes | Built-in | OOM on 4x32GB |
| f-distill | Yes | Yes | Built-in | OOM on 4x32GB |
| LADD | Yes | Yes | Built-in | OOM on 4x32GB |
| MeanFlow | Yes | Yes | Built-in | Not tested |
| ECT | Yes | Yes | Custom | Trained successfully |
| CD | Yes | Yes | Custom | Partially trained |

### 6.3 VRAM Constraints (Measured)

A critical finding: Wan2.1-1.3B's latent dimensions ([16, 21, 60, 104]) create significant memory pressure.

| Method | Networks in Memory | Peak VRAM/GPU (4-GPU FSDP) | Feasible on 4x32GB? |
|--------|-------------------|---------------------------|---------------------|
| ECT | Student only | **25.8 GB** | Yes |
| CD | Student + Teacher | **~28-30 GB** | Yes (marginal) |
| LADD | + Discriminator | **>31 GB** | No (OOM) |
| DMD2 | + FakeScore + Disc | **>31 GB** | No (OOM) |
| f-distill | + FakeScore + Disc | **>31 GB** | No (OOM) |

**Root cause:** FSDP all-gathers full parameters during forward pass. Text encoder (UMT5-XXL ~10GB) replicated on each GPU.

**Implication for Phase 1:** The 14B model will require A100/H100 80GB GPUs, or aggressive memory optimization (activation checkpointing, CPU offload, model parallelism).

### 6.4 What FastGen Does NOT Support (Out of the Box)

`[TODO]` Document limitations discovered:

- No built-in WanT2V config for ECT/CD (had to create custom configs)
- No flash_attn support for RTX 5090 Blackwell (CausVid needed patching)
- `[TODO]` Other limitations found during exploration

### 6.5 Scalability Analysis

`[TODO]` How do the following scale?
- Resolution increase (480p → 720p → 1080p)
- Frame count increase (81 → 161 → 241)
- Model size increase (1.3B → 14B)
- Batch size scaling

---

## 7. Summary & Phase 1 Recommendations

### 7.1 Key Findings

`[TODO]` Summarize top 5 findings from Phase 0.

1. **FastGen architecture:** ...
2. **Speed vs quality tradeoff:** ECT achieves 94x speedup (1-step), CausVid 18x (3-step)
3. **VRAM is the primary bottleneck:** Only consistency-based methods fit on 4x32GB
4. **Custom config creation:** FastGen requires custom work for unsupported model-method combinations
5. **...**: ...

### 7.2 Method Recommendation for Phase 1

`[TODO]` Based on all findings, recommend which distillation method(s) to use for Phase 1 progressive distillation (50 → 16 → 8 → 4 steps) with the Task 2 Teacher Model.

### 7.3 Hardware Recommendation for Phase 1

`[TODO]` Based on VRAM analysis, what GPU configuration is needed for Phase 1.

---

## Appendix

### A. Environment & Versions

| Package | Version |
|---------|---------|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| FastGen | 0.1.0 |
| Diffusers | 0.35.1 |
| Transformers | 4.49.0 |

### B. Server Setup

| Item | Value |
|------|-------|
| GPU | 8x NVIDIA RTX 5090 32GB |
| CPU | 384 cores |
| RAM | 1TB |
| OS | Ubuntu, Linux 5.15.0 |

### C. Pretrained Checkpoints Used

| Model | Source | Path on Server |
|-------|--------|---------------|
| Wan2.1-T2V-1.3B (Diffusers) | HuggingFace | `/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/` |
| Wan2.1-T2V-1.3B (Original) | HuggingFace | `/data/chenqingzhan/CausVid/wan_models/Wan2.1-T2V-1.3B/` |
| CausVid DMD checkpoint | `tianweiy/CausVid` | `/data/chenqingzhan/causvid_checkpoints/bidirectional_checkpoint2/model.pt` |
| ECT (self-trained, 6000 iter) | Self-trained | `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_ct/ect_wan1.3b_4gpu/checkpoints/0006000` |
| CD (partial, 1500 iter) | Self-trained | `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_cd/cd_wan1.3b_2gpu/checkpoints/0001500` |

### D. References

- [FastGen GitHub](https://github.com/NVlabs/FastGen) — NVIDIA's distillation framework
- [CausVid](https://github.com/tianweiy/CausVid) — DMD-based video distillation (CVPR 2025)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — Base video generation model
- [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) — Public video-text dataset
- Song & Dhariwal 2023 — "Improved Techniques for Training Consistency Models" (ECT)
- Song et al. 2023 — "Consistency Models" (CD)
- Yin et al. 2024 — "Improved Distribution Matching Distillation" (DMD2)
