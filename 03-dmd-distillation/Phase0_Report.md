# Phase 0 Report — FastGen Video Networks: Architecture, Inference & Capability Boundaries

> **Author:** Chen Hing Chin (陈庆展)
> **Date:** 2026-03-12
> **Task:** Task 3 — DMD Distillation & Acceleration
> **Branch:** `Task3_dev_ChenHingChin`
> **Status:** Complete

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
  ├── CausVid (DMD pretrained, 3-step)
  ├── rCM (NVlabs pretrained, 4-step Consistency Model)
  ├── TurboDiffusion (rCM + quantization, attempted)
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
| [CausVid](https://github.com/tianweiy/CausVid) (CVPR 2025) | DMD pretrained checkpoint for inference comparison |
| [rCM](https://github.com/NVlabs/rcm) (NVlabs, ICLR 2026) | Consistency Model pretrained checkpoint |
| [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) (THU) | rCM + quantization optimization (attempted) |
| OpenVid-1M | Public video-text dataset for data pipeline testing |
| Wan2.1-T2V-1.3B | Base video generation model (DiT architecture) |

---

## 1. FastGen Framework Architecture

### 1.1 Overview

FastGen is NVIDIA's unified framework for **diffusion model distillation** — compressing many-step diffusion models into few-step student models while preserving generation quality.

**Design philosophy:**
- **Modular methods:** Each distillation algorithm (DMD2, ECT, CD, etc.) is a self-contained module implementing a common `single_train_step()` interface
- **Config composition:** Hierarchical OmegaConf DictConfig system that separates model, method, data, and training concerns
- **Callback-driven:** Logging, checkpointing, EMA, and scheduling are implemented as composable callbacks
- **Multi-backend:** Supports FSDP and DDP distributed training out of the box

**Supported scope:**
- **Models:** Wan2.1 (1.3B / 14B), EDM2 (image), SDXL variants
- **Methods:** 6+ distillation methods across two families (consistency-based, distribution matching)
- **Data:** WebDataset-based video/image loading with VAE encoding, or pre-computed latents

### 1.2 Codebase Structure

```
FastGen/
├── train.py                            # Training entry point
├── scripts/inference/
│   └── video_model_inference.py        # Inference entry point
│
├── fastgen/
│   ├── configs/                        # === Config Layer ===
│   │   ├── base.py                     # Base config classes (OmegaConf DictConfig)
│   │   ├── data.py                     # VideoLoaderConfig / VideoLatentLoaderConfig
│   │   ├── net.py                      # Model configs (Wan_1_3B_Config, Wan_14B_Config)
│   │   ├── methods/                    # Base method configs
│   │   │   ├── config_cm.py            # Consistency model base config
│   │   │   └── config_dmd2.py          # DMD2 base config
│   │   └── experiments/WanT2V/         # Experiment-specific configs (compose base configs)
│   │       ├── config_dmd2.py          # DMD2 on Wan2.1-1.3B
│   │       ├── config_fdistill.py      # f-distill on Wan2.1
│   │       ├── config_ladd.py          # LADD on Wan2.1
│   │       ├── config_mf.py            # MeanFlow on Wan2.1
│   │       ├── config_cm_ct.py         # ECT on Wan2.1 (CUSTOM, created for this study)
│   │       └── config_cm_cd.py         # CD on Wan2.1 (CUSTOM, created for this study)
│   │
│   ├── methods/                        # === Method Layer ===
│   │   ├── consistency_model/
│   │   │   ├── CM.py                   # ECT/CD implementation (use_cd=False/True)
│   │   │   └── mean_flow.py           # MeanFlow method
│   │   └── distribution_matching/
│   │       ├── dmd2.py                 # DMD2: VSD + GAN loss
│   │       ├── f_distill.py            # f-divergence weighted DMD2
│   │       └── ladd.py                 # LADD: pure adversarial (GAN only)
│   │
│   ├── datasets/                       # === Data Layer ===
│   │   ├── wds_dataloaders.py          # WebDataset loader (tar shards → batches)
│   │   └── decoders.py                 # Video decoding (mp4 → tensor)
│   │
│   ├── trainer.py                      # === Training Layer ===
│   │                                   # Training loop with FSDP/DDP, gradient accumulation
│   │
│   ├── callbacks/                      # === Callback Layer ===
│   │   ├── ema.py                      # EMA (Exponential Moving Average) for student weights
│   │   ├── ct_schedule.py              # CTSchedule: curriculum control for consistency training
│   │   ├── wandb.py                    # W&B logging (loss, metrics, video samples)
│   │   ├── train_profiler.py           # Per-iteration timing profiler
│   │   └── gpu_stats.py               # GPU memory & utilization monitor
│   │
│   └── utils/                          # === Utility Layer ===
│       ├── checkpointer.py             # FSDP-aware checkpoint save/load
│       ├── lr_scheduler.py             # LambdaInverseSquareRootScheduler
│       └── lazy_call.py                # LazyCall (L) for deferred config instantiation
```

### 1.3 Core Abstractions

**Config system — hierarchical composition with LazyCall:**

```
base config (configs/methods/config_cm.py)
    ↓ extends & overrides
experiment config (configs/experiments/WanT2V/config_cm_ct.py)
    ↓ CLI overrides at runtime
python train.py --config=... - trainer.fsdp=True model.net.model_id_or_local_path=...
```

LazyCall (`L`) enables deferred instantiation — configs describe *what* to create without creating it:

```python
from fastgen.utils import LazyCall as L
from fastgen.callbacks.ema import EMACallback

config.trainer.callbacks.ema_1 = L(EMACallback)(
    type="power", gamma=96.99, ema_name="ema_1"
)
# EMACallback is NOT instantiated here — only when Trainer builds callbacks
```

**Method interface — `single_train_step()`:**

Every distillation method implements one core function:
```python
class CM(BaseMethod):
    def single_train_step(self, batch, state):
        # 1. Unpack batch (latents, text embeddings)
        # 2. Sample timestep, add noise
        # 3. Compute method-specific loss
        # 4. Return loss dict
```

**Trainer — orchestrates the training loop:**
```
Trainer.train()
  → for iter in range(max_iter):
      → batch = next(dataloader)
      → loss = method.single_train_step(batch, state)
      → loss.backward()  (FSDP handles gradient sharding)
      → optimizer.step()
      → for callback in callbacks: callback.on_training_step_end()
```

**Callbacks — modular, composable hooks:**

| Callback | Trigger | Purpose |
|----------|---------|---------|
| EMACallback | Every step | Update EMA copy of student weights |
| CTScheduleCallback | Every step | Adjust consistency ratio curriculum |
| WandBCallback | Every `logging_iter` | Log losses, generate sample videos |
| CheckpointCallback | Every `save_ckpt_iter` | Save model checkpoint |

---

## 2. Video Networks: Wan2.1 Architecture

### 2.1 Wan2.1 DiT (Diffusion Transformer)

Wan2.1 is Alibaba's video generation model based on the **DiT (Diffusion Transformer)** architecture:

- **3D Attention:** Joint spatial-temporal attention — each latent frame token attends to all spatial and temporal positions
- **Flow Matching:** Uses continuous-time flow matching noise schedule (not DDPM discrete steps)
- **Conditioning:** Text embeddings from UMT5-XXL injected via cross-attention and adaptive LayerNorm
- **Model variants:** 1.3B (lightweight, used in this study) and 14B (full-scale, used by NVIDIA for benchmarks)

### 2.2 Components

| Component | Architecture | Parameters | Purpose |
|-----------|-------------|------------|---------|
| **Transformer (DiT)** | 3D Diffusion Transformer | 1.3B / 14B | Core denoising network with spatial-temporal attention |
| **VAE** | 3D Causal VAE | ~100M | Video ↔ Latent compression (temporal 4x, spatial 8x8) |
| **Text Encoder** | UMT5-XXL | ~4.7B | Text prompt → embedding vectors |
| **Noise Scheduler** | Flow Matching | - | Continuous-time noise schedule for training & sampling |

### 2.3 Latent Space

| Parameter | Value |
|-----------|-------|
| Input video | 832 x 480, 81 frames |
| Latent shape | `[16, 21, 60, 104]` (C, T, H, W) |
| Temporal compression | 4x (81 frames → 21 latent frames) |
| Spatial compression | 8x8 (832x480 → 104x60) |
| Per-sample latent size | ~5.3 MB (bf16) |

**VRAM impact of latent dimensions:**

The large latent shape `[16, 21, 60, 104]` is the root cause of VRAM pressure. During a forward pass, each network must process these high-dimensional tensors. Methods requiring multiple networks (DMD2: 4 networks, LADD: 3) multiply this memory cost, causing OOM on 32GB GPUs. The text encoder (UMT5-XXL, ~10GB) further compounds the issue as it is replicated per GPU in FSDP.

---

## 3. Distillation Methods in FastGen

### 3.1 Method Taxonomy

FastGen organizes distillation methods into two families based on their loss formulation:

```
Distillation Methods in FastGen
│
├── Consistency-Based (learn self-consistency along ODE trajectories)
│   ├── ECT  — Enhanced Consistency Training (no teacher needed)
│   ├── CD   — Consistency Distillation (teacher provides ODE targets)
│   └── MeanFlow — Pre-computed latents, mean flow matching
│
└── Distribution Matching (adversarial / score matching)
    ├── DMD2      — VSD loss + GAN loss + FakeScore network
    ├── f-distill — f-divergence weighted DMD2 variant
    └── LADD      — Pure adversarial (GAN only, no VSD/FakeScore)
```

### 3.2 Method Comparison (Conceptual)

**ECT (Enhanced Consistency Training)**
- **Core idea:** Train the student to produce *consistent* outputs at different noise levels — denoising from any point on the ODE trajectory should converge to the same clean output. No teacher model is needed.
- **Loss:** `L = w(t,r) * ||student(y_t, t) - sg[student(y_r, r)]||^2` where `sg` = stop-gradient
- **Networks:** Student only
- **Tradeoff:** Lowest VRAM, simplest to train; may sacrifice quality without teacher guidance

**CD (Consistency Distillation)**
- **Core idea:** Same consistency constraint as ECT, but the target `y_r` is computed by running the *teacher* model's ODE solver from `t` to `r`, providing higher-quality supervision.
- **Loss:** `L = w(t,r) * ||student(y_t, t) - sg[student(y_r_teacher, r)]||^2`
- **Networks:** Student + Teacher
- **Tradeoff:** Better quality than ECT (teacher-guided); higher VRAM due to teacher in memory

**DMD2 (Distribution Matching Distillation v2)**
- **Core idea:** Match the *distribution* of student outputs to the teacher's output distribution using two complementary losses: VSD (Variational Score Distillation) aligns score functions, and GAN loss adds adversarial refinement via a discriminator.
- **Loss:** `L = L_VSD + lambda * L_GAN`, with FakeScore network for VSD computation
- **Networks:** Student + Teacher + FakeScore + Discriminator (4 networks)
- **Tradeoff:** Highest quality (NVIDIA's primary method); highest VRAM and complexity

**f-distill (f-divergence weighted DMD2)**
- **Core idea:** Variant of DMD2 that uses f-divergence weighting to rebalance the distribution matching loss, emphasizing regions where student and teacher distributions diverge most.
- **Networks:** Same as DMD2 (4 networks)
- **Tradeoff:** Potentially better convergence than DMD2; same VRAM requirements

**LADD (Latent Adversarial Distillation)**
- **Core idea:** Pure adversarial approach — train the student using only GAN loss against a discriminator, without score matching (VSD) or FakeScore network.
- **Networks:** Student + Teacher + Discriminator (3 networks)
- **Tradeoff:** Simpler than DMD2 but still needs teacher; faster per-iteration but may have GAN training instability

**MeanFlow**
- **Core idea:** Flow matching in the consistency model family. Uses pre-computed VAE latents and text embeddings (different data format from the other methods).
- **Networks:** Student only
- **Data:** Requires `VideoLatentLoaderConfig` with pre-computed `.pth` files, not raw mp4

### 3.3 How Methods Plug Into FastGen

Every method implements the `single_train_step()` interface. Here is the ECT flow in `CM.py`:

```python
# CM.py — single_train_step() for ECT (use_cd=False):

def single_train_step(self, batch, state):
    data = batch["latents"]       # VAE-encoded video latent [B, 16, 21, 60, 104]
    eps = torch.randn_like(data)  # Random noise

    # 1. Sample timestep t from logitnormal distribution
    t = sample_time(dist_type="logitnormal", p_mean=-0.8, p_std=1.6)

    # 2. Add noise at timestep t
    y_t = noise_scheduler.forward_process(data, eps, t)

    # 3. Compute target timestep r via sigmoid mapping (ECT paper)
    #    r controls how far apart the two points on the ODE trajectory are
    #    CTSchedule callback gradually increases ratio over training
    r = t - t * (1 - ratio) * (1 + 8 * sigmoid(-t))

    # 4. ECT target: add noise at timestep r (no teacher needed)
    y_r = noise_scheduler.forward_process(data, eps, r)

    # 5. Consistency loss: student outputs at (y_t, t) and (y_r, r) should match
    pred_t = student(y_t, t)
    pred_r = student(y_r, r).detach()  # stop-gradient on target
    loss = huber_loss(pred_t, pred_r) * weight(t, r)

    # 6. CTSchedule callback updates ratio after each step
    return {"cm_loss": loss}
```

For CD (`use_cd=True`), step 4 changes: instead of `forward_process(data, eps, r)`, the teacher runs an ODE step from `y_t` at time `t` to produce the target at time `r`:
```python
    # CD target: teacher ODE solver (t → r)
    y_r = ode_solver(teacher, y_t, t, r, guidance_scale=5.0)
```

For DMD2, the `single_train_step()` alternates between:
1. **Student update:** Compute VSD loss (via FakeScore) + GAN generator loss → update student
2. **Discriminator update:** Compute GAN discriminator loss → update discriminator

---

## 4. Inference Experiments

### 4.1 Experimental Setup

| Item | Value |
|------|-------|
| GPU | Single RTX 5090 32GB (GPU index 5) |
| Evaluation Prompts | 5 standardized prompts (see Section 4.2) |
| Precision | bfloat16 |
| Seed | 42 |
| Resolution | 832×480 (480p), 81 frames |
| Models Compared | Teacher (50-step), CausVid DMD (3-step), rCM (4-step) |
| Date | 2026-03-12 |

### 4.2 Evaluation Prompts

All models were evaluated on the same 5 prompts to ensure fair comparison:

| # | Prompt |
|---|--------|
| 1 | A golden retriever puppy playing joyfully in a sunny garden with colorful flowers blooming around it |
| 2 | A futuristic city skyline at sunset with flying cars and brilliant neon lights reflecting off glass towers |
| 3 | Ocean waves crashing dramatically on rocky cliffs during a powerful storm with dark clouds overhead |
| 4 | An astronaut in a white spacesuit riding a brown cow through a lush green meadow under blue skies |
| 5 | A red sports car driving fast through a winding mountain road with autumn foliage on both sides |

### 4.3 Results: Teacher Baseline (50-step)

| Metric | Value |
|--------|-------|
| Framework | NVIDIA FastGen |
| Denoising Steps | 50 |
| CFG Scale | 5.0 |
| Per-Video Time | **~167s** (inference only) / **~183s** (including model load + VAE decode + save) |
| Per-Step Time | ~3.36s |
| Total (5 videos) | 919s |
| Quality | Reference standard — coherent motion, good text alignment, high visual quality |

### 4.4 Results: CausVid DMD (3-step, pretrained)

| Metric | Value |
|--------|-------|
| Source | Pretrained checkpoint (`tianweiy/CausVid`, bidirectional DMD) |
| Method Family | Distribution Matching Distillation (DMD) |
| Denoising Steps | 3 |
| Per-Video Time | **~28.5s** |
| Speedup vs Teacher | **6.4x** |
| Total (5 videos) | 142.4s |

**Notes:**
- CausVid uses its own inference pipeline, separate from FastGen
- Required patching `attention.py` for SDPA fallback (no `flash_attn` on RTX 5090 Blackwell)
- The 28.5s/video includes model loading overhead; pure inference is faster
- Qualitative: Good overall quality, occasional minor artifacts in complex scenes

### 4.5 Results: rCM (4-step, NVlabs pretrained)

| Metric | Value |
|--------|-------|
| Source | Pretrained checkpoint (`NVlabs/rcm`, ICLR 2026) |
| Method Family | Consistency Model (reflow Consistency Model) |
| Denoising Steps | 4 |
| Sigma Max | 80 |
| Per-Video Time | **~37.6s** (avg across 5 prompts) |
| Speedup vs Teacher | **4.9x** |
| Total (5 videos) | 188.1s |

**Per-prompt breakdown:**

| Prompt | Time |
|--------|------|
| 1 (Golden retriever) | 40.34s |
| 2 (Futuristic city) | 36.91s |
| 3 (Ocean waves) | 36.54s |
| 4 (Astronaut + cow) | 36.68s |
| 5 (Red sports car) | 37.64s |

**Notes:**
- Required patching `wan2pt1.py` to replace `flash_apply_rotary_emb` with pure PyTorch rotary embedding implementation (RTX 5090 Blackwell compatibility)
- Prompt 1 is slower (40s) due to model warm-up; subsequent prompts stabilize at ~37s
- Qualitative: Good quality with 4 steps; slight blurriness compared to teacher but coherent motion

### 4.6 Results: TurboDiffusion (4-step, attempted)

| Metric | Value |
|--------|-------|
| Source | Pretrained checkpoint (`thu-ml/TurboDiffusion`) |
| Method Family | rCM + INT8 Quantization + Sparse Linear Attention |
| Status | **Failed — custom CUDA ops not compatible with RTX 5090** |

**Failure analysis:**
- TurboDiffusion requires custom CUDA extensions (`ops` module: `FastLayerNorm`, `FastRMSNorm`, `Int8Linear`) for its quantization and optimization pipeline
- These CUDA kernels are compiled for older GPU architectures and do not support RTX 5090's Blackwell SM 12.0
- The `ops` import is at module level in `modify_model.py`, preventing any inference without the compiled extensions
- TurboDiffusion is essentially rCM with inference-time optimizations (quantization + sparse attention), so rCM results serve as its unoptimized baseline

### 4.7 Speed Comparison Summary

| Model | Method | Steps | Time/Video | Speedup | Status |
|-------|--------|-------|-----------|---------|--------|
| **Teacher** (Wan2.1-1.3B) | Baseline | 50 | 183s | 1x | Reference |
| **CausVid** (DMD pretrained) | DMD | 3 | 28.5s | **6.4x** | Good quality |
| **rCM** (NVlabs pretrained) | Consistency Model | 4 | 37.6s | **4.9x** | Good quality |
| **TurboDiffusion** | rCM + Quant | 4 | — | — | CUDA ops incompatible |

**Key observations:**

1. **CausVid (DMD) is faster than rCM despite fewer steps:** CausVid uses 3 steps vs rCM's 4, and CausVid's pipeline appears more optimized for single-GPU inference.

2. **Both distilled models achieve >4x speedup:** Even without flash attention optimizations (falling back to SDPA), both CausVid and rCM deliver meaningful speedup over the 50-step teacher.

3. **Per-step cost varies by architecture:** Teacher ~3.36s/step, rCM ~9.4s/step (but only 4 steps needed). The per-step overhead in rCM comes from the larger sigma range and different sampling strategy.

4. **Model loading dominates for short runs:** For rCM, the first prompt (40.3s) is ~10% slower than subsequent ones (~37s) due to warm-up. In production, this is amortized.

### 4.8 Qualitative Observations

Based on visual inspection of the 15 generated videos (5 prompts × 3 models):

| Aspect | Teacher (50-step) | CausVid (3-step) | rCM (4-step) |
|--------|-------------------|------------------|--------------|
| **Text Alignment** | Excellent — all prompts accurately rendered | Good — captures main subjects, occasional detail loss | Good — captures main subjects |
| **Motion Quality** | Smooth, coherent temporal motion | Slightly less smooth, minor temporal artifacts | Good coherence, slight jitter |
| **Visual Sharpness** | High detail and clarity | Good overall, minor blur in complex regions | Slightly softer than teacher |
| **Color & Lighting** | Rich, natural color palette | Comparable to teacher | Comparable, slightly muted |
| **Artifacts** | Minimal | Occasional edge artifacts | Minor noise patterns |

**Overall ranking (subjective):** Teacher > CausVid ≈ rCM — both distilled models produce watchable, coherent videos with only moderate quality loss compared to the 50-step teacher.

---

## 5. Training Pipeline Analysis

### 5.1 Training Flow (Code Walkthrough)

The complete execution path of a FastGen training run:

```
torchrun --nproc_per_node=N train.py --config=path/to/config.py - key=value ...
│
├── 1. Config Loading
│   ├── Load experiment config (OmegaConf DictConfig)
│   ├── Apply CLI overrides (trainer.fsdp=True, etc.)
│   └── Resolve LazyCall instances
│
├── 2. Model Construction
│   ├── Student network (always): Wan2.1 DiT from model_id_or_local_path
│   ├── Teacher network (if needed): same architecture, frozen weights
│   ├── FakeScore network (DMD2/f-distill): denoising score estimator
│   ├── Discriminator (DMD2/f-distill/LADD): adversarial classifier
│   ├── VAE: 3D Causal VAE encoder/decoder
│   └── Text Encoder: UMT5-XXL
│
├── 3. Dataloader Construction
│   ├── VideoLoaderConfig: reads WebDataset tar shards
│   └── Distributed sampler (for multi-GPU)
│
├── 4. Optimizer & Scheduler
│   ├── AdamW (lr=1e-5 typical)
│   └── LambdaInverseSquareRootScheduler (warm_up + decay)
│
├── 5. Callbacks Registration
│   ├── EMACallback (student weight averaging)
│   ├── CTScheduleCallback (consistency ratio curriculum)
│   ├── WandBCallback (logging)
│   ├── CheckpointCallback (save every N iters)
│   └── GPUStatsCallback (memory monitoring)
│
└── 6. Trainer.train() — Main Loop
    └── for iter in range(max_iter):
        ├── batch = next(dataloader)     # mp4 → decode → VAE encode → latent
        ├── loss = method.single_train_step(batch, state)
        ├── loss.backward()              # FSDP handles gradient sharding
        ├── optimizer.step()
        ├── scheduler.step()
        └── for cb in callbacks: cb.on_training_step_end()
            ├── EMA: update shadow weights
            ├── CTSchedule: adjust ratio
            ├── WandB: log loss (every logging_iter)
            ├── Checkpoint: save (every save_ckpt_iter)
            └── Validation: generate samples (every validation_iter)
```

### 5.2 Data Pipeline

How training data flows from disk to loss computation:

```
WebDataset Shards (.tar files on disk)
│  ├── sample_000000.mp4 (832x480, 81 frames, raw video)
│  └── sample_000000.txt (text caption)
│
├── wds_dataloaders.py
│   ├── webdataset.WebLoader reads tar shards
│   ├── Shuffles across shards for randomization
│   └── Returns (video_bytes, caption_text) pairs
│
├── decoders.py
│   ├── Decodes mp4 bytes → video tensor [B, C, T, H, W]
│   ├── Resizes to target resolution (832x480)
│   └── Normalizes pixel values to [-1, 1]
│
├── VAE Encoder (runs on GPU, within training step)
│   ├── Encodes video [B, 3, 81, 480, 832] → latent [B, 16, 21, 60, 104]
│   └── Temporal 4x compression, spatial 8x8 compression
│
├── Text Encoder (UMT5-XXL)
│   ├── Tokenizes caption → input_ids
│   └── Encodes → text_embedding [B, seq_len, dim]
│
└── Method receives: {latents, text_embeddings, ...}
    └── Computes distillation loss on latent space
```

**Data prepared for this study:**
- **Source:** OpenVid-1M (public dataset, 1M+ video-text pairs)
- **Quality filters:** frames >= 81, aesthetic score >= 5.0, duration >= 2.0s
- **Result:** 21,133 samples in 22 WebDataset tar shards (22GB total)
- **Location:** `/data/datasets/OpenVid-1M/webdataset/shard-{000000..000021}.tar`

### 5.3 Key Configuration Parameters

| Parameter | Description | Typical Value | Impact |
|-----------|-------------|---------------|--------|
| `trainer.batch_size_global` | Total batch size across all GPUs | 8 | Higher = better convergence, more VRAM |
| `trainer.max_iter` | Total training iterations | 6000 | Longer = better quality, more time |
| `trainer.fsdp` | Enable Fully Sharded Data Parallel | True | Required for multi-GPU |
| `trainer.logging_iter` | Log loss every N iters | 50 | Monitoring frequency |
| `trainer.save_ckpt_iter` | Save checkpoint every N iters | 500 | Recovery & evaluation points |
| `trainer.validation_iter` | Generate sample videos every N iters | 500 | Visual quality tracking |
| `model.guidance_scale` | Classifier-Free Guidance scale | 5.0 | Higher = more text-aligned, less diverse |
| `model.num_steps_student` | Student denoising steps | 4 | Target compression level |
| `model.loss_config.use_cd` | Use teacher for consistency distillation | True/False | ECT vs CD switch |
| `model.loss_config.huber_const` | Huber loss transition constant | 0.06 | Loss smoothness |
| `model.sample_t_cfg.time_dist_type` | Timestep sampling distribution | logitnormal | Controls which noise levels are trained |
| `model.sample_t_cfg.train_p_mean` | Logitnormal mean parameter | -0.8 | Biases toward mid-range timesteps |
| `model.use_ema` | EMA callback names | ["ema_1"] | Stabilizes student weights |
| `dataloader_train.datatags` | WebDataset shard paths | ["WDS:/path/..."] | Training data location |

### 5.4 Training Experiments Conducted

During Phase 0, two self-training experiments were conducted to validate understanding of the training pipeline:

**ECT Training (6000 iterations, 4-GPU FSDP):**
- Completed successfully, 36.5 hours wall time
- Loss stabilized around ~650 by iteration 2000
- Checkpoint saved at `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_ct/ect_wan1.3b_4gpu/checkpoints/0006000`
- **Quality assessment:** Very poor — 1-step generation produces blurry, noisy outputs. This is expected for ECT without extensive hyperparameter tuning and longer training

**CD Training (1500/6000 iterations, 2-GPU FSDP):**
- Stopped early at iteration 1500 due to slow convergence and poor intermediate quality
- Checkpoint saved at `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_cd/cd_wan1.3b_2gpu/checkpoints/0001500`
- **Quality assessment:** Very poor — teacher-guided ODE targets did not converge to coherent outputs in 1500 iterations

**Key takeaway:** Self-training consistency models from scratch requires significant compute budget and careful tuning. The pretrained checkpoints (CausVid, rCM) from published papers with full-scale training demonstrate far superior quality, confirming the value of the Phase 0 pivot to pretrained inference comparison.

---

## 6. Capability Boundaries of FastGen

### 6.1 Supported Models

| Model | Type | Parameters | Configs Available |
|-------|------|------------|------------------|
| Wan2.1-T2V-1.3B | Text-to-Video | 1.3B | DMD2, ECT, CD, f-distill, LADD, MeanFlow |
| Wan2.1-T2V-14B | Text-to-Video | 14B | Same methods (requires significantly more VRAM) |
| EDM2 | Image | Various | CM, DMD2 (image-only configs available) |
| SDXL variants | Image | 2.6B | Subset of methods |

> **Note:** FastGen's experiment configs under `experiments/WanT2V/` are specifically for Wan2.1 video models. Image model configs exist under `experiments/EDM2/` and `experiments/SDXL/`.

### 6.2 Supported Distillation Methods

| Method | Image | Video | Wan2.1 Config? | Tested in This Study? | Result |
|--------|-------|-------|---------------|----------------------|--------|
| DMD2 | Yes | Yes | Built-in | Attempted | OOM on 4x32GB |
| f-distill | Yes | Yes | Built-in | Attempted | OOM on 4x32GB |
| LADD | Yes | Yes | Built-in | Attempted | OOM on 4x32GB |
| MeanFlow | Yes | Yes | Built-in | Not tested | Requires pre-computed latents |
| ECT | Yes | Yes | **Custom** | **Trained successfully** | 6000 iter, 36.5h |
| CD | Yes | Yes | **Custom** | **Partially trained** | 1500/6000 iter |

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

Limitations discovered during this study:

1. **No built-in WanT2V configs for ECT/CD:** FastGen ships with DMD2, f-distill, LADD, and MeanFlow configs for Wan2.1, but consistency model configs (ECT/CD) had to be custom-created by combining EDM2 CM configs with WanT2V model settings.

2. **No flash_attn on RTX 5090 (Blackwell):** The `flash_attn` library (used by CausVid and some FastGen code paths) does not yet support RTX 5090's Blackwell architecture. Required patching attention code to fall back to PyTorch native `F.scaled_dot_product_attention`.

3. **Text encoder replication in FSDP:** UMT5-XXL (~10GB) is replicated across GPUs rather than sharded, significantly reducing available VRAM per GPU.

4. **No automatic CPU offload for networks:** When multiple networks exceed GPU VRAM, manual CPU offload can be attempted but is impractically slow (~8 min/iter vs ~21s normal).

5. **No built-in evaluation metrics:** FastGen does not include VBench, CLIP Score, or FVD computation. These must be set up separately.

6. **W&B dependency for logging:** Loss logging is tightly coupled to Weights & Biases; running with `wandb_mode=disabled` still attempts video encoding, adding ~10s overhead per logging step.

### 6.5 Scalability Analysis

**How VRAM scales with key dimensions (estimated for Wan2.1 DiT):**

| Dimension | Change | VRAM Impact | Feasibility on 4x32GB |
|-----------|--------|-------------|----------------------|
| Resolution | 480p → 720p | ~2.3x latent size increase | ECT marginal, CD/DMD2 impossible |
| Resolution | 480p → 1080p | ~5x latent size increase | All methods impossible |
| Frame count | 81 → 161 frames | ~2x temporal dimension | ECT marginal, others impossible |
| Model size | 1.3B → 14B | ~10x parameter increase | All methods require 80GB+ GPUs |
| Batch size | 1 → 2 per GPU | ~1.5x VRAM increase | Only ECT at 480p |

**Key insight:** For Phase 1 with the 14B Teacher Model from Task 2, **A100 80GB or H100 80GB GPUs will be required** for any distillation method. Even ECT (the lightest method) would need at least 2x80GB GPUs for the 14B model.

---

## 7. Summary & Phase 1 Recommendations

### 7.1 Key Findings

1. **FastGen is a well-structured, modular framework:** Clean separation of config, methods, training, and inference. New methods can be added by implementing `single_train_step()` and creating a config file. The callback system enables flexible logging, checkpointing, and scheduling.

2. **Pretrained distilled models deliver significant speedup with acceptable quality:**
   - CausVid (DMD, 3-step): **6.4x speedup** (28.5s vs 183s), good visual quality
   - rCM (Consistency Model, 4-step): **4.9x speedup** (37.6s vs 183s), good visual quality
   - Both methods produce coherent, watchable videos — quality loss vs 50-step teacher is moderate and likely acceptable for many applications

3. **VRAM is the primary constraint:** The large latent dimensions of Wan2.1 video models ([16, 21, 60, 104]) combined with multiple network requirements make DMD2/f-distill/LADD infeasible on 32GB GPUs. Only consistency-based methods (ECT, CD) with <= 2 networks fit.

4. **Self-training from scratch is challenging:** ECT and CD trained from scratch for 6000/1500 iterations produced very poor quality, while published pretrained models (CausVid, rCM) with full-scale training achieve excellent results. This highlights the importance of proper hyperparameter tuning, training scale, and potentially better training data.

5. **RTX 5090 compatibility gaps exist:** Flash attention libraries and custom CUDA extensions (e.g., TurboDiffusion's quantization ops) don't support Blackwell architecture (SM 12.0) yet. Pure PyTorch fallbacks work but may sacrifice performance. This needs monitoring for Phase 1.

### 7.2 Method Recommendation for Phase 1

Based on Phase 0 findings, for progressive distillation (50 → 16 → 8 → 4 steps):

| Priority | Method | Rationale |
|----------|--------|-----------|
| **Primary** | DMD (CausVid-style) | Best speedup (6.4x) with good quality in Phase 0 experiments; NVIDIA's primary approach |
| **Secondary** | rCM (Consistency Model) | 4.9x speedup, good quality; NVlabs has pretrained checkpoints for Wan2.1 |
| **Backup** | ECT | Lowest VRAM, no teacher dependency; useful if VRAM is the bottleneck |
| **If 80GB GPUs available** | DMD2 (full FastGen) | NVIDIA's full pipeline with VSD + GAN loss (VBench 84.72) |

### 7.3 Hardware Recommendation for Phase 1

| Scenario | Model | Method | Minimum GPU | Recommended GPU |
|----------|-------|--------|-------------|-----------------|
| ECT with 1.3B | Wan2.1-1.3B | ECT | 4x RTX 5090 32GB | 4x RTX 5090 32GB |
| CD with 1.3B | Wan2.1-1.3B | CD | 4x RTX 5090 32GB | 4x A100 40GB |
| DMD2 with 1.3B | Wan2.1-1.3B | DMD2 | 4x A100 40GB | 4x A100 80GB |
| Any method with 14B | Wan2.1-14B | Any | 8x A100 80GB | 8x H100 80GB |

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
| rCM checkpoint | `NVlabs/rcm` | `/data/chenqingzhan/rcm/assets/checkpoints/rCM_Wan2.1_T2V_1.3B_480p.pt` |
| TurboDiffusion checkpoint | `thu-ml/TurboDiffusion` | `/data/chenqingzhan/TurboDiffusion/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth` |
| ECT (self-trained, 6000 iter) | Self-trained | `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_ct/ect_wan1.3b_4gpu/checkpoints/0006000` |
| CD (partial, 1500 iter) | Self-trained | `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_cd/cd_wan1.3b_2gpu/checkpoints/0001500` |

### D. RTX 5090 Compatibility Patches Applied

| Component | Issue | Fix |
|-----------|-------|-----|
| CausVid `attention.py` | `flash_attn` not available on Blackwell | Patched to use PyTorch `F.scaled_dot_product_attention` (SDPA) fallback |
| rCM `wan2pt1.py` | `flash_apply_rotary_emb` uses unsupported `interleaved` kwarg | Replaced with pure PyTorch rotary embedding (cos/sin interleaved) |
| TurboDiffusion `modify_model.py` | `ops` CUDA extension (INT8 kernels) not compilable for SM 12.0 | **Unsolved** — TurboDiffusion requires custom CUDA ops; inference not possible on RTX 5090 |

### E. References

- [FastGen GitHub](https://github.com/NVlabs/FastGen) — NVIDIA's distillation framework
- [CausVid](https://github.com/tianweiy/CausVid) — DMD-based video distillation (CVPR 2025)
- [rCM](https://github.com/NVlabs/rcm) — Reflow Consistency Model (ICLR 2026)
- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) — rCM + quantization optimization
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — Base video generation model
- [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) — Public video-text dataset
- Song & Dhariwal 2023 — "Improved Techniques for Training Consistency Models" (ECT)
- Song et al. 2023 — "Consistency Models" (CD)
- Yin et al. 2024 — "Improved Distribution Matching Distillation" (DMD2)
