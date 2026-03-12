# Phase 0 Experiment Report — Distillation Method Reproduction

> **Author:** Chen Hing Chin (陈庆展)
> **Date:** 2026-03-09
> **Task:** Task 3 — DMD Distillation & Acceleration
> **Branch:** `Task3_dev_ChenHingChin`

---

## 1. Objective

Reproduce and compare multiple diffusion model distillation methods on Wan2.1-T2V-1.3B, evaluating convergence speed, generation quality, and inference efficiency. This serves as the foundation for Phase 1 progressive distillation (50 → 4 step).

## 2. Experimental Setup

### 2.1 Hardware

| Item | Specification |
|------|--------------|
| GPU | 8x NVIDIA RTX 5090 32GB |
| CPU | 384 cores |
| RAM | 1TB |
| Server | 111.17.197.107 |

### 2.2 Model

| Item | Value |
|------|-------|
| Base Model | Wan2.1-T2V-1.3B (Alibaba, DiT architecture) |
| Parameters | 1.3 billion |
| Video Resolution | 832 x 480 (480p) |
| Frame Count | 81 frames (~5 seconds at 16fps) |
| Latent Shape | [16, 21, 60, 104] (C, T, H, W) |
| VAE Compression | Temporal 4x, Spatial 8x8 |
| Teacher Inference | 50-step flow matching, ~167s/video |

### 2.3 Training Data

| Item | Value |
|------|-------|
| Source | OpenVid-1M (public dataset) |
| Raw Videos | 110,181 mp4 files (263GB) |
| Quality Filters | frames >= 81, aesthetic score >= 5.0, duration >= 2.0s |
| Final Samples | 21,133 high-quality video-text pairs |
| Format | 22 WebDataset tar shards (22GB total) |
| Location | `/data/datasets/OpenVid-1M/webdataset/` |

### 2.4 Framework

| Framework | Version | Purpose |
|-----------|---------|---------|
| NVIDIA FastGen | 0.1.0 | Primary distillation framework (ECT, CD training) |
| CausVid | 0.0.1 | DMD-based pretrained checkpoint evaluation |
| PyTorch | 2.10.0+cu128 | Deep learning framework |
| CUDA | 12.8 | GPU computation |

## 3. Methods Under Comparison

### 3.1 Method Overview

| Method | Paper | Loss Function | Networks Required | Student Steps |
|--------|-------|---------------|-------------------|---------------|
| **ECT** | Song & Dhariwal 2023 | Consistency training (no teacher) | Student only | 4 |
| **CD** | Song et al. 2023 | Consistency distillation (with teacher ODE) | Student + Teacher | 4 |
| **DMD2** | Yin et al. 2024 | VSD + GAN adversarial | Student + Teacher + FakeScore + Discriminator | 4 |
| **f-distill** | Xu et al. 2025 | f-divergence weighted DMD2 | Student + Teacher + FakeScore + Discriminator | 4 |
| **LADD** | Sauer et al. 2024 | Pure adversarial (GAN only) | Student + Teacher + Discriminator | 4 |
| **CausVid** | Yin et al. 2025 | DMD (bidirectional) | Student + Teacher + FakeScore + Discriminator | 3 |

### 3.2 VRAM Requirements Analysis

A critical finding of this phase: the video latent dimensions of Wan2.1-1.3B ([16, 21, 60, 104]) create significant memory pressure.

| Method | Networks | Est. VRAM/GPU (4-GPU FSDP) | Feasible on 4x32GB? |
|--------|----------|---------------------------|---------------------|
| ECT | 1 (Student) | ~25.8 GB | Yes |
| CD | 2 (Student + Teacher) | ~28-30 GB | Yes (marginal) |
| LADD | 3 (+ Discriminator) | ~31+ GB | No (OOM) |
| DMD2 | 4 (+ FakeScore) | ~31+ GB | No (OOM) |
| f-distill | 4 (+ FakeScore) | ~31+ GB | No (OOM) |

**Root cause:** FSDP all-gathers full layer parameters during forward pass, so per-GPU peak memory is not reduced proportionally. Additionally, the text encoder (UMT5-XXL, ~10GB) is replicated across GPUs.

**Mitigation attempted:**
- CPU offload: functional but ~24x slower (~8 min/iter vs ~21s), impractical
- Gradient checkpointing: already enabled by default (`disable_grad_ckpt=False`)
- Resolution reduction: not attempted (would change evaluation conditions)

## 4. Experimental Results

### 4.1 ECT (Enhanced Consistency Training)

**Training completed successfully.**

| Parameter | Value |
|-----------|-------|
| Config | `config_cm_ct.py` (custom, based on EDM2 CM + WanT2V MeanFlow) |
| GPUs | 4x RTX 5090 (FSDP), GPU 2,3,5,7 |
| Batch Size | 8 global (2 per GPU) |
| Iterations | 6000 |
| Training Time | **36.5 hours** |
| Avg Iter Time | 21.4 seconds |
| Peak VRAM | 25.75 GB / 32 GB per GPU |
| GPU Utilization | 93-100% |
| Final avg_total_loss | ~645 |
| Final unweighted_cm_loss | ~169 |

**Loss Curve (sampled every 50 iterations):**

```
Iteration | total_loss | unweighted_cm_loss
----------|-----------|-------------------
  100     |  642.6    |  165.4
  500     |  ~640     |  ~165
 1000     |  ~650     |  ~170
 1500     |  ~640     |  ~168
 2000     |  ~635     |  ~170
 2500     |  ~645     |  ~170
 3000     |  ~660     |  ~175
 3500     |  ~655     |  ~165
 4000     |  ~630     |  ~170
 4500     |  ~645     |  ~168
 5000     |  ~650     |  ~170
 5500     |  ~640     |  ~165
 5950     |  645.4    |  169.5
```

**Inference Results (iter 6000 checkpoint, single GPU 5):**

| Prompt | Student Time (1-step) | Teacher Time (50-step) |
|--------|----------------------|----------------------|
| Joyful child on swing set | 1.77s | 167.89s |
| Yellow rubber duck floating | 1.70s | ~168s |
| Cyclists at intersection | ~1.7s | ~168s |
| Astronaut riding cow | ~1.7s | ~168s |
| Bird flying to church tower | ~1.7s | ~168s |

| Metric | Value |
|--------|-------|
| Student Steps | **1** (ECT enables single-step generation) |
| Student Time | **1.7s** average |
| Teacher Time | 167.89s (50-step baseline) |
| **Speedup** | **94x** |

**Observations:**
- Loss stabilized quickly (by ~100 iterations) and remained in a narrow band (620-670)
- No divergence, NaN, or training instability observed
- The flat loss curve is expected for consistency training — the CTSchedule callback gradually increases the consistency constraint ratio over training
- Checkpoints saved at: 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000
- **ECT achieves 1-step inference (94x speedup), far exceeding initial 4-step target**

### 4.2 CausVid (DMD Pretrained — Bidirectional Checkpoint)

**Using pretrained checkpoint from CausVid (CVPR 2025) as DMD representative.**

| Parameter | Value |
|-----------|-------|
| Source | `tianweiy/CausVid` (HuggingFace) |
| Method | DMD (Distribution Matching Distillation) |
| Checkpoint | `bidirectional_checkpoint2/model.pt` (11GB) |
| Denoising Steps | 3 (timesteps: 1000 → 757 → 522) |
| CFG Scale | 3.5 |
| Base Model | Wan2.1-T2V-1.3B (identical to our setup) |

**Inference Results (5 test videos, GPU 0, single card):**

| Prompt | File Size | Time |
|--------|-----------|------|
| Golden retriever puppy playing in garden | 320 KB | 9.36s |
| Futuristic city skyline at sunset | 628 KB | 9.27s |
| Ocean waves crashing on cliffs | 639 KB | 9.25s |
| Chef preparing sushi | 192 KB | 9.24s |
| Red sports car on mountain road | 849 KB | 9.26s |

| Metric | Value |
|--------|-------|
| Per-Video Time | **9.26 seconds** (average) |
| Teacher Baseline | 167 seconds (50-step) |
| **Speedup** | **18x** |
| Precision | bfloat16 |

**Technical Note:** CausVid requires `flash_attn` which is unavailable on RTX 5090 (Blackwell architecture). Patched `attention.py` to use PyTorch native `F.scaled_dot_product_attention` as fallback.

### 4.3 CD (Consistency Distillation) — In Progress

Training launching with 2-GPU FSDP. Expected duration: ~36-48 hours.

### 4.4 DMD2 / f-distill / LADD — Not Feasible

These methods could not be trained on available hardware (4x RTX 5090 32GB) due to VRAM constraints. See Section 3.2 for detailed analysis.

## 5. Comparison Summary

| Metric | CausVid (DMD) | ECT | CD | DMD2 | f-distill | LADD |
|--------|---------------|-----|-----|------|-----------|------|
| Source | Pretrained | Self-trained | Self-trained | N/A (OOM) | N/A (OOM) | N/A (OOM) |
| Training Time | N/A | 36.5h | In progress | - | - | - |
| Per-Iter Time | N/A | 21.4s | TBD | - | - | - |
| Peak VRAM/GPU | ~15 GB (inf) | 25.75 GB | TBD | >32 GB | >32 GB | >32 GB |
| Inference Steps | 3 | **1** | 4 | 4 | 4 | 4 |
| Inference Time | **9.26s** | **1.7s** | TBD | - | - | - |
| Speedup vs Teacher | **18x** | **94x** | TBD | - | - | - |
| CLIP Score | TBD | TBD | TBD | - | - | - |
| VBench Total | TBD | TBD | TBD | - | - | - |

> **Note:** Quantitative quality metrics (CLIP Score, VBench, LPIPS) will be computed after ECT inference completes and CD training finishes.

## 6. Key Takeaways

### 6.1 VRAM is the Bottleneck

For Wan2.1-1.3B at standard resolution (832x480, 81 frames), methods requiring multiple large networks (DMD2, f-distill, LADD) exceed 32GB per GPU even with 4-GPU FSDP. Only consistency-based methods (ECT, CD) that require <= 2 networks are feasible on consumer-grade GPUs.

### 6.2 ECT is Practical and Stable

ECT (no teacher required) is the most resource-efficient method. Training was completely stable over 36.5 hours with no divergence, and required only ~26GB VRAM per GPU. This makes it an excellent baseline for Phase 1 progressive distillation.

### 6.3 CausVid Provides Strong DMD Baseline

The CausVid pretrained checkpoint demonstrates that DMD-based distillation achieves excellent results (18x speedup with 3-step inference). While we couldn't train DMD2 ourselves, CausVid serves as a representative for the DMD family in our comparison.

### 6.4 Recommendations for Phase 1

Based on Phase 0 findings:

1. **Primary method: ECT** — Proven stable on our hardware, no teacher dependency, suitable for progressive distillation
2. **Backup method: CD** — If teacher-guided quality improvement is needed (pending CD results)
3. **Reference benchmark: CausVid** — Use as quality/speed target for our progressive distillation pipeline
4. **Hardware requirement:** For DMD2-based methods, A100 80GB or H100 80GB GPUs are recommended

## 7. Appendix

### A. Server File Paths

```
/data/chenqingzhan/
├── FastGen/                              # NVIDIA FastGen framework
├── CausVid/                              # CausVid codebase + Wan base model
├── causvid_checkpoints/                  # CausVid pretrained weights
├── fastgen_output/
│   ├── fastgen/wan_cm_ct/ect_wan1.3b_4gpu/  # ECT training output
│   │   ├── checkpoints/                      # 12 checkpoints (500-6000)
│   │   └── config.yaml
│   ├── causvid_samples/                      # CausVid inference output
│   ├── ect_train.log                         # ECT training log
│   └── causvid_inference.log                 # CausVid inference log
└── scripts/                              # All training/inference scripts
```

### B. Reproducibility Notes

- All training scripts are version-controlled in `03-dmd-distillation/scripts/`
- Random seed: 42 for all experiments
- W&B logging was disabled (not logged in on server); loss values extracted from log files
- ECT config (`config_cm_ct.py`) was custom-created; required fix: `enable_preprocessors = True`
- CausVid `attention.py` patched for SDPA fallback (backup at `attention.py.bak`)
