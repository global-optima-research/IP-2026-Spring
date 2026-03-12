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
| 1.5 | 五种蒸馏方案配置 & 脚本部署 | ✅ Done | 2026-03-06 |
| 1.6 | 训练数据准备 (OpenVid-1M → WebDataset) | ✅ Done | 2026-03-07 |
| 1.7 | 训练脚本更新 (数据路径 + 统一参数) | ✅ Done | 2026-03-07 |
| 2 | DMD2 复现 (Wan2.1 1.3B, 4-step) | ⚠️ OOM — using CausVid pretrained | 2026-03-08 |
| 2.1 | CausVid 预训练权重下载 & 推理评估 | ✅ Inference done (3-step, 9.26s/video) | 2026-03-08 |
| 3 | ECT 复现 | ✅ Done — 6000 iter, 36.5h (quality poor, kept as checkpoint) | 2026-03-09 |
| 4 | CD 复现 | ⏹️ Stopped at iter 1500/6000 (quality poor, kept as checkpoint) | 2026-03-10 |
| 4.1 | f-distill / LADD 复现 | ⚠️ OOM — skipped | - |
| 5 | **方案调整：使用预训练模型做推理对比** | ✅ Done | 2026-03-10 |
| 5.1 | 下载 rCM 预训练权重 + 部署推理 | ✅ Done (patched rope_apply for RTX 5090) | 2026-03-12 |
| 5.2 | 下载 TurboDiffusion 预训练权重 + 部署推理 | ❌ Failed (CUDA ops incompatible with SM 12.0) | 2026-03-12 |
| 5.3 | 三模型统一 prompt 推理对比 (Teacher + CausVid + rCM) | ✅ Done (15 videos, 5 per model) | 2026-03-12 |
| 5.4 | Phase0 Report 完成 | ✅ Done (all sections filled, committed & pushed) | 2026-03-12 |
| 5.5 | 脚本整理 + 文档编写 | ✅ Done (scripts reorganized into 5 subdirs + README) | 2026-03-12 |

### Phase 0 Strategy Pivot (2026-03-10)

**Mentor feedback:** Phase 0 的目标是"熟悉 FastGen 框架 + 形成 report + 探索能力边界"，不需要从零复现训练。自训练的 ECT/CD 质量很差，应改用公开预训练模型做推理对比。

**New comparison plan — 4 pretrained models:**

| # | Model | Method | Steps | Source |
|---|-------|--------|-------|--------|
| 1 | Teacher (Wan2.1-1.3B) | Baseline (no distillation) | 50 | Already on server |
| 2 | CausVid | DMD (Distribution Matching) | 3 | Already on server |
| 3 | rCM | Consistency Model (rCM, ICLR 2026) | 2-4 | Download from HF: `worstcoder/rcm-Wan` |
| 4 | TurboDiffusion | rCM + Quantization | 4 | Download from HF: `TurboDiffusion/TurboWan2.1-T2V-1.3B-480P` |

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

### Data Strategy — Decision Made (2026-03-06 Meeting)

**Meeting conclusion: Use OpenVid-1M as the unified public dataset for Phase 0.**

All team members (4 people) will use the same dataset for reproducibility and fair comparison.

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| A: Use VidProM prompts + 1.3B Teacher to generate data | Simple, follows NVIDIA's approach | 1.3B quality < 14B, takes time to generate | Phase 1+ backup |
| B: Use project's own product video data + captions | Domain-specific, better for PVTT | Task 1 data Week 8-9 delivery | Phase 1+ |
| **C: OpenVid-1M (public dataset)** | **Large scale, diverse, immediate availability** | Large download | **✅ Phase 0 chosen** |

### Data Preparation Pipeline

**Dataset:** [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) (1M+ open-domain video-text pairs)
**Shared location:** `/data/datasets/OpenVid-1M/` (team-accessible)
**Target:** ~50,000 samples converted to WebDataset format

Scripts (in `03-dmd-distillation/scripts/`):
1. `download_openvid.sh` — Download 7 zip parts (~50K videos) + CSV metadata from hf-mirror.com
2. `convert_to_webdataset.py` — Filter by quality (frames≥81, aesthetic≥5.0, duration≥2s) and pack into tar shards
3. `prepare_training_data.sh` — One-click pipeline (download → convert → set team permissions)

**Output:** `/data/datasets/OpenVid-1M/webdataset/shard-{000000..000021}.tar`
**Config usage:** `dataloader_train.datatags=["WDS:/data/datasets/OpenVid-1M/webdataset/shard-{000000..000021}.tar"]`

### Data Preparation Result (2026-03-07)

| Item | Value |
|------|-------|
| CSV metadata | 1,019,958 records (818MB) |
| Downloaded zips | 7 parts (part0-part6), ~264GB total |
| Extracted videos | 110,181 mp4 files (263GB) |
| Quality filters | frames >= 81, aesthetic >= 5.0, seconds >= 2.0 |
| Filtered out | 624,410 missing + 374,414 below threshold |
| **Final samples** | **21,133** high-quality video-text pairs |
| WebDataset shards | 22 tar files (shard-000000 ~ shard-000021), 22GB |
| Permissions | 755 (team-readable at `/data/datasets/OpenVid-1M/`) |

> **Note:** 21K samples is sufficient for Phase 0 reproduction. NVIDIA used ~50K synthetic samples
> generated by 14B teacher; our 1.3B-sourced OpenVid-1M subset is a reasonable alternative for
> method comparison. Sample count can be increased by relaxing filters if needed.

### Steps
- [x] Decide data strategy → **OpenVid-1M** (2026-03-06 meeting)
- [x] Write download + conversion scripts (2026-03-06)
- [x] Upload scripts to server and start download (2026-03-07)
- [x] Download complete: 7 zips + CSV + extraction all done (2026-03-07)
- [x] Convert to WebDataset shards: 21,133 samples in 22 shards (2026-03-07)
- [x] Fix bug: CSV column `aesthetic score` (space) vs code `aesthetic_score` (underscore)
- [x] Update all 5 training scripts with correct data paths (2026-03-07)
- [x] Attempted DMD2 training: single GPU OOM (31GB needed, 32GB limit) (2026-03-07)
- [x] Attempted DMD2 training: 4-GPU FSDP still OOM (each GPU ~31GB due to all-gather) (2026-03-07)
- [x] Attempted CPU offload: too slow (~8 min/iter vs ~20s normal) — abandoned (2026-03-07)
- [x] Root cause: DMD2 requires 4 networks simultaneously (Student+Teacher+FakeScore+Discriminator)
- [x] Alternative: Downloaded CausVid pretrained DMD checkpoint for evaluation (2026-03-08)
- [x] Run CausVid inference: 5 videos, 9.26s/video, 18x speedup (2026-03-08)
- [ ] Compare with ECT/CD trained models (pending CD completion + quantitative eval)

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

### ECT Training Status (2026-03-09) — ✅ COMPLETE

**Training completed** on 4-GPU FSDP (GPU 2,3,5,7). Duration: 2026-03-07 22:40 ~ 2026-03-09 11:08 (~36.5 hours).

| Item | Value |
|------|-------|
| Config fix | `enable_preprocessors = False` → `True` (text encoder was disabled) |
| GPUs | 4x RTX 5090 (CUDA 2,3,5,7), FSDP |
| Batch size | 8 global (2 per GPU) |
| Total iterations | 6000 / 6000 (100%) |
| Training time | **36.5 hours** |
| Avg iter time | 21.4 seconds |
| Peak GPU memory | 25.75 GB / 32 GB per GPU |
| GPU utilization | 93-100% |
| Final avg_total_loss | ~645 |
| Final unweighted_cm_loss | ~169 |
| Checkpoints saved | 12 (iter 500 ~ 6000, every 500) |
| Log file | `/data/chenqingzhan/fastgen_output/ect_train.log` |

**Loss trend (sampled every 50 iters):**
```
iter  100: total=642.6, unweighted=165.4
iter  500: total=~640,  unweighted=~165
iter 1000: total=~650,  unweighted=~170
iter 2000: total=~635,  unweighted=~170
iter 3000: total=~660,  unweighted=~175
iter 4000: total=~630,  unweighted=~170
iter 5000: total=~650,  unweighted=~170
iter 5950: total=645.4, unweighted=169.5
```

**Observations:**
- Loss stabilized quickly (~100 iters) and remained in narrow band (620-670)
- No divergence, NaN, or training instability
- Flat loss curve is expected — CTSchedule gradually increases consistency constraint ratio

### ECT Inference Results (2026-03-09) — ✅ COMPLETE

**Inference using iter 6000 checkpoint, single GPU 5.**

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

**Output path:** `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_ct/ect_inference_6000/ect_wan1.3b_4gpu/0006000/validation_aug_qwen_2_5_14b_seed42/`

### Setup Status
- [x] Config created: `config_cm_ct.py` → uploaded to `FastGen/fastgen/configs/experiments/WanT2V/`
- [x] Training script: `run_ect_single_gpu.sh` / `launch_ect_now.sh` → uploaded to server
- [x] Prepare training data (WebDataset shards, shared with DMD2)
- [x] Fix bug: `enable_preprocessors = False` → `True` (text encoder must be enabled)
- [x] Launch full ECT training on 4-GPU FSDP (2026-03-07)
- [x] Training completed: 6000 iters, 36.5 hours, loss ~645 (2026-03-09)
- [x] ECT inference: 1-step, 1.7s/video, 94x speedup (2026-03-09)
- [ ] Quantitative evaluation (CLIP Score, VBench)

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

### CD Training Status (2026-03-09) — ⏹️ STOPPED

**Training launched** on 2-GPU FSDP (GPU 5,7), started 2026-03-09 ~13:54.
**Stopped** on 2026-03-10 at iter 1750 (process killed manually). Quality poor, using pretrained models instead.

| Item | Value |
|------|-------|
| GPUs | 2x RTX 5090 (CUDA 5,7), FSDP |
| Batch size | 8 global (4 per GPU) |
| Iterations completed | 1750 / 6000 (29%) |
| Avg iter time | 56 seconds |
| Peak VRAM | 29.1 GB / 32 GB per GPU |
| Final avg_total_loss | ~571 |
| Final unweighted_cm_loss | ~132 |
| Checkpoints saved | 500, 1000, 1500 |
| Log file | `/data/chenqingzhan/fastgen_output/cd_train.log` |
| Output path | `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_cd/cd_wan1.3b_2gpu/` |

### Setup Status
- [x] Config created: `config_cm_cd.py` → uploaded to `FastGen/fastgen/configs/experiments/WanT2V/`
- [x] Training script: `run_cd_train.sh` → uploaded to server
- [x] Training data ready (WebDataset shards, shared with DMD2 and ECT)
- [x] Launch full CD training on 2-GPU FSDP (2026-03-09)
- [x] Training stopped at iter 1750 — quality poor (2026-03-10)
- [x] **Pivoted to using pretrained rCM/TurboDiffusion for comparison**

---

## Task 2.1: CausVid Pretrained DMD Checkpoint (Alternative to DMD2 Training)

### Background

DMD2 training requires 4 networks simultaneously (Student + Teacher + FakeScore + Discriminator),
exceeding 32GB VRAM even with 4-GPU FSDP. Instead of training from scratch, we use pretrained
checkpoints from **CausVid** (CVPR 2025) for DMD-based evaluation.

### CausVid Overview

| Item | Value |
|------|-------|
| Paper | "From Slow Bidirectional to Fast Autoregressive Video Diffusion Models" (CVPR 2025) |
| Authors | Tianwei Yin et al. |
| Method | DMD-based distillation (same family as DMD2) |
| Base model | Wan2.1-T2V-1.3B (identical to our setup) |
| Distillation loss | DMD (`distillation_loss: dmd` in config) |
| Denoising steps | 3-step (timesteps: 1000 → 757 → 522) |
| HuggingFace | `tianweiy/CausVid` |
| GitHub | `tianweiy/CausVid` |

### Downloaded Assets (2026-03-08)

| Asset | Path on Server | Size |
|-------|---------------|------|
| CausVid codebase | `/data/chenqingzhan/CausVid/` | 6.3 MB |
| CausVid package | `causvid 0.0.1` (pip install -e) | installed |
| Bidirectional checkpoint 2 | `/data/chenqingzhan/causvid_checkpoints/bidirectional_checkpoint2/model.pt` | 11 GB |
| Wan2.1-T2V-1.3B (original format) | `/data/chenqingzhan/CausVid/wan_models/Wan2.1-T2V-1.3B/` | 17 GB |

> **Note:** CausVid uses Wan's original model format (.pth files), not HuggingFace Diffusers format.
> A separate download of `Wan-AI/Wan2.1-T2V-1.3B` (original) was required.

### CausVid Config (wan_bidirectional_dmd_from_scratch.yaml)

```yaml
distillation_loss: dmd
denoising_step_list: [1000, 757, 522]    # 3-step denoising
real_guidance_scale: 3.5                   # CFG scale
image_or_video_shape: [1, 21, 16, 60, 104]  # Same latent shape as our setup
gradient_checkpointing: true
mixed_precision: true                      # bfloat16
negative_prompt: '...'                     # Chinese negative prompt
```

### CausVid Inference Command

```bash
cd /data/chenqingzhan/CausVid
CUDA_VISIBLE_DEVICES=0 python minimal_inference/bidirectional_inference.py \
    --config_path configs/wan_bidirectional_dmd_from_scratch.yaml \
    --checkpoint_folder /data/chenqingzhan/causvid_checkpoints/bidirectional_checkpoint2 \
    --output_folder /data/chenqingzhan/fastgen_output/causvid_samples \
    --prompt_file_path <prompt_file>
```

### CausVid Inference Results (2026-03-08)

| Item | Value |
|------|-------|
| Denoising steps | 3 (timesteps: 1000 → 757 → 522) |
| Per-video time | **9.26 seconds** (single RTX 5090) |
| Teacher baseline | 167 seconds (50-step) |
| **Speedup** | **18x** |
| GPU | GPU 0, single card |
| Total time (5 videos) | 46 seconds |
| Precision | bfloat16 |

**Code fix required:** CausVid depends on `flash_attn` which is not available on RTX 5090.
Patched `attention.py` to fall back to PyTorch native `F.scaled_dot_product_attention` (SDPA).
Backup at `attention.py.bak`.

**Generated samples (5 videos):**

| File | Prompt | Size |
|------|--------|------|
| `output_000.mp4` | Golden retriever puppy playing in a sunny garden | 320 KB |
| `output_001.mp4` | Futuristic city skyline at sunset with flying cars | 628 KB |
| `output_002.mp4` | Ocean waves crashing on rocky cliffs during storm | 639 KB |
| `output_003.mp4` | Chef preparing sushi in traditional Japanese kitchen | 192 KB |
| `output_004.mp4` | Red sports car driving through mountain road with autumn foliage | 849 KB |

Local copy: `03-dmd-distillation/results/causvid_samples/`

### Status
- [x] CausVid codebase downloaded (via tarball, GitHub clone failed in China) (2026-03-08)
- [x] CausVid package installed (`pip install -e .`) (2026-03-08)
- [x] Bidirectional checkpoint 2 downloaded from HF mirror (11GB) (2026-03-08)
- [x] Wan2.1-T2V-1.3B original format model downloaded (17GB) (2026-03-08)
- [x] Patched `attention.py` for SDPA fallback (no flash_attn on RTX 5090) (2026-03-08)
- [x] Installed missing deps: `easydict`, `lmdb`, `open_clip_torch`, `decord`, `moviepy` (2026-03-08)
- [x] CausVid inference completed: 5 videos, 9.26s/video, 18x speedup (2026-03-08)
- [ ] Quantitative evaluation (CLIP Score, VBench) — after ECT completes
- [ ] Compare CausVid (DMD 3-step) vs ECT (4-step) vs CD (4-step)

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

## Task 5: Five-Method Training Plan

### 5.1 Training Configuration (Updated for Multi-GPU)

**Actual training setup** (updated 2026-03-08 based on VRAM constraints):

| Parameter | Value | Notes |
|-----------|-------|-------|
| GPU | 4x RTX 5090 32GB (FSDP) | GPU 2,3,5,7 for ECT/CD |
| Model | Wan2.1-T2V-1.3B | 1.3B DiT |
| Precision | bfloat16 | From config defaults |
| Batch size (per GPU) | 2 | Video latent [16,21,60,104] |
| Batch size (global) | 8 | 2 per GPU × 4 GPUs |
| Max iterations | 6,000 | Same as NVIDIA default |
| Logging interval | 50 iters | Loss logged every 50 steps |
| Checkpoint interval | 500 iters | 12 checkpoints total |
| Validation interval | 500 iters | Sample generation every 500 steps |
| Student steps | 4 | Target 4-step generation |
| Data | 21,133 samples | 22 WebDataset shards |
| FSDP | Enabled | `trainer.fsdp=True, trainer.ddp=False` |
| W&B | Disabled | `log_config.wandb_mode=disabled` (not logged in) |

**VRAM constraint summary:**

| Method | Networks | Peak VRAM/GPU | 4×32GB Feasible? |
|--------|----------|---------------|-----------------|
| ECT | Student only | ~25.8 GB | ✅ Running |
| CD | Student + Teacher | ~28-30 GB | ✅ Should fit |
| LADD | Student + Teacher + Discriminator | ~31+ GB | ❌ OOM |
| DMD2 | Student + Teacher + FakeScore + Disc | ~31+ GB | ❌ OOM |
| f-distill | Student + Teacher + FakeScore + Disc | ~31+ GB | ❌ OOM |

**Strategy:** Train ECT & CD ourselves; use CausVid pretrained checkpoint for DMD evaluation.

### 5.2 Training Order & Time Estimates

**Estimated time per training iteration (single RTX 5090, Wan2.1-1.3B):**

Based on teacher inference baseline (3.35s/step, 50-step) and method complexity:

| # | Method | Per-Iter Estimate | Networks in Memory | 6000-Iter Total | Priority |
|---|--------|------------------|--------------------|-----------------|----------|
| 1 | **DMD2** | ~20-30s | Student + Teacher + FakeScore + Discriminator | **~33-50h** | Highest |
| 2 | **ECT** | ~10-15s | Student only (no teacher) | **~17-25h** | High |
| 3 | **CD** | ~15-25s | Student + Teacher | **~25-42h** | High |
| 4 | **f-distill** | ~20-30s | Student + Teacher + FakeScore + Discriminator | **~33-50h** | Medium |
| 5 | **LADD** | ~15-25s | Student + Discriminator (no FakeScore) | **~25-42h** | Medium |

**Per-iteration breakdown:**
- **DMD2/f-distill:** Most complex — alternates student updates (student fwd 4-step + teacher fwd + fake_score fwd + discriminator fwd + backward) and discriminator updates (student fwd + fake_score fwd + discriminator fwd + backward). ~20-30s/iter.
- **ECT:** Simplest — student fwd at t + student fwd at r + consistency loss backward. No teacher in memory. ~10-15s/iter.
- **CD:** Student fwd at t + teacher ODE step (t→r) + consistency loss backward. Teacher adds VRAM but only 1 fwd pass. ~15-25s/iter.
- **LADD:** Student fwd + discriminator fwd + backward. No fake_score network, simpler than DMD2. ~15-25s/iter.

**Total estimate for all 5 methods: ~130-210 hours (~5.5-8.5 days sequential)**

**Recommended execution order:**
1. **DMD2** (first, highest priority — NVIDIA's primary method, must reproduce)
2. **ECT** (fastest, no teacher — good quick baseline)
3. **LADD** (simpler adversarial — compare with DMD2)
4. **f-distill** (DMD2 variant — compare divergence weighting effect)
5. **CD** (with teacher — compare consistency distillation vs training)

> **Note:** Methods can run sequentially on the same GPU. Consider running ECT while
> reviewing DMD2 results. If time is tight, prioritize DMD2 + ECT + one adversarial (LADD or f-distill).

### 5.3 Evaluation Metrics

#### A. Training Convergence Metrics (logged automatically to W&B)

These are logged every `logging_iter=50` steps and vary by method:

| Metric | DMD2 | f-distill | LADD | ECT | CD | Description |
|--------|:----:|:---------:|:----:|:---:|:--:|-------------|
| `total_loss` | x | x | x | x | x | Overall training loss (primary convergence indicator) |
| `vsd_loss` | x | x | | | | Variational Score Distillation loss (score matching) |
| `gan_loss_gen` | x | x | x | | | GAN generator loss (student fools discriminator) |
| `gan_loss_disc` | x | x | x | | | GAN discriminator loss (real vs fake classification) |
| `fake_score_loss` | x | x | | | | Denoising score matching for fake score network |
| `gan_loss_ar1` | x | x | | | | R1 gradient penalty (discriminator regularization) |
| `cm_loss` | | | | x | x | Consistency model loss (weighted) |
| `unweighted_cm_loss` | | | | x | x | Raw L2 consistency distance (unweighted) |

**Convergence indicators to watch:**
- `total_loss` should decrease steadily; sudden spikes may indicate instability
- `gan_loss_gen` and `gan_loss_disc` should roughly balance (neither dominating)
- `cm_loss` should decrease monotonically for ECT/CD
- If `vsd_loss` oscillates wildly, learning rate may need adjustment

#### B. Generation Quality Metrics (post-training evaluation)

Run after training completes, using saved checkpoints + inference script:

| Metric | Category | Description | Our Target | How to Compute |
|--------|----------|-------------|------------|----------------|
| **VBench Total** | Video Quality | Comprehensive video benchmark (16 dimensions) | > 80.0 | `vbench` package (requires setup) |
| **VBench Quality** | Video Quality | Temporal consistency, motion, aesthetic, etc. | > 82.0 | Subset of VBench |
| **VBench Semantic** | Text Alignment | Text-video semantic alignment | > 75.0 | Subset of VBench |
| **FVD** | Video Quality | Frechet Video Distance (lower = better) | < 200 | I3D features, real vs generated |
| **FID** | Frame Quality | Frechet Inception Distance per-frame | < 30 | InceptionV3 features |
| **CLIP Score (CLIP-T)** | Text Alignment | Text-video cosine similarity | > 0.25 | CLIP ViT-B/32 |
| **LPIPS** | Perceptual Sim | Perceptual distance: teacher vs student output | < 0.3 | VGG-based, same prompt comparison |

**NVIDIA's reference results (WanT2V DMD2, 14B teacher data):**

| Steps | VBench Total | VBench Quality | VBench Semantic |
|-------|-------------|----------------|-----------------|
| 2-step | 84.53 | 85.69 | 79.92 |
| 4-step | 84.72 | 85.86 | 80.15 |

> Our results with 1.3B model + OpenVid-1M data will likely be lower than NVIDIA's 14B numbers.
> The goal is **relative comparison between methods**, not matching NVIDIA's absolute scores.

#### C. Efficiency Metrics (measured during training & inference)

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **Training time** | Total wall-clock hours for 6000 iterations | From training log timestamps |
| **Per-iter time** | Seconds per training iteration | From W&B or log moving average |
| **Peak VRAM** | Max GPU memory during training | `nvidia-smi` or W&B GPU stats |
| **Inference time** | Seconds per video (student, 4-step) | Inference script with timing |
| **Speedup ratio** | Teacher time / Student time | Baseline: 167s (50-step teacher) |
| **Convergence speed** | Iterations to reach quality plateau | From loss curve elbow detection |

#### D. Recommended Evaluation Workflow

```
Phase 0 Evaluation Pipeline:
│
├── During Training (automatic)
│   ├── W&B loss curves: total_loss, method-specific losses
│   ├── W&B sample videos: every validation_iter=500 steps
│   └── GPU memory monitoring: nvidia-smi snapshots
│
├── After Each Method Completes
│   ├── Student inference: generate 50-100 videos with fixed prompts
│   ├── Teacher inference: generate same prompts for comparison (already done: 5 videos)
│   ├── CLIP Score: text-video alignment (quick, automated)
│   ├── LPIPS: teacher vs student perceptual similarity (quick, automated)
│   └── Qualitative review: side-by-side video comparison
│
└── Final Comparison Report
    ├── VBench evaluation: full 16-dimension benchmark (if setup ready)
    ├── FVD computation: requires real video reference set
    ├── Convergence speed comparison: overlay loss curves
    ├── Efficiency table: time, VRAM, speedup
    └── Recommendation: best method for Phase 1 progressive distillation
```

**Suggested priority for Phase 0:**
1. **Must-have:** Loss curves (auto), CLIP Score, LPIPS, inference time, VRAM
2. **Should-have:** VBench (needs `pip install vbench`), qualitative comparison
3. **Nice-to-have:** FVD (needs reference dataset), FID

### 5.4 Comparison Table (to be filled)

| Metric | DMD (CausVid) | ECT | CD | f-distill | LADD |
|--------|---------------|-----|-----|-----------|------|
| Source | Pretrained ckpt | Self-trained | Self-trained | OOM | OOM |
| Training time (hours) | N/A (pretrained) | **36.5h** | In progress | - | - |
| Per-iter time (seconds) | N/A | 21.4 | TBD | - | - |
| Peak VRAM (GB) | ~15 (inference) | 25.75 | TBD | >32 | >32 |
| Convergence (iter to plateau) | N/A | ~100 iters | TBD | - | - |
| Final total_loss | N/A | ~645 | TBD | - | - |
| CLIP Score (CLIP-T) | TBD | TBD | TBD | - | - |
| LPIPS (vs teacher) | TBD | TBD | TBD | - | - |
| VBench Total | TBD | TBD | TBD | - | - |
| Inference steps | 3 | **1** | 4 | - | - |
| Inference time (per video) | **9.26s** | **1.7s** | TBD | - | - |
| Speedup vs teacher (167s) | **18x** | **94x** | TBD | - | - |
| Subjective quality (1-5) | TBD | TBD | TBD | - | - |

### Deliverables
- [ ] All 5 methods trained to completion
- [ ] Comparison table filled with measured values
- [ ] Loss curve plots (overlaid for all methods)
- [ ] Sample video comparisons (same prompts, side-by-side)
- [ ] Written reproduction report with analysis
- [ ] Recommendation for Phase 1 progressive distillation method

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
├── CausVid/                             # CausVid repo (DMD distillation, CVPR 2025)
│   ├── causvid/                         # Python package (installed via pip -e)
│   │   └── models/wan/                  # Wan model wrappers & inference pipelines
│   ├── configs/
│   │   └── wan_bidirectional_dmd_from_scratch.yaml  # CausVid config
│   ├── minimal_inference/
│   │   └── bidirectional_inference.py   # CausVid inference script
│   └── wan_models/
│       └── Wan2.1-T2V-1.3B/            # Wan original format (17GB, NOT Diffusers)
│           ├── models_t5_umt5-xxl-enc-bf16.pth  # T5 text encoder (11GB)
│           ├── Wan2.1_VAE.pth           # VAE (485MB)
│           └── diffusion_pytorch_model.safetensors  # DiT weights
├── causvid_checkpoints/
│   └── bidirectional_checkpoint2/
│       └── model.pt                     # CausVid DMD-distilled weights (11GB)
├── hf-download.py                       # HuggingFace mirror download script
├── setup_server.sh                      # Server setup script
├── scripts/
│   ├── download_model.sh               # Model download script
│   ├── download_causvid.sh             # CausVid checkpoint download script
│   ├── run_inference.sh                 # Inference script
│   ├── run_dmd2_single_gpu.sh          # DMD2 training script (OOM on 4x32GB)
│   ├── run_ect_single_gpu.sh           # ECT training script (custom config)
│   ├── launch_ect_now.sh               # ECT launcher (4-GPU, currently running)
│   ├── run_cd_single_gpu.sh            # CD training script (custom config)
│   ├── run_fdistill_single_gpu.sh      # f-distill training script
│   ├── run_ladd_single_gpu.sh          # LADD training script
│   ├── run_meanflow_single_gpu.sh      # MeanFlow training script
│   └── run_all_sequential.sh           # Sequential 5-method runner
├── fastgen_output/
│   ├── ect_train.log                    # ECT training log (ongoing)
│   ├── causvid_inference.log            # CausVid inference log
│   └── causvid_samples/                 # CausVid generated videos
│       ├── output_000.mp4 ~ output_004.mp4  # 5 test videos
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

## Task 5: Pretrained Model Inference Comparison — ✅ COMPLETE

### Background (2026-03-10)

Mentor feedback: Phase 0 should focus on understanding FastGen + exploring capability boundaries,
not reproducing training from scratch. Self-trained ECT/CD quality was very poor. Pivoted to
pretrained model inference comparison.

### Execution Log (2026-03-12)

**First attempt (2026-03-10):** Batch script `run_all_inference_comparison.sh` launched via nohup.
All 4 models failed:
- Teacher: `torchrun` crashed with exit=1 (should use direct Python for single-GPU)
- CausVid: `ModuleNotFoundError: No module named 'causvid'` (package not installed)
- rCM: `TypeError: _pytorch_apply_rotary_emb() got unexpected keyword argument 'interleaved'` (flash_attn fallback incompatible)
- TurboDiffusion: `ModuleNotFoundError: No module named 'ops'` (custom CUDA extensions not compiled)

**Fixes applied (2026-03-12):**
1. **Teacher:** Changed from `torchrun --nproc_per_node=1` to direct `python`; used `--prompt_file` instead of OmegaConf `model.prompts=` override
2. **CausVid:** `pip install -e .` in CausVid directory
3. **rCM:** Rewrote `rope_apply()` function in `wan2pt1.py` — replaced `flash_apply_rotary_emb(x, cos, sin, interleaved=True)` with pure PyTorch interleaved rotary embedding (cos/sin multiplication + stack)
4. **TurboDiffusion:** Made `ops` and `SLA` imports conditional, but `create_model()` still calls `FastLayerNorm.from_layernorm()` which is None — requires the CUDA extensions to be compiled for SM 12.0. **Unsolvable without updating kernel code.**

**Second attempt (2026-03-12):** Two scripts deployed:
- `fix_and_run_inference.sh` → CausVid ✅ (5/5), rCM ✅ (5/5), Teacher ❌ (config error), TurboDiffusion ❌ (ops)
- `run_teacher_inference.sh` → Teacher ✅ (5/5)

### Final Results

| Model | Method | Steps | Time/Video | Speedup | Videos |
|-------|--------|-------|-----------|---------|--------|
| Teacher (Wan2.1-1.3B) | Baseline | 50 | **183s** | 1x | 5/5 ✅ |
| CausVid (DMD pretrained) | DMD | 3 | **28.5s** | **6.4x** | 5/5 ✅ |
| rCM (NVlabs pretrained) | Consistency Model | 4 | **37.6s** | **4.9x** | 5/5 ✅ |
| TurboDiffusion | rCM + Quant | 4 | — | — | 0/5 ❌ |

**Outputs:**
- Server: `/data/chenqingzhan/fastgen_output/comparison_2026_03_12/{teacher,causvid,rcm}/`
- Local: `03-dmd-distillation/results/comparison/{teacher,causvid,rcm}/`
- 15 videos total (5 prompts × 3 models), all using seed=42, 480p 81 frames

### Scripts & Files Reorganization (2026-03-12)

Reorganized 26 scripts from flat `scripts/` into 5 categorized subdirectories:
- `scripts/setup/` — Environment & model download (2 scripts)
- `scripts/data/` — Training data preparation (3 scripts)
- `scripts/configs/` — Custom FastGen experiment configs (2 files)
- `scripts/inference/` — Inference & evaluation (8 scripts)
- `scripts/training/` — Distillation training (11 scripts)

Added `scripts/README.md` with detailed documentation for each script.
Moved `experiment-report-phase0.md` to `archive/` (superseded by `Phase0_Report.md`).
Moved early test samples to `results/early_tests/`.

### Git Commit

```
5f5d649 feat(task3): complete Phase 0 inference comparison & reorganize scripts
```
Pushed to `origin/Task3_dev_ChenHingChin` on 2026-03-12.

---

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
