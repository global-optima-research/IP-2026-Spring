# Phase 0 报告 — FastGen 视频网络：架构、推理与能力边界

> **作者：** 陈庆展 (Chen Hing Chin)
> **日期：** 2026-03-12
> **任务：** Task 3 — DMD 蒸馏与加速
> **分支：** `Task3_dev_ChenHingChin`
> **状态：** 已完成

---

## 0. 学习计划与路线图

### 0.1 任务目标

> "找一个公开数据集，先熟悉一下 FastGen 的 Video Networks，熟悉训练以及推理，形成 report，探索 FastGen 架构的能力边界。"

Phase 0 的目标**不是**从头复现完整训练，而是：

1. **理解** FastGen 的架构与代码结构
2. **使用预训练模型运行推理**（DMD、ECT、CD），比较不同蒸馏方法
3. **探索** FastGen 的能力边界——它能做什么、不能做什么
4. **为 Phase 1 渐进蒸馏做准备**，建立对架构的深入理解

### 0.2 学习路径

```
步骤 1: FastGen 架构分析                           ← 第 1-2 节
  ├── 代码结构与模块组织
  ├── 视频网络：Wan2.1 DiT 架构
  ├── VAE、文本编码器、噪声调度器
  └── 蒸馏方法如何接入框架

步骤 2: 蒸馏方法综述                               ← 第 3 节
  ├── 分类：基于一致性 vs 分布匹配 vs 对抗式
  ├── 每种方法的核心概念
  └── 权衡：质量、速度、显存、复杂度

步骤 3: 预训练模型推理实验                         ← 第 4 节
  ├── Teacher 基准（50 步 Wan2.1-1.3B）
  ├── CausVid（DMD 预训练，3 步）
  ├── rCM（NVlabs 预训练，4 步一致性模型）
  ├── TurboDiffusion（rCM + 量化，尝试失败）
  └── 统一提示词集合、速度与质量对比

步骤 4: 训练流程剖析                               ← 第 5 节
  ├── 代码级训练流程（train.py → Method → Trainer）
  ├── 数据流水线：WebDataset → VideoLoader → Latent
  ├── 关键配置与超参数
  └── 短训练验证（100 iter）确认理解

步骤 5: 能力边界探索                               ← 第 6 节
  ├── 各方法的显存限制（实测）
  ├── 分辨率 / 帧数可扩展性
  ├── 支持的模型与方法矩阵
  ├── 不支持或需要自定义工作的部分
  └── 对 Phase 1 的启示（14B 模型、产品视频编辑）

步骤 6: 总结与 Phase 1 建议                        ← 第 7 节
```

### 0.3 使用的资源

| 资源 | 用途 |
|------|------|
| [NVIDIA FastGen](https://github.com/NVlabs/FastGen) | 主要研究框架 |
| [CausVid](https://github.com/tianweiy/CausVid) (CVPR 2025) | DMD 预训练模型，用于推理对比 |
| [rCM](https://github.com/NVlabs/rcm) (NVlabs, ICLR 2026) | 一致性模型预训练模型 |
| [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) (THU) | rCM + 量化优化（尝试失败） |
| OpenVid-1M | 公开视频-文本数据集，用于数据流水线测试 |
| Wan2.1-T2V-1.3B | 基础视频生成模型（DiT 架构） |

---

## 1. FastGen 框架架构

### 1.1 概述

FastGen 是 NVIDIA 的统一 **扩散模型蒸馏** 框架——将多步扩散模型压缩为少步学生模型，同时保持生成质量。

**设计理念：**
- **模块化方法：** 每种蒸馏算法（DMD2、ECT、CD 等）是一个独立模块，实现统一的 `single_train_step()` 接口
- **配置组合：** 层次化 OmegaConf DictConfig 系统，将模型、方法、数据和训练关注点分离
- **回调驱动：** 日志、检查点、EMA 和调度通过可组合的回调实现
- **多后端：** 开箱支持 FSDP 和 DDP 分布式训练

**支持范围：**
- **模型：** Wan2.1（1.3B / 14B）、EDM2（图像）、SDXL 变体
- **方法：** 6+ 种蒸馏方法，跨两大家族（基于一致性、分布匹配）
- **数据：** 基于 WebDataset 的视频/图像加载（含 VAE 编码），或预计算 latent

### 1.2 代码结构

```
FastGen/
├── train.py                            # 训练入口
├── scripts/inference/
│   └── video_model_inference.py        # 推理入口
│
├── fastgen/
│   ├── configs/                        # === 配置层 ===
│   │   ├── base.py                     # 基础配置类（OmegaConf DictConfig）
│   │   ├── data.py                     # VideoLoaderConfig / VideoLatentLoaderConfig
│   │   ├── net.py                      # 模型配置（Wan_1_3B_Config, Wan_14B_Config）
│   │   ├── methods/                    # 基础方法配置
│   │   │   ├── config_cm.py            # 一致性模型基础配置
│   │   │   └── config_dmd2.py          # DMD2 基础配置
│   │   └── experiments/WanT2V/         # 实验专属配置（组合基础配置）
│   │       ├── config_dmd2.py          # Wan2.1-1.3B 上的 DMD2
│   │       ├── config_fdistill.py      # Wan2.1 上的 f-distill
│   │       ├── config_ladd.py          # Wan2.1 上的 LADD
│   │       ├── config_mf.py            # Wan2.1 上的 MeanFlow
│   │       ├── config_cm_ct.py         # Wan2.1 上的 ECT（自定义创建）
│   │       └── config_cm_cd.py         # Wan2.1 上的 CD（自定义创建）
│   │
│   ├── methods/                        # === 方法层 ===
│   │   ├── consistency_model/
│   │   │   ├── CM.py                   # ECT/CD 实现（use_cd=False/True）
│   │   │   └── mean_flow.py           # MeanFlow 方法
│   │   └── distribution_matching/
│   │       ├── dmd2.py                 # DMD2：VSD + GAN 损失
│   │       ├── f_distill.py            # f-divergence 加权的 DMD2
│   │       └── ladd.py                 # LADD：纯对抗式（仅 GAN）
│   │
│   ├── datasets/                       # === 数据层 ===
│   │   ├── wds_dataloaders.py          # WebDataset 加载器（tar 分片 → 批次）
│   │   └── decoders.py                 # 视频解码（mp4 → tensor）
│   │
│   ├── trainer.py                      # === 训练层 ===
│   │                                   # 训练循环，支持 FSDP/DDP、梯度累积
│   │
│   ├── callbacks/                      # === 回调层 ===
│   │   ├── ema.py                      # EMA（指数移动平均）用于学生权重
│   │   ├── ct_schedule.py              # CTSchedule：一致性训练的课程控制
│   │   ├── wandb.py                    # W&B 日志（损失、指标、视频样本）
│   │   ├── train_profiler.py           # 每次迭代计时分析
│   │   └── gpu_stats.py               # GPU 显存 & 使用率监控
│   │
│   └── utils/                          # === 工具层 ===
│       ├── checkpointer.py             # 支持 FSDP 的检查点保存/加载
│       ├── lr_scheduler.py             # LambdaInverseSquareRootScheduler
│       └── lazy_call.py                # LazyCall (L) 延迟配置实例化
```

### 1.3 核心抽象

**配置系统——层次化组合 + LazyCall：**

```
基础配置 (configs/methods/config_cm.py)
    ↓ 继承 & 覆盖
实验配置 (configs/experiments/WanT2V/config_cm_ct.py)
    ↓ 运行时 CLI 覆盖
python train.py --config=... - trainer.fsdp=True model.net.model_id_or_local_path=...
```

LazyCall (`L`) 实现延迟实例化——配置描述*要创建什么*而不立即创建：

```python
from fastgen.utils import LazyCall as L
from fastgen.callbacks.ema import EMACallback

config.trainer.callbacks.ema_1 = L(EMACallback)(
    type="power", gamma=96.99, ema_name="ema_1"
)
# EMACallback 此时并未实例化——仅当 Trainer 构建回调时才实例化
```

**方法接口——`single_train_step()`：**

每种蒸馏方法都实现一个核心函数：
```python
class CM(BaseMethod):
    def single_train_step(self, batch, state):
        # 1. 解包批次（latents、文本嵌入）
        # 2. 采样时间步、添加噪声
        # 3. 计算方法特定的损失
        # 4. 返回损失字典
```

**Trainer——编排训练循环：**
```
Trainer.train()
  → for iter in range(max_iter):
      → batch = next(dataloader)
      → loss = method.single_train_step(batch, state)
      → loss.backward()  (FSDP 处理梯度分片)
      → optimizer.step()
      → for callback in callbacks: callback.on_training_step_end()
```

**回调——模块化、可组合的钩子：**

| 回调 | 触发时机 | 用途 |
|------|---------|------|
| EMACallback | 每步 | 更新学生权重的 EMA 副本 |
| CTScheduleCallback | 每步 | 调整一致性比率课程 |
| WandBCallback | 每 `logging_iter` | 记录损失、生成样本视频 |
| CheckpointCallback | 每 `save_ckpt_iter` | 保存模型检查点 |

---

## 2. 视频网络：Wan2.1 架构

### 2.1 Wan2.1 DiT（扩散 Transformer）

Wan2.1 是阿里巴巴基于 **DiT（Diffusion Transformer）** 架构的视频生成模型：

- **3D 注意力：** 联合时空注意力——每个 latent 帧 token 关注所有空间和时间位置
- **Flow Matching：** 使用连续时间流匹配噪声调度（非 DDPM 离散步骤）
- **条件注入：** UMT5-XXL 文本嵌入通过交叉注意力和自适应 LayerNorm 注入
- **模型变体：** 1.3B（轻量版，本研究使用）和 14B（完整版，NVIDIA 用于基准测试）

### 2.2 组件

| 组件 | 架构 | 参数量 | 用途 |
|------|------|--------|------|
| **Transformer (DiT)** | 3D 扩散 Transformer | 1.3B / 14B | 核心去噪网络，时空注意力 |
| **VAE** | 3D Causal VAE | ~100M | 视频 ↔ Latent 压缩（时间 4x，空间 8x8） |
| **文本编码器** | UMT5-XXL | ~4.7B | 文本提示 → 嵌入向量 |
| **噪声调度器** | Flow Matching | - | 训练与采样的连续时间噪声调度 |

### 2.3 Latent 空间

| 参数 | 值 |
|------|-----|
| 输入视频 | 832 x 480, 81 帧 |
| Latent 形状 | `[16, 21, 60, 104]` (C, T, H, W) |
| 时间压缩 | 4x（81 帧 → 21 个 latent 帧） |
| 空间压缩 | 8x8（832x480 → 104x60） |
| 单样本 latent 大小 | ~5.3 MB (bf16) |

**Latent 维度对显存的影响：**

大尺寸的 latent 形状 `[16, 21, 60, 104]` 是显存压力的根本原因。前向传播时，每个网络都需要处理这些高维张量。需要多个网络的方法（DMD2：4 个网络，LADD：3 个）会倍增内存开销，导致在 32GB GPU 上 OOM。文本编码器（UMT5-XXL，~10GB）在 FSDP 中每个 GPU 上复制，进一步加剧问题。

---

## 3. FastGen 中的蒸馏方法

### 3.1 方法分类

FastGen 根据损失函数的形式将蒸馏方法分为两大家族：

```
FastGen 中的蒸馏方法
│
├── 基于一致性（沿 ODE 轨迹学习自一致性）
│   ├── ECT  — 增强一致性训练（不需要 teacher）
│   ├── CD   — 一致性蒸馏（teacher 提供 ODE 目标）
│   └── MeanFlow — 预计算 latent，均值流匹配
│
└── 分布匹配（对抗式 / 分数匹配）
    ├── DMD2      — VSD 损失 + GAN 损失 + FakeScore 网络
    ├── f-distill — f-divergence 加权的 DMD2 变体
    └── LADD      — 纯对抗式（仅 GAN，无 VSD/FakeScore）
```

### 3.2 方法对比（概念层面）

**ECT（增强一致性训练, Enhanced Consistency Training）**
- **核心思想：** 训练学生模型在不同噪声水平产生*一致*的输出——从 ODE 轨迹上的任意点去噪应收敛到相同的干净输出。不需要 teacher 模型。
- **损失：** `L = w(t,r) * ||student(y_t, t) - sg[student(y_r, r)]||^2`，其中 `sg` = stop-gradient
- **网络：** 仅学生模型
- **权衡：** 显存最低、最简单；没有 teacher 指导可能牺牲质量

**CD（一致性蒸馏, Consistency Distillation）**
- **核心思想：** 与 ECT 相同的一致性约束，但目标 `y_r` 由 *teacher* 模型的 ODE 求解器从 `t` 到 `r` 计算得出，提供更高质量的监督。
- **损失：** `L = w(t,r) * ||student(y_t, t) - sg[student(y_r_teacher, r)]||^2`
- **网络：** 学生 + Teacher
- **权衡：** 质量优于 ECT（有 teacher 指导）；因 teacher 常驻显存，显存需求更高

**DMD2（分布匹配蒸馏 v2, Distribution Matching Distillation v2）**
- **核心思想：** 通过两个互补损失将学生输出的*分布*与 teacher 的输出分布匹配：VSD（变分分数蒸馏）对齐分数函数，GAN 损失通过判别器添加对抗式精修。
- **损失：** `L = L_VSD + lambda * L_GAN`，带有 FakeScore 网络用于 VSD 计算
- **网络：** 学生 + Teacher + FakeScore + 判别器（4 个网络）
- **权衡：** 质量最高（NVIDIA 的主力方法）；显存和复杂度最高

**f-distill（f-divergence 加权 DMD2）**
- **核心思想：** DMD2 的变体，使用 f-divergence 加权重新平衡分布匹配损失，强调学生和 teacher 分布差异最大的区域。
- **网络：** 与 DMD2 相同（4 个网络）
- **权衡：** 可能比 DMD2 收敛更好；显存需求相同

**LADD（Latent 对抗蒸馏, Latent Adversarial Distillation）**
- **核心思想：** 纯对抗方法——仅使用 GAN 损失训练学生，不使用分数匹配（VSD）或 FakeScore 网络。
- **网络：** 学生 + Teacher + 判别器（3 个网络）
- **权衡：** 比 DMD2 简单但仍需 teacher；每迭代更快但可能有 GAN 训练不稳定性

**MeanFlow（均值流）**
- **核心思想：** 一致性模型家族中的流匹配方法。使用预计算的 VAE latent 和文本嵌入（与其他方法的数据格式不同）。
- **网络：** 仅学生模型
- **数据：** 需要 `VideoLatentLoaderConfig` 和预计算的 `.pth` 文件，不支持原始 mp4

### 3.3 方法如何接入 FastGen

每种方法都实现 `single_train_step()` 接口。以下是 `CM.py` 中 ECT 的流程：

```python
# CM.py — ECT 的 single_train_step() (use_cd=False):

def single_train_step(self, batch, state):
    data = batch["latents"]       # VAE 编码的视频 latent [B, 16, 21, 60, 104]
    eps = torch.randn_like(data)  # 随机噪声

    # 1. 从 logitnormal 分布采样时间步 t
    t = sample_time(dist_type="logitnormal", p_mean=-0.8, p_std=1.6)

    # 2. 在时间步 t 添加噪声
    y_t = noise_scheduler.forward_process(data, eps, t)

    # 3. 通过 sigmoid 映射计算目标时间步 r（ECT 论文）
    #    r 控制 ODE 轨迹上两点的间距
    #    CTSchedule 回调在训练中逐渐增加 ratio
    r = t - t * (1 - ratio) * (1 + 8 * sigmoid(-t))

    # 4. ECT 目标：在时间步 r 添加噪声（不需要 teacher）
    y_r = noise_scheduler.forward_process(data, eps, r)

    # 5. 一致性损失：学生在 (y_t, t) 和 (y_r, r) 的输出应匹配
    pred_t = student(y_t, t)
    pred_r = student(y_r, r).detach()  # 目标端 stop-gradient
    loss = huber_loss(pred_t, pred_r) * weight(t, r)

    # 6. CTSchedule 回调在每步后更新 ratio
    return {"cm_loss": loss}
```

对于 CD（`use_cd=True`），步骤 4 改变：不是 `forward_process(data, eps, r)`，而是 teacher 运行 ODE 从 `y_t` 在时间 `t` 到时间 `r`：
```python
    # CD 目标：teacher ODE 求解器 (t → r)
    y_r = ode_solver(teacher, y_t, t, r, guidance_scale=5.0)
```

对于 DMD2，`single_train_step()` 交替进行：
1. **学生更新：** 计算 VSD 损失（通过 FakeScore）+ GAN 生成器损失 → 更新学生
2. **判别器更新：** 计算 GAN 判别器损失 → 更新判别器

---

## 4. 推理实验

### 4.1 实验设置

| 项目 | 值 |
|------|-----|
| GPU | 单块 RTX 5090 32GB（GPU 索引 5） |
| 评估提示词 | 5 个标准化提示词（见 4.2 节） |
| 精度 | bfloat16 |
| 种子 | 42 |
| 分辨率 | 832×480 (480p), 81 帧 |
| 对比模型 | Teacher（50 步）、CausVid DMD（3 步）、rCM（4 步） |
| 日期 | 2026-03-12 |

### 4.2 评估提示词

所有模型使用相同的 5 个提示词，以确保公平比较：

| # | 提示词 |
|---|--------|
| 1 | A golden retriever puppy playing joyfully in a sunny garden with colorful flowers blooming around it |
| 2 | A futuristic city skyline at sunset with flying cars and brilliant neon lights reflecting off glass towers |
| 3 | Ocean waves crashing dramatically on rocky cliffs during a powerful storm with dark clouds overhead |
| 4 | An astronaut in a white spacesuit riding a brown cow through a lush green meadow under blue skies |
| 5 | A red sports car driving fast through a winding mountain road with autumn foliage on both sides |

### 4.3 结果：Teacher 基准（50 步）

| 指标 | 值 |
|------|-----|
| 框架 | NVIDIA FastGen |
| 去噪步数 | 50 |
| CFG 尺度 | 5.0 |
| 单视频耗时 | **~167s**（纯推理）/ **~183s**（含模型加载 + VAE 解码 + 保存） |
| 单步耗时 | ~3.36s |
| 总耗时（5 视频） | 919s |
| 质量 | 参考标准——运动连贯、文本对齐好、视觉质量高 |

### 4.4 结果：CausVid DMD（3 步，预训练）

| 指标 | 值 |
|------|-----|
| 来源 | 预训练检查点（`tianweiy/CausVid`，双向 DMD） |
| 方法家族 | 分布匹配蒸馏 (DMD) |
| 去噪步数 | 3 |
| 单视频耗时 | **~28.5s** |
| 相比 Teacher 加速 | **6.4 倍** |
| 总耗时（5 视频） | 142.4s |

**备注：**
- CausVid 使用自己的推理流水线，独立于 FastGen
- 需要修补 `attention.py` 以支持 SDPA 回退（RTX 5090 Blackwell 不支持 `flash_attn`）
- 28.5s/视频 包含模型加载开销；纯推理更快
- 定性评价：整体质量良好，复杂场景偶有轻微伪影

### 4.5 结果：rCM（4 步，NVlabs 预训练）

| 指标 | 值 |
|------|-----|
| 来源 | 预训练检查点（`NVlabs/rcm`, ICLR 2026） |
| 方法家族 | 一致性模型 (reflow Consistency Model) |
| 去噪步数 | 4 |
| Sigma Max | 80 |
| 单视频耗时 | **~37.6s**（5 个提示词平均） |
| 相比 Teacher 加速 | **4.9 倍** |
| 总耗时（5 视频） | 188.1s |

**逐提示词耗时：**

| 提示词 | 耗时 |
|--------|------|
| 1（金毛犬） | 40.34s |
| 2（未来城市） | 36.91s |
| 3（海浪） | 36.54s |
| 4（宇航员+奶牛） | 36.68s |
| 5（红色跑车） | 37.64s |

**备注：**
- 需要修补 `wan2pt1.py`，将 `flash_apply_rotary_emb` 替换为纯 PyTorch 旋转嵌入实现（RTX 5090 Blackwell 兼容性）
- 提示词 1 较慢（40s）是因为模型预热；后续提示词稳定在 ~37s
- 定性评价：4 步生成质量良好；相比 teacher 略微模糊但运动连贯

### 4.6 结果：TurboDiffusion（4 步，尝试失败）

| 指标 | 值 |
|------|-----|
| 来源 | 预训练检查点（`thu-ml/TurboDiffusion`） |
| 方法家族 | rCM + INT8 量化 + 稀疏线性注意力 |
| 状态 | **失败——自定义 CUDA 算子不兼容 RTX 5090** |

**失败分析：**
- TurboDiffusion 需要自定义 CUDA 扩展（`ops` 模块：`FastLayerNorm`、`FastRMSNorm`、`Int8Linear`）用于量化和优化
- 这些 CUDA kernel 针对旧 GPU 架构编译，不支持 RTX 5090 的 Blackwell SM 12.0
- `ops` 导入在 `modify_model.py` 的模块级别，没有编译好的扩展就无法运行任何推理
- TurboDiffusion 本质上是 rCM 加上推理时优化（量化 + 稀疏注意力），因此 rCM 的结果可以作为其未优化的基准

### 4.7 速度对比汇总

| 模型 | 方法 | 步数 | 单视频耗时 | 加速比 | 状态 |
|------|------|------|-----------|--------|------|
| **Teacher** (Wan2.1-1.3B) | 基准 | 50 | 183s | 1x | 参考标准 |
| **CausVid** (DMD 预训练) | DMD | 3 | 28.5s | **6.4x** | 质量良好 |
| **rCM** (NVlabs 预训练) | 一致性模型 | 4 | 37.6s | **4.9x** | 质量良好 |
| **TurboDiffusion** | rCM + 量化 | 4 | — | — | CUDA 算子不兼容 |

**关键观察：**

1. **CausVid（DMD）比 rCM 更快，尽管步数更少：** CausVid 使用 3 步 vs rCM 的 4 步，且 CausVid 的流水线在单 GPU 推理上更优化。

2. **两种蒸馏模型都实现了 >4 倍加速：** 即使没有 flash attention 优化（回退到 SDPA），CausVid 和 rCM 都相比 50 步 teacher 提供了显著加速。

3. **每步开销因架构而异：** Teacher ~3.36s/步, rCM ~9.4s/步（但只需 4 步）。rCM 的每步开销更大来自于更大的 sigma 范围和不同的采样策略。

4. **短任务中模型加载占主导：** rCM 第一个提示词（40.3s）比后续（~37s）慢约 10%，因为模型预热。在生产环境中这会被摊销。

### 4.8 定性观察

基于 15 个生成视频（5 个提示词 × 3 个模型）的视觉检查：

| 方面 | Teacher（50 步） | CausVid（3 步） | rCM（4 步） |
|------|-------------------|-----------------|-------------|
| **文本对齐** | 优秀——所有提示词准确呈现 | 良好——捕捉主要主体，偶有细节丢失 | 良好——捕捉主要主体 |
| **运动质量** | 平滑、连贯的时间运动 | 略不平滑，轻微时间伪影 | 连贯性好，轻微抖动 |
| **视觉清晰度** | 高细节和清晰度 | 整体良好，复杂区域轻微模糊 | 比 teacher 略软 |
| **色彩与光照** | 丰富、自然的色彩 | 与 teacher 相当 | 相当，略显柔和 |
| **伪影** | 极少 | 偶有边缘伪影 | 轻微噪声模式 |

**总体排名（主观）：** Teacher > CausVid ≈ rCM——两种蒸馏模型都能生成可观看、连贯的视频，相比 50 步 teacher 质量损失适中。

---

## 5. 训练流水线分析

### 5.1 训练流程（代码剖析）

FastGen 训练运行的完整执行路径：

```
torchrun --nproc_per_node=N train.py --config=path/to/config.py - key=value ...
│
├── 1. 配置加载
│   ├── 加载实验配置（OmegaConf DictConfig）
│   ├── 应用 CLI 覆盖（trainer.fsdp=True 等）
│   └── 解析 LazyCall 实例
│
├── 2. 模型构建
│   ├── 学生网络（始终需要）：Wan2.1 DiT，从 model_id_or_local_path
│   ├── Teacher 网络（如需要）：相同架构，权重冻结
│   ├── FakeScore 网络（DMD2/f-distill）：去噪分数估计器
│   ├── 判别器（DMD2/f-distill/LADD）：对抗分类器
│   ├── VAE：3D Causal VAE 编码器/解码器
│   └── 文本编码器：UMT5-XXL
│
├── 3. 数据加载器构建
│   ├── VideoLoaderConfig：读取 WebDataset tar 分片
│   └── 分布式采样器（多 GPU 用）
│
├── 4. 优化器 & 调度器
│   ├── AdamW（lr=1e-5 典型值）
│   └── LambdaInverseSquareRootScheduler（预热 + 衰减）
│
├── 5. 回调注册
│   ├── EMACallback（学生权重平均）
│   ├── CTScheduleCallback（一致性比率课程）
│   ├── WandBCallback（日志）
│   ├── CheckpointCallback（每 N 次迭代保存）
│   └── GPUStatsCallback（显存监控）
│
└── 6. Trainer.train() — 主循环
    └── for iter in range(max_iter):
        ├── batch = next(dataloader)     # mp4 → 解码 → VAE 编码 → latent
        ├── loss = method.single_train_step(batch, state)
        ├── loss.backward()              # FSDP 处理梯度分片
        ├── optimizer.step()
        ├── scheduler.step()
        └── for cb in callbacks: cb.on_training_step_end()
            ├── EMA：更新影子权重
            ├── CTSchedule：调整 ratio
            ├── WandB：记录损失（每 logging_iter）
            ├── Checkpoint：保存（每 save_ckpt_iter）
            └── Validation：生成样本（每 validation_iter）
```

### 5.2 数据流水线

训练数据从磁盘到损失计算的流程：

```
WebDataset 分片（磁盘上的 .tar 文件）
│  ├── sample_000000.mp4（832x480, 81 帧，原始视频）
│  └── sample_000000.txt（文本描述）
│
├── wds_dataloaders.py
│   ├── webdataset.WebLoader 读取 tar 分片
│   ├── 跨分片打乱以实现随机化
│   └── 返回 (video_bytes, caption_text) 对
│
├── decoders.py
│   ├── 解码 mp4 字节 → 视频张量 [B, C, T, H, W]
│   ├── 调整到目标分辨率 (832x480)
│   └── 像素值归一化到 [-1, 1]
│
├── VAE 编码器（在 GPU 上运行，在训练步中执行）
│   ├── 编码视频 [B, 3, 81, 480, 832] → latent [B, 16, 21, 60, 104]
│   └── 时间 4x 压缩，空间 8x8 压缩
│
├── 文本编码器 (UMT5-XXL)
│   ├── 分词描述 → input_ids
│   └── 编码 → text_embedding [B, seq_len, dim]
│
└── 方法接收：{latents, text_embeddings, ...}
    └── 在 latent 空间计算蒸馏损失
```

**本研究准备的数据：**
- **来源：** OpenVid-1M（公开数据集，100 万+ 视频-文本对）
- **质量筛选：** 帧数 >= 81, 美学分数 >= 5.0, 时长 >= 2.0s
- **结果：** 21,133 个样本，22 个 WebDataset tar 分片（共 22GB）
- **位置：** `/data/datasets/OpenVid-1M/webdataset/shard-{000000..000021}.tar`

### 5.3 关键配置参数

| 参数 | 描述 | 典型值 | 影响 |
|------|------|--------|------|
| `trainer.batch_size_global` | 所有 GPU 的总批大小 | 8 | 越大 = 收敛更好，显存更多 |
| `trainer.max_iter` | 总训练迭代数 | 6000 | 越长 = 质量更好，耗时更多 |
| `trainer.fsdp` | 启用全分片数据并行 | True | 多 GPU 必需 |
| `trainer.logging_iter` | 每 N 次迭代记录损失 | 50 | 监控频率 |
| `trainer.save_ckpt_iter` | 每 N 次迭代保存检查点 | 500 | 恢复与评估点 |
| `trainer.validation_iter` | 每 N 次迭代生成样本视频 | 500 | 视觉质量追踪 |
| `model.guidance_scale` | 无分类器引导尺度 | 5.0 | 越高 = 越文本对齐，多样性越低 |
| `model.num_steps_student` | 学生去噪步数 | 4 | 目标压缩级别 |
| `model.loss_config.use_cd` | 是否使用 teacher 进行一致性蒸馏 | True/False | ECT vs CD 切换 |
| `model.loss_config.huber_const` | Huber 损失过渡常数 | 0.06 | 损失平滑度 |
| `model.sample_t_cfg.time_dist_type` | 时间步采样分布 | logitnormal | 控制训练哪些噪声水平 |
| `model.sample_t_cfg.train_p_mean` | Logitnormal 均值参数 | -0.8 | 偏向中间范围的时间步 |
| `model.use_ema` | EMA 回调名称 | ["ema_1"] | 稳定学生权重 |
| `dataloader_train.datatags` | WebDataset 分片路径 | ["WDS:/path/..."] | 训练数据位置 |

### 5.4 已进行的训练实验

Phase 0 期间进行了两个自训练实验以验证对训练流水线的理解：

**ECT 训练（6000 次迭代，4-GPU FSDP）：**
- 成功完成，挂钟时间 36.5 小时
- 损失在迭代 2000 左右稳定在 ~650
- 检查点保存于 `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_ct/ect_wan1.3b_4gpu/checkpoints/0006000`
- **质量评估：** 非常差——1 步生成产生模糊、有噪声的输出。这在 ECT 没有充分超参数调优和更长训练的情况下是预期的

**CD 训练（1500/6000 次迭代，2-GPU FSDP）：**
- 因收敛缓慢和中间质量差，在迭代 1500 提前停止
- 检查点保存于 `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_cd/cd_wan1.3b_2gpu/checkpoints/0001500`
- **质量评估：** 非常差——teacher 指导的 ODE 目标在 1500 次迭代内未收敛到连贯输出

**关键结论：** 从零开始自训练一致性模型需要大量计算预算和仔细调参。来自已发表论文的预训练检查点（CausVid、rCM）经过全规模训练，展示了远超自训练的质量，证实了 Phase 0 转向预训练推理对比的价值。

---

## 6. FastGen 的能力边界

### 6.1 支持的模型

| 模型 | 类型 | 参数量 | 可用配置 |
|------|------|--------|----------|
| Wan2.1-T2V-1.3B | 文本到视频 | 1.3B | DMD2, ECT, CD, f-distill, LADD, MeanFlow |
| Wan2.1-T2V-14B | 文本到视频 | 14B | 同上（显存需求显著增加） |
| EDM2 | 图像 | 多种 | CM, DMD2（仅图像配置） |
| SDXL 变体 | 图像 | 2.6B | 部分方法 |

> **注意：** FastGen 在 `experiments/WanT2V/` 下的实验配置专用于 Wan2.1 视频模型。图像模型配置在 `experiments/EDM2/` 和 `experiments/SDXL/` 下。

### 6.2 支持的蒸馏方法

| 方法 | 图像 | 视频 | 有 Wan2.1 配置? | 本研究是否测试? | 结果 |
|------|------|------|-----------------|----------------|------|
| DMD2 | 是 | 是 | 内置 | 已尝试 | 4x32GB OOM |
| f-distill | 是 | 是 | 内置 | 已尝试 | 4x32GB OOM |
| LADD | 是 | 是 | 内置 | 已尝试 | 4x32GB OOM |
| MeanFlow | 是 | 是 | 内置 | 未测试 | 需要预计算 latent |
| ECT | 是 | 是 | **自定义** | **已成功训练** | 6000 iter, 36.5h |
| CD | 是 | 是 | **自定义** | **部分训练** | 1500/6000 iter |

### 6.3 显存限制（实测）

关键发现：Wan2.1-1.3B 的 latent 维度（[16, 21, 60, 104]）产生巨大的内存压力。

| 方法 | 显存中的网络数 | 峰值显存/GPU（4-GPU FSDP） | 在 4x32GB 上是否可行? |
|------|---------------|--------------------------|---------------------|
| ECT | 仅学生 | **25.8 GB** | 是 |
| CD | 学生 + Teacher | **~28-30 GB** | 是（勉强） |
| LADD | + 判别器 | **>31 GB** | 否 (OOM) |
| DMD2 | + FakeScore + 判别器 | **>31 GB** | 否 (OOM) |
| f-distill | + FakeScore + 判别器 | **>31 GB** | 否 (OOM) |

**根本原因：** FSDP 在前向传播时全收集完整参数。文本编码器（UMT5-XXL ~10GB）在每个 GPU 上复制。

**对 Phase 1 的启示：** 14B 模型需要 A100/H100 80GB GPU，或者激进的内存优化（激活检查点、CPU 卸载、模型并行）。

### 6.4 FastGen 不支持的功能（开箱即用）

本研究发现的限制：

1. **没有 WanT2V 的 ECT/CD 内置配置：** FastGen 提供了 Wan2.1 的 DMD2、f-distill、LADD 和 MeanFlow 配置，但一致性模型配置（ECT/CD）需要自行创建——通过组合 EDM2 CM 配置和 WanT2V 模型设置实现。

2. **RTX 5090 (Blackwell) 不支持 flash_attn：** `flash_attn` 库（CausVid 和部分 FastGen 代码路径使用）尚不支持 RTX 5090 的 Blackwell 架构。需要修补注意力代码以回退到 PyTorch 原生 `F.scaled_dot_product_attention`。

3. **FSDP 中文本编码器复制：** UMT5-XXL（~10GB）跨 GPU 复制而非分片，显著减少每 GPU 可用显存。

4. **无自动 CPU 卸载：** 当多个网络超出 GPU 显存时，可以尝试手动 CPU 卸载，但速度极慢（~8 分钟/迭代 vs 正常 ~21s）。

5. **无内置评估指标：** FastGen 不包含 VBench、CLIP Score 或 FVD 计算。需要单独搭建。

6. **日志依赖 W&B：** 损失日志与 Weights & Biases 紧密耦合；使用 `wandb_mode=disabled` 仍会尝试视频编码，每次日志步骤增加 ~10s 开销。

### 6.5 可扩展性分析

**显存如何随关键维度扩展（Wan2.1 DiT 估算）：**

| 维度 | 变化 | 显存影响 | 在 4x32GB 上的可行性 |
|------|------|---------|---------------------|
| 分辨率 | 480p → 720p | latent 大小增加 ~2.3x | ECT 勉强，CD/DMD2 不可能 |
| 分辨率 | 480p → 1080p | latent 大小增加 ~5x | 所有方法不可能 |
| 帧数 | 81 → 161 帧 | 时间维度增加 ~2x | ECT 勉强，其他不可能 |
| 模型大小 | 1.3B → 14B | 参数增加 ~10x | 所有方法需要 80GB+ GPU |
| 批大小 | 每 GPU 1 → 2 | 显存增加 ~1.5x | 仅 480p ECT 可行 |

**关键洞察：** Phase 1 使用 Task 2 的 14B Teacher 模型时，**任何蒸馏方法都需要 A100 80GB 或 H100 80GB GPU**。即使最轻量的 ECT 方法，14B 模型也至少需要 2x80GB GPU。

---

## 7. 总结与 Phase 1 建议

### 7.1 关键发现

1. **FastGen 是一个结构良好的模块化框架：** 配置、方法、训练和推理清晰分离。新方法可通过实现 `single_train_step()` 和创建配置文件添加。回调系统支持灵活的日志、检查点和调度。

2. **预训练蒸馏模型在可接受的质量下提供显著加速：**
   - CausVid（DMD，3 步）：**6.4 倍加速**（28.5s vs 183s），视觉质量良好
   - rCM（一致性模型，4 步）：**4.9 倍加速**（37.6s vs 183s），视觉质量良好
   - 两种方法都能生成连贯、可观看的视频——相比 50 步 teacher 的质量损失适中，对多数应用可能是可接受的

3. **显存是首要限制：** Wan2.1 视频模型的大 latent 维度（[16, 21, 60, 104]）加上多网络需求，使 DMD2/f-distill/LADD 在 32GB GPU 上不可行。只有 ≤ 2 个网络的基于一致性的方法（ECT、CD）能在 32GB 上运行。

4. **从零自训练具有挑战性：** 从头训练 6000/1500 次迭代的 ECT 和 CD 产生了非常差的质量，而已发表论文的预训练模型（CausVid、rCM）经过全规模训练取得了优秀结果。这突显了正确超参数调优、训练规模和更好训练数据的重要性。

5. **RTX 5090 存在兼容性缺口：** Flash attention 库和自定义 CUDA 扩展（如 TurboDiffusion 的量化算子）尚不支持 Blackwell 架构（SM 12.0）。纯 PyTorch 回退可以工作但可能牺牲性能。Phase 1 需要持续关注此问题。

### 7.2 Phase 1 方法推荐

基于 Phase 0 的发现，对于渐进蒸馏（50 → 16 → 8 → 4 步）：

| 优先级 | 方法 | 理由 |
|--------|------|------|
| **首选** | DMD（CausVid 风格） | Phase 0 实验中最佳加速比（6.4x）且质量良好；NVIDIA 的主要方法 |
| **次选** | rCM（一致性模型） | 4.9x 加速，质量良好；NVlabs 有 Wan2.1 的预训练检查点 |
| **备选** | ECT | 显存最低，不依赖 teacher；在显存受限时有用 |
| **如有 80GB GPU** | DMD2（完整 FastGen） | NVIDIA 的完整流水线，VSD + GAN 损失（VBench 84.72） |

### 7.3 Phase 1 硬件推荐

| 场景 | 模型 | 方法 | 最低 GPU 配置 | 推荐 GPU 配置 |
|------|------|------|-------------|-------------|
| 1.3B + ECT | Wan2.1-1.3B | ECT | 4x RTX 5090 32GB | 4x RTX 5090 32GB |
| 1.3B + CD | Wan2.1-1.3B | CD | 4x RTX 5090 32GB | 4x A100 40GB |
| 1.3B + DMD2 | Wan2.1-1.3B | DMD2 | 4x A100 40GB | 4x A100 80GB |
| 14B + 任意方法 | Wan2.1-14B | 任意 | 8x A100 80GB | 8x H100 80GB |

---

## 附录

### A. 环境与版本

| 包 | 版本 |
|----|------|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| FastGen | 0.1.0 |
| Diffusers | 0.35.1 |
| Transformers | 4.49.0 |

### B. 服务器配置

| 项目 | 值 |
|------|-----|
| GPU | 8x NVIDIA RTX 5090 32GB |
| CPU | 384 核 |
| 内存 | 1TB |
| 操作系统 | Ubuntu, Linux 5.15.0 |

### C. 使用的预训练检查点

| 模型 | 来源 | 服务器路径 |
|------|------|-----------|
| Wan2.1-T2V-1.3B (Diffusers) | HuggingFace | `/data/chenqingzhan/.cache/huggingface/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/` |
| Wan2.1-T2V-1.3B (原始) | HuggingFace | `/data/chenqingzhan/CausVid/wan_models/Wan2.1-T2V-1.3B/` |
| CausVid DMD 检查点 | `tianweiy/CausVid` | `/data/chenqingzhan/causvid_checkpoints/bidirectional_checkpoint2/model.pt` |
| rCM 检查点 | `NVlabs/rcm` | `/data/chenqingzhan/rcm/assets/checkpoints/rCM_Wan2.1_T2V_1.3B_480p.pt` |
| TurboDiffusion 检查点 | `thu-ml/TurboDiffusion` | `/data/chenqingzhan/TurboDiffusion/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth` |
| ECT（自训练, 6000 iter） | 自训练 | `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_ct/ect_wan1.3b_4gpu/checkpoints/0006000` |
| CD（部分, 1500 iter） | 自训练 | `/data/chenqingzhan/fastgen_output/fastgen/wan_cm_cd/cd_wan1.3b_2gpu/checkpoints/0001500` |

### D. RTX 5090 兼容性补丁

| 组件 | 问题 | 修复方案 |
|------|------|---------|
| CausVid `attention.py` | Blackwell 上无 `flash_attn` | 修补为使用 PyTorch `F.scaled_dot_product_attention` (SDPA) 回退 |
| rCM `wan2pt1.py` | `flash_apply_rotary_emb` 使用不支持的 `interleaved` 参数 | 替换为纯 PyTorch 旋转嵌入（cos/sin 交错模式） |
| TurboDiffusion `modify_model.py` | `ops` CUDA 扩展（INT8 kernel）无法为 SM 12.0 编译 | **未解决**——TurboDiffusion 需要自定义 CUDA 算子；RTX 5090 上无法推理 |

### E. 参考文献

- [FastGen GitHub](https://github.com/NVlabs/FastGen) — NVIDIA 蒸馏框架
- [CausVid](https://github.com/tianweiy/CausVid) — 基于 DMD 的视频蒸馏 (CVPR 2025)
- [rCM](https://github.com/NVlabs/rcm) — Reflow 一致性模型 (ICLR 2026)
- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) — rCM + 量化优化
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — 基础视频生成模型
- [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) — 公开视频-文本数据集
- Song & Dhariwal 2023 — "Improved Techniques for Training Consistency Models" (ECT)
- Song et al. 2023 — "Consistency Models" (CD)
- Yin et al. 2024 — "Improved Distribution Matching Distillation" (DMD2)
