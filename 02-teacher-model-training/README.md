# PVTT教师模型训练技术综述

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [视频生成基础模型](#2-视频生成基础模型)
3. [扩散模型基础](#3-扩散模型基础)
4. [参考图像条件注入](#4-参考图像条件注入)
5. [LoRA微调技术](#5-lora微调技术)
6. [视频编辑与目标替换](#6-视频编辑与目标替换)
7. [训练框架与工程实践](#7-训练框架与工程实践)
8. [视频一致性与时序建模](#8-视频一致性与时序建模)
9. [评估指标体系](#9-评估指标体系)
10. [PVTT教师模型实施建议](#10-pvtt教师模型实施建议)
11. [参考文献](#11-参考文献)

---

## 1. 项目背景与目标

### 1.1 PVTT项目简介

PVTT (Product Video Template Transfer) 是面向电商场景的产品视频智能编辑系统。教师模型是整个系统的核心组件，负责生成高质量的产品替换视频。

### 1.2 教师模型技术路线

```
基础模型: Wan2.1/Wan2.2 (阿里巴巴DiT视频生成模型, 1.3B/14B)
    ↓
参考图像条件注入: IP-Adapter / ReferenceNet → 交叉注意力注入产品外观
    ↓
LoRA微调: 时序自注意力 + 交叉注意力层
    ↓
渐进式训练: 1.3B模型 → 14B模型
    ↓
训练基础设施: DiffSynth-Studio + DeepSpeed ZeRO-2/3
```

### 1.3 目标指标

| 指标 | 目标值 |
|------|--------|
| CLIP-I (身份保持) | > 0.85 |
| FVD (视频质量) | < 100 |
| 时序一致性 | > 0.90 |
| BG PSNR (背景保持) | > 30dB |

---

## 2. 视频生成基础模型

### 2.1 Wan2.1 / Wan2.2

| 字段 | 信息 |
|------|------|
| **论文** | Wan: Open and Advanced Large-Scale Video Generative Models |
| **arXiv** | [2503.20314](https://arxiv.org/abs/2503.20314) |
| **机构** | 阿里巴巴通义实验室 |
| **代码** | [github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| **规模** | 1.3B / 14B参数 |

#### 架构概述

Wan2.1基于Diffusion Transformer (DiT) 架构，是目前最先进的开源视频生成模型之一。

**核心组件**:

1. **3D VAE编码器/解码器**
   - 将视频压缩到潜在空间
   - 时空压缩比: 4×8×8 (时间×高度×宽度)
   - 通道数: 16
   - 支持任意分辨率和长度的视频

2. **DiT骨干网络**
   - 基于Transformer架构的扩散模型
   - 1.3B版本: 适合快速原型和微调
   - 14B版本: 最高质量生成
   - 支持文本到视频 (T2V) 和图像到视频 (I2V)

3. **文本编码器**
   - 多语言T5编码器
   - 支持中英文提示

4. **时序建模**
   - 3D Full Attention: 完全时空注意力（14B版本）
   - 因果注意力: 用于自回归扩展长视频

#### 模型变体

| 变体 | 参数量 | 分辨率 | 帧数 | 用途 |
|------|--------|--------|------|------|
| Wan2.1-T2V-1.3B | 1.3B | 480P | 81帧 | 快速原型 |
| Wan2.1-T2V-14B | 14B | 720P | 81帧 | 高质量T2V |
| Wan2.1-I2V-14B | 14B | 720P | 81帧 | 图像驱动 |
| Wan2.1-VACE-14B | 14B | 720P | 81帧 | 视频编辑 |

#### VACE扩展

| 字段 | 信息 |
|------|------|
| **论文** | VACE: All-in-One Video Creation and Editing |
| **arXiv** | [2503.07598](https://arxiv.org/abs/2503.07598) |
| **会议** | ICCV 2025 |
| **代码** | [github.com/ali-vilab/VACE](https://github.com/ali-vilab/VACE) |

VACE是Wan2.1的原生视频编辑框架:

- **视频条件单元 (VCU)**: 统一输入 V = [T; F; M] (文本 + 上下文帧 + 时空掩码)
- **上下文适配器**: Res-Tuning方法 + 分布式Transformer块
- **概念解耦**: 活跃帧 F_c = F × M, 非活跃帧 F_k = F × (1-M)
- **五类任务**: T2V, Reference-to-Video, V2V, Masked V2V, 任务组合

**与PVTT的关联**: VACE定义了Wan2.1/2.2的编辑范式，PVTT的教师模型应基于VACE架构扩展参考图像条件注入能力。

### 2.2 CogVideoX

| 字段 | 信息 |
|------|------|
| **论文** | CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer |
| **arXiv** | [2408.06072](https://arxiv.org/abs/2408.06072) |
| **机构** | 智谱AI (Zhipu AI / THU) |
| **代码** | [github.com/THUDM/CogVideo](https://github.com/THUDM/CogVideo) |
| **规模** | 2B / 5B参数 |

#### 核心架构

- **Expert Transformer**: 专家级3D Transformer，视频和文本token通过全注意力层交互
- **3D VAE**: 空间压缩8×, 时间压缩4×
- **Expert Adaptive LayerNorm**: 视觉和文本token使用不同的LayerNorm参数
- **3D RoPE**: 三维旋转位置编码（时间×高度×宽度）

#### 模型变体

| 变体 | 分辨率 | 帧数 | 长度 |
|------|--------|------|------|
| CogVideoX-2B | 720×480 | 49帧 | 6秒 |
| CogVideoX-5B | 720×480 | 49帧 | 6秒 |
| CogVideoX-5B-I2V | 720×480 | 49帧 | 图像驱动 |

**与PVTT的关联**: CogVideoX的Expert Transformer和3D RoPE设计可作为参考，但Wan2.1/2.2在规模和质量上更优。VideoPainter和GenCompositor均基于CogVideoX骨干。

### 2.3 Open-Sora系列

| 字段 | 信息 |
|------|------|
| **项目** | Open-Sora / Open-Sora-Plan |
| **代码** | [github.com/hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| **机构** | HPC-AI Tech / PKU |

#### 核心特点

- **STDiT (Spatial-Temporal DiT)**: 空间-时序分解的DiT架构
- **Open-Sora 1.2**: 支持2s-16s视频, 分辨率144p-720p
- **完全开源**: 模型权重、训练代码、数据处理管线全部开源
- **多阶段训练**: 图像预训练 → 短视频 → 长视频 → 高分辨率

**与PVTT的关联**: Open-Sora的STDiT空间-时序分解策略可用于理解不同注意力方案的权衡。其开源训练管线可作为工程参考。

### 2.4 Stable Video Diffusion (SVD)

| 字段 | 信息 |
|------|------|
| **论文** | Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets |
| **arXiv** | [2311.15127](https://arxiv.org/abs/2311.15127) |
| **机构** | Stability AI |

#### 核心特点

- **基于SD2.1**: 从图像扩散模型微调而来
- **时序注意力**: 在U-Net中插入时序注意力层
- **三阶段训练**: 图像预训练 → 视频预训练 (LVD-F) → 视频微调
- **数据策展**: 580M片段 → 152M策展片段 (74%拒绝率)

**与PVTT的关联**: SVD的图像→视频微调范式和数据策展方法是重要参考。

### 2.5 DiT架构深度分析

#### Diffusion Transformer (DiT) 核心论文

| 字段 | 信息 |
|------|------|
| **论文** | Scalable Diffusion Models with Transformers |
| **arXiv** | [2212.09748](https://arxiv.org/abs/2212.09748) |
| **会议** | ICCV 2023 |
| **作者** | William Peebles, Saining Xie |

#### DiT vs U-Net

| 特性 | U-Net | DiT |
|------|-------|-----|
| 下采样/上采样 | 有 (编码器-解码器) | 无 (等分辨率) |
| 注意力范围 | 局部 → 全局 | 全局自注意力 |
| 条件注入 | 交叉注意力 + AdaGN | adaLN-Zero |
| 缩放性 | 有限 | 优秀 (遵循scaling law) |
| 参数效率 | 中等 | 高 |
| 计算复杂度 | O(N²) per resolution | O(N²) 全局 |

#### adaLN-Zero条件注入

DiT使用Adaptive Layer Normalization with Zero-initialization:

```
h = γ(c) * LayerNorm(x) + β(c)
```

其中γ和β由条件嵌入c通过MLP生成，初始化为零（zero-init）。

#### 3D Full Attention vs 时空因子化注意力

**3D Full Attention** (Wan2.1-14B):
- 所有token（时间×高度×宽度）参与同一注意力计算
- 优点: 最强的时空建模能力
- 缺点: 计算复杂度 O(T²×H²×W²)

**时空因子化注意力** (STDiT):
- 空间注意力: 每帧内部 O(H²×W²)
- 时序注意力: 同位置跨帧 O(T²)
- 优点: 计算效率高
- 缺点: 时空交互不够充分

**Wan2.1的选择**: 14B模型使用3D Full Attention以最大化质量，1.3B模型使用因子化注意力以控制成本。

---

## 3. 扩散模型基础

### 3.1 DDPM (Denoising Diffusion Probabilistic Models)

| 字段 | 信息 |
|------|------|
| **论文** | Denoising Diffusion Probabilistic Models |
| **arXiv** | [2006.11239](https://arxiv.org/abs/2006.11239) |
| **会议** | NeurIPS 2020 |
| **作者** | Jonathan Ho, Ajay Jain, Pieter Abbeel |

#### 前向过程

逐步向数据添加高斯噪声:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)
```

等价于:
```
x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε, ε ~ N(0, I)
```

其中 ᾱ_t = ∏_{s=1}^{t} (1 - β_s)

#### 反向过程

学习从噪声恢复数据:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² * I)
```

#### 训练目标

简化的噪声预测损失:

```
L_simple = E_{x_0, ε, t}[||ε - ε_θ(x_t, t)||²]
```

### 3.2 DDIM (Denoising Diffusion Implicit Models)

| 字段 | 信息 |
|------|------|
| **论文** | Denoising Diffusion Implicit Models |
| **arXiv** | [2010.02502](https://arxiv.org/abs/2010.02502) |
| **会议** | ICLR 2021 |
| **作者** | Jiaming Song, Chenlin Meng, Stefano Ermon |

#### 核心创新

- **确定性采样**: 给定相同噪声，生成相同结果
- **加速采样**: 可跳过中间步骤 (如从1000步跳到50步)
- **隐式概率模型**: 非马尔可夫过程的一般化

#### 采样公式

```
x_{t-1} = √(ᾱ_{t-1}) * (x_t - √(1-ᾱ_t) * ε_θ(x_t, t)) / √(ᾱ_t)
         + √(1-ᾱ_{t-1}-σ_t²) * ε_θ(x_t, t) + σ_t * ε
```

当σ_t = 0时为完全确定性采样。

### 3.3 Flow Matching与Rectified Flow

| 字段 | 信息 |
|------|------|
| **论文** | Flow Matching for Generative Modeling |
| **arXiv** | [2210.02747](https://arxiv.org/abs/2210.02747) |
| **会议** | ICLR 2023 |

#### 核心思想

将生成建模转化为学习概率流ODE:

```
dx/dt = v_θ(x, t)
```

其中v_θ是学习的速度场，将噪声分布映射到数据分布。

#### Rectified Flow

| 字段 | 信息 |
|------|------|
| **论文** | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow |
| **arXiv** | [2209.03003](https://arxiv.org/abs/2209.03003) |
| **代码** | [github.com/gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) |

**核心改进**:
- **直线轨迹**: 通过reflow过程拉直概率流轨迹
- **最优传输**: 优化噪声与数据之间的耦合
- **训练目标**: L_RF = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
- **Wan2.1/2.2采用**: 基于Rectified Flow的训练范式

**与PVTT的关联**: Wan2.1使用Flow Matching训练范式。理解Rectified Flow对于正确实现教师模型训练和后续DMD蒸馏至关重要。

### 3.4 Score-Based Models

| 字段 | 信息 |
|------|------|
| **论文** | Score-Based Generative Modeling through Stochastic Differential Equations |
| **arXiv** | [2011.13456](https://arxiv.org/abs/2011.13456) |
| **会议** | ICLR 2021 (Outstanding Paper) |
| **作者** | Yang Song et al. |

#### 统一框架

用SDE统一DDPM和Score Matching:

```
dx = f(x, t)dt + g(t)dw  (前向SDE)
dx = [f(x, t) - g(t)² * ∇_x log p_t(x)]dt + g(t)dw̄  (反向SDE)
```

- **VP-SDE**: 方差保持 (对应DDPM)
- **VE-SDE**: 方差爆炸 (对应SMLD)
- **Sub-VP SDE**: 子方差保持

### 3.5 Classifier-Free Guidance (CFG)

| 字段 | 信息 |
|------|------|
| **论文** | Classifier-Free Diffusion Guidance |
| **arXiv** | [2207.12598](https://arxiv.org/abs/2207.12598) |

#### 核心公式

```
ε̃_θ(x_t, c) = (1 + w) * ε_θ(x_t, c) - w * ε_θ(x_t, ∅)
```

其中w是引导强度，c是条件（文本/图像），∅是无条件。

**训练方法**: 随机以概率p(通常10-20%)将条件替换为空条件∅，使模型同时学习条件和无条件生成。

**与PVTT的关联**: CFG是教师模型推理时提升质量的关键技术。需要在参考图像条件上也实现CFG。

---

## 4. 参考图像条件注入

### 4.1 IP-Adapter

| 字段 | 信息 |
|------|------|
| **论文** | IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models |
| **arXiv** | [2308.06721](https://arxiv.org/abs/2308.06721) |
| **代码** | [github.com/tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) |
| **机构** | Tencent AI Lab |

#### 核心架构

IP-Adapter通过解耦的交叉注意力机制将图像特征注入扩散模型:

```
Attention = Softmax(Q * K_text^T / √d) * V_text
          + Softmax(Q * K_image^T / √d) * V_image
```

**关键组件**:

1. **图像编码器**: 预训练CLIP图像编码器 (ViT-H/14)
2. **图像投影层**: 将CLIP特征投影到与文本特征相同的维度
3. **解耦交叉注意力**: 文本和图像分别通过独立的交叉注意力层，然后相加
4. **仅训练新增参数**: 冻结原始模型，仅训练图像投影层和交叉注意力的K/V投影

#### 训练细节

- **数据**: 约10M图文对 (LAION-2B子集)
- **训练**: 冻结文本编码器、图像编码器和U-Net
- **可训练参数**: ~22M (仅图像投影层和新增交叉注意力)
- **步数**: 1M steps, batch 8, lr=1e-4

#### 变体

**IP-Adapter-Plus**:
- 使用CLIP倒数第二层的patch token (而非仅CLS token)
- 通过Perceiver Resampler将257个token压缩为16个
- 保留更多细粒度视觉信息

**IP-Adapter-FaceID**:
- 使用InsightFace提取人脸特征
- 专为人脸身份保持设计

**与PVTT的关联**: IP-Adapter的解耦交叉注意力是PVTT参考图像注入的首选方案。使用产品图像的CLIP/DINOv2特征代替文本特征，通过新增交叉注意力层注入到Wan2.1的DiT骨干中。

### 4.2 ReferenceNet

| 字段 | 信息 |
|------|------|
| **代表工作** | Animate Anyone / MagicAnimate |
| **核心思想** | 使用完整的参考模型提取多尺度特征 |

#### 核心架构

- **参考分支**: 与主网络结构相同的完整网络副本
- **输入**: 参考图像 (产品图像)
- **特征注入**: 参考分支每层的自注意力K/V与主网络拼接
- **权重初始化**: 从主网络复制权重

#### 自注意力特征注入

```
K_combined = Concat(K_main, K_ref)
V_combined = Concat(V_main, V_ref)
Attention = Softmax(Q_main * K_combined^T / √d) * V_combined
```

#### 优缺点对比

| 特性 | IP-Adapter | ReferenceNet |
|------|-----------|-------------|
| 新增参数 | ~22M | ~模型等量 |
| 特征粒度 | 全局/中等 (CLIP) | 多尺度 (像素级) |
| 训练成本 | 低 | 高 |
| 推理开销 | 低 | 高 (双倍forward) |
| 身份保持 | 中等 | 强 |
| 兼容性 | 高 (LoRA友好) | 中等 |

**与PVTT的关联**: 对于电商产品，需要精确的外观保持。建议先尝试IP-Adapter-Plus（训练成本低），如果身份保持不够再升级到ReferenceNet。

### 4.3 InstantID

| 字段 | 信息 |
|------|------|
| **论文** | InstantID: Zero-shot Identity-Preserving Generation in Seconds |
| **arXiv** | [2401.07519](https://arxiv.org/abs/2401.07519) |
| **代码** | [github.com/InstantID/InstantID](https://github.com/InstantID/InstantID) |

#### 核心创新

- **双路径**: IP-Adapter (语义特征) + IdentityNet (空间特征)
- **IdentityNet**: 轻量级ControlNet变体，注入面部关键点空间信息
- **零样本**: 单张参考图即可生成

**与PVTT的关联**: 双路径设计（语义 + 空间）可参考 — 用CLIP特征捕获产品语义，用DINOv2 patch特征捕获产品空间细节。

### 4.4 Emu系列 (图像编辑中的参考注入)

| 字段 | 信息 |
|------|------|
| **论文** | Emu Edit: Precise Image Editing via Recognition and Generation Tasks |
| **arXiv** | [2311.10089](https://arxiv.org/abs/2311.10089) |
| **机构** | Meta AI |

- **多任务学习**: 训练16种编辑任务（包括目标替换）
- **学习条件注入**: 通过任务嵌入区分不同编辑类型
- **参考机制**: 源图像通过额外通道输入

### 4.5 视频特定的参考注入

#### ConsistI2V

| 字段 | 信息 |
|------|------|
| **论文** | ConsistI2V: Enhancing Visual Consistency for Image-to-Video Generation |
| **arXiv** | [2402.04324](https://arxiv.org/abs/2402.04324) |

- **帧级注意力**: 首帧作为参考，通过注意力传播外观到后续帧
- **时序稳定**: 避免生成过程中的风格漂移
- **自适应权重**: 根据帧距离调整参考影响力

#### ID-Animator

| 字段 | 信息 |
|------|------|
| **论文** | ID-Animator: Zero-Shot Identity-Preserving Human Video Generation |
| **arXiv** | [2404.15275](https://arxiv.org/abs/2404.15275) |

- **面部适配器**: 类IP-Adapter的面部特征注入
- **视频生成**: 将身份保持扩展到视频域
- **身份一致性**: 跨帧保持人物身份

---

## 5. LoRA微调技术

### 5.1 LoRA基础

| 字段 | 信息 |
|------|------|
| **论文** | LoRA: Low-Rank Adaptation of Large Language Models |
| **arXiv** | [2106.09685](https://arxiv.org/abs/2106.09685) |
| **会议** | ICLR 2022 |
| **作者** | Edward Hu et al. (Microsoft) |

#### 核心原理

低秩分解近似全参数微调:

```
W' = W + ΔW = W + B * A
```

其中 W ∈ R^{d×k}, B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d, k)

#### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| **rank (r)** | 低秩矩阵的秩 | 4-128 |
| **alpha (α)** | 缩放因子 | rank或2×rank |
| **dropout** | LoRA dropout | 0.0-0.1 |
| **目标层** | 应用LoRA的层 | attention Q/K/V/O |

#### 缩放机制

实际更新 = (α/r) × B × A

当α = r时，缩放因子为1（最常用）。

### 5.2 QLoRA

| 字段 | 信息 |
|------|------|
| **论文** | QLoRA: Efficient Finetuning of Quantized LLMs |
| **arXiv** | [2305.14314](https://arxiv.org/abs/2305.14314) |
| **会议** | NeurIPS 2023 |

#### 核心创新

1. **4-bit NormalFloat (NF4)**: 新的数据类型，对正态分布权重信息最优
2. **双重量化**: 对量化常数再次量化，减少内存
3. **分页优化器**: 防止长序列OOM

**内存节省**: 在单张48GB GPU上微调65B参数模型

### 5.3 DoRA

| 字段 | 信息 |
|------|------|
| **论文** | DoRA: Weight-Decomposed Low-Rank Adaptation |
| **arXiv** | [2402.09353](https://arxiv.org/abs/2402.09353) |
| **会议** | ICML 2024 |

#### 核心创新

将权重分解为幅度和方向分量:

```
W' = m * (W + B*A) / ||W + B*A||
```

其中m是可学习的幅度向量。

**优势**: 在相同rank下性能优于LoRA，更接近全参数微调。

### 5.4 视频扩散模型的LoRA策略

#### 应用层选择

对于Wan2.1 DiT架构:

```
DiT Block:
├── Self-Attention (时空)
│   ├── Q_proj  ← LoRA ✓
│   ├── K_proj  ← LoRA ✓
│   ├── V_proj  ← LoRA ✓
│   └── O_proj  ← LoRA ✓
├── Cross-Attention (文本/图像条件)
│   ├── Q_proj  ← LoRA ✓
│   ├── K_proj  ← LoRA ✓
│   ├── V_proj  ← LoRA ✓
│   └── O_proj  ← LoRA ✓
├── MLP/FFN
│   ├── fc1    ← LoRA (可选)
│   └── fc2    ← LoRA (可选)
└── adaLN (条件注入)
    └── MLP    ← 通常冻结
```

#### PVTT推荐策略

1. **时序自注意力 (Temporal Self-Attention)**:
   - 必须微调 — 学习产品运动模式
   - rank: 64-128
   - 微调所有Q/K/V/O投影

2. **交叉注意力 (Cross-Attention)**:
   - 必须微调 — 注入产品参考特征
   - rank: 64-128
   - 重点微调K/V投影（接受参考图像特征）

3. **空间自注意力 (Spatial Self-Attention)**:
   - 可选微调 — 学习产品外观细节
   - rank: 32-64

4. **MLP/FFN**:
   - 通常不微调
   - 除非需要强适应（如全新领域）

#### Rank选择指南

| 场景 | 推荐Rank | 参数量 |
|------|----------|--------|
| 轻量适应 | 16-32 | ~5M |
| 标准微调 | 64-128 | ~20-40M |
| 深度适应 | 256 | ~80M |
| InsertAnywhere参考 | 128 | ~40M |

### 5.5 LoRA合并与部署

#### 权重合并

```python
W_merged = W_pretrained + (alpha/rank) * B @ A
```

合并后无额外推理开销。

#### 多LoRA组合

- **LoRA Composer**: 多个LoRA线性组合
- **LoRA Switch**: 运行时动态切换LoRA

---

## 6. 视频编辑与目标替换

### 6.1 AnyV2V

| 字段 | 信息 |
|------|------|
| **论文** | AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks |
| **arXiv** | [2403.14468](https://arxiv.org/abs/2403.14468) |
| **代码** | [github.com/TIGER-AI-Lab/AnyV2V](https://github.com/TIGER-AI-Lab/AnyV2V) |

#### 核心方法

两阶段框架:
1. **首帧编辑**: 使用任意图像编辑模型编辑首帧
2. **视频传播**: 使用I2V模型将编辑效果传播到所有帧

**DDIM Inversion**: 对原始视频做DDIM反演获取噪声序列，然后以编辑后首帧为条件重新生成。

**与PVTT的关联**: AnyV2V的"首帧编辑 + 视频传播"范式与PVTT管线高度相关。

### 6.2 TokenFlow

| 字段 | 信息 |
|------|------|
| **论文** | TokenFlow: Consistent Diffusion Features for Consistent Video Editing |
| **arXiv** | [2307.10373](https://arxiv.org/abs/2307.10373) |
| **会议** | ICLR 2024 |

#### 核心方法

- **特征传播**: 在扩散模型的自注意力层中，用关键帧的token特征替换非关键帧
- **帧间对应**: 通过PnP features建立帧间对应关系
- **训练无关**: 无需微调，仅修改推理过程

### 6.3 FateZero

| 字段 | 信息 |
|------|------|
| **论文** | FateZero: Fusing Attentions for Zero-shot Text-based Video Editing |
| **arXiv** | [2303.09535](https://arxiv.org/abs/2303.09535) |
| **会议** | ICCV 2023 |

#### 核心方法

- **注意力融合**: 将DDIM Inversion过程中的自注意力/交叉注意力特征融合到编辑过程
- **注意力混合掩码**: 自动生成编辑区域掩码
- **零样本**: 无需逐视频训练

### 6.4 Rerender-a-Video

| 字段 | 信息 |
|------|------|
| **论文** | Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation |
| **arXiv** | [2306.07954](https://arxiv.org/abs/2306.07954) |

#### 核心方法

- **跨帧约束**: 利用光流实现跨帧像素对应
- **形状感知注意力**: 保持原始视频结构
- **时序感知**: 考虑帧间关系的生成过程

### 6.5 视频目标替换的关键技术

#### DDIM Inversion

将视频反向转换为噪声表示:

```
x_{t+1} = √(ᾱ_{t+1}) * (x_t - √(1-ᾱ_t) * ε_θ(x_t, t)) / √(ᾱ_t)
         + √(1-ᾱ_{t+1}) * ε_θ(x_t, t)
```

**挑战**: 多步反演会累积误差，导致重建不精确。

**解决方案**:
- **Null-text Inversion**: 优化null-text embedding补偿误差
- **Negative Prompt Inversion**: 用负提示词替代null-text优化

#### 注意力操控

- **Prompt-to-Prompt**: 修改交叉注意力图实现局部编辑
- **Self-Attention Injection**: 注入原始视频的自注意力特征保持结构

---

## 7. 训练框架与工程实践

### 7.1 DiffSynth-Studio

| 字段 | 信息 |
|------|------|
| **代码** | [github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) |
| **机构** | 阿里巴巴达摩院 / ModelScope |

#### 核心特点

- **Wan2.1/2.2原生支持**: 官方推荐的训练和推理框架
- **模块化设计**: 支持多种扩散模型（SD, SDXL, Wan, Flux等）
- **训练能力**: LoRA训练、全参数微调、DreamBooth
- **推理优化**: 支持多种采样器和加速策略

#### 训练配置示例

```python
# Wan2.1 LoRA训练
trainer = WanTrainer(
    model_path="Wan2.1-T2V-14B",
    lora_rank=128,
    lora_alpha=128,
    target_modules=["to_q", "to_k", "to_v", "to_out"],
    learning_rate=1e-4,
    batch_size=1,
    gradient_accumulation_steps=4,
    mixed_precision="bf16",
)
```

### 7.2 HuggingFace Diffusers

| 字段 | 信息 |
|------|------|
| **代码** | [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers) |
| **文档** | [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers) |

#### 关键能力

- **统一API**: Pipeline抽象覆盖主流模型
- **Wan2.1支持**: 内置Wan2.1 T2V/I2V Pipeline
- **LoRA集成**: `load_lora_weights()` / `fuse_lora()`
- **训练脚本**: 提供标准化训练脚本模板

#### 训练基础设施

```python
from diffusers import WanPipeline
from peft import LoraConfig

lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=["to_q", "to_k", "to_v", "to_out"],
    lora_dropout=0.0,
)
```

### 7.3 DeepSpeed

| 字段 | 信息 |
|------|------|
| **代码** | [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) |
| **机构** | Microsoft Research |

#### ZeRO优化阶段

| 阶段 | 分片内容 | 内存节省 | 通信开销 |
|------|----------|----------|----------|
| **ZeRO-1** | 优化器状态 | 4× | 无额外 |
| **ZeRO-2** | 优化器状态 + 梯度 | 8× | 低 |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | N× | 中 |

#### PVTT推荐配置

- **14B模型**: ZeRO-3 (参数量过大，需要全分片)
- **1.3B模型**: ZeRO-2 (梯度分片即可)
- **混合精度**: BF16 (适合Ampere及以上GPU)

#### DeepSpeed配置示例

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},
        "offload_optimizer": {"device": "cpu"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto"
    },
    "bf16": {"enabled": true},
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto"
}
```

### 7.4 FSDP (Fully Sharded Data Parallelism)

| 字段 | 信息 |
|------|------|
| **框架** | PyTorch原生 |
| **文档** | [pytorch.org/tutorials/intermediate/FSDP_tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) |

#### vs DeepSpeed ZeRO-3

| 特性 | FSDP | DeepSpeed ZeRO-3 |
|------|------|-------------------|
| 集成 | PyTorch原生 | 第三方库 |
| API | `FullyShardedDataParallel` | `deepspeed.initialize()` |
| CPU Offload | 支持 | 支持 |
| NVMe Offload | 不支持 | 支持 |
| 生态 | PyTorch原生 | 更成熟的大模型支持 |

### 7.5 混合精度训练

#### BF16 vs FP16

| 特性 | BF16 | FP16 |
|------|------|------|
| 指数位 | 8位 | 5位 |
| 尾数位 | 7位 | 10位 |
| 动态范围 | 与FP32相同 | 较小 |
| 精度 | 较低 | 较高 |
| 训练稳定性 | 更好 (无需loss scaling) | 需要loss scaling |
| 硬件要求 | Ampere+ (A100, H100) | 所有GPU |

**PVTT推荐**: BF16（Wan2.1默认训练精度）

### 7.6 Gradient Checkpointing

**原理**: 用重计算换内存，只保存部分中间激活值。

**效果**:
- 内存节省: ~60-70%
- 速度开销: ~20-30%

**应用**: 对于14B参数的Wan2.1，gradient checkpointing几乎是必须的。

### 7.7 工程最佳实践

#### 数据加载优化

```
1. 预处理: 将视频预编码为潜在表示 (VAE encoding)
2. 缓存: 预计算文本/图像嵌入
3. 多Worker: num_workers=4-8, pin_memory=True
4. 预取: prefetch_factor=2
```

#### 训练监控

```
关键指标:
├── 损失曲线 (总损失 + 各分量)
├── 学习率曲线
├── 梯度范数
├── 显存使用
├── 吞吐量 (samples/sec)
└── 定期可视化 (每N步生成样本)
```

#### Checkpointing策略

```
├── 每N步保存完整checkpoint
├── 保留最近K个checkpoint
├── 每个epoch保存一次
└── 基于验证指标保存best checkpoint
```

---

## 8. 视频一致性与时序建模

### 8.1 时序注意力机制

#### Temporal Self-Attention

在同一空间位置的不同时间帧之间计算注意力:

```
Attention_temporal = Softmax(Q_t * K_t^T / √d) * V_t
```

其中Q_t, K_t, V_t来自同一空间位置的不同帧。

#### 3D Full Attention (Wan2.1-14B)

所有时空位置参与同一注意力计算:

```
Tokens = [t1_h1w1, t1_h1w2, ..., t1_HHWW, t2_h1w1, ..., tT_HHWW]
Attention_3D = Softmax(Q * K^T / √d) * V
```

**复杂度**: O((T×H×W)²)

### 8.2 AnimateDiff

| 字段 | 信息 |
|------|------|
| **论文** | AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning |
| **arXiv** | [2307.04725](https://arxiv.org/abs/2307.04725) |
| **代码** | [github.com/guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) |

#### 运动模块 (Motion Module)

- **结构**: 时序Transformer块，插入到每个U-Net/DiT块的空间注意力之后
- **输入**: 将空间特征重排为 (B×H×W, T, C)
- **位置编码**: 正弦位置编码 + 可学习位置偏置
- **训练**: 仅训练运动模块，冻结空间层

#### 运动模块变体

- **AnimateDiff v1**: 基础时序Transformer
- **AnimateDiff v2**: 改进的运动模块 + 运动LoRA
- **AnimateDiff v3**: 支持跨域动画

**与PVTT的关联**: AnimateDiff的运动模块设计思路可参考，但Wan2.1的DiT架构已原生支持3D Full Attention，无需额外添加运动模块。

### 8.3 视频一致性保持策略

#### 帧间注意力传播

```
方法1: 关键帧传播
  - 选取第1帧作为关键帧
  - 后续帧的K/V包含关键帧信息

方法2: 滑动窗口
  - 每帧关注前N帧 (如N=4)
  - 保持局部一致性

方法3: 全局记忆
  - 类似SAM2的记忆机制
  - 维持全局外观一致性
```

#### 光流引导

- **RAFT**: 预计算帧间光流
- **扭曲约束**: 使用光流将上一帧扭曲到当前帧，约束一致性
- **运动嵌入**: 将光流作为额外条件注入

#### 时序平滑

- **帧间插值**: 对潜在表示做时序插值平滑
- **时域卷积**: 在潜在空间应用1D时域卷积
- **动量更新**: 当前帧 = α × 上一帧 + (1-α) × 预测帧

---

## 9. 评估指标体系

### 9.1 身份保持指标

#### CLIP-I (图像相似度)

```
CLIP-I = cos(CLIP_image(reference), CLIP_image(generated))
```

- 度量语义层面的身份相似度
- 阈值建议: > 0.85
- 工具: OpenCLIP (ViT-H/14)

#### DINO-I (细粒度相似度)

```
DINO-I = cos(DINOv2(reference), DINOv2(generated))
```

- 捕获更细粒度的视觉结构
- 对产品纹理、标识更敏感
- 准确率是CLIP-I的2.25倍

### 9.2 视频质量指标

#### FVD (Fréchet Video Distance)

```
FVD = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real Σ_gen)^{0.5})
```

特征来自I3D模型。FVD越低越好。
- 目标: FVD < 100

#### FID (逐帧)

对每帧独立计算FID，评估单帧图像质量。

### 9.3 时序一致性指标

#### 帧间相似度

```
Temporal_Consistency = mean(SSIM(frame_t, frame_{t+1}))
```

- 目标: > 0.90

#### 时序扭曲误差 (TWE)

```
TWE = mean(||Warp(frame_{t+1}, flow_{t→t+1}) - frame_t||)
```

越低越好。

### 9.4 背景保持指标

#### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 * log10(MAX² / MSE)
```

仅在背景区域（非产品区域）计算。
- 目标: > 30dB

#### SSIM (Structural Similarity Index)

```
SSIM(x, y) = (2μ_xμ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
```

仅在背景区域计算。

#### LPIPS (Learned Perceptual Image Patch Similarity)

越低越好。基于深度特征的感知差异。

### 9.5 文本对齐指标

#### CLIP-T (文本-视频对齐)

```
CLIP-T = cos(CLIP_text(prompt), CLIP_image(frame))
```

评估生成视频与文本描述的一致性。

### 9.6 综合评估方案

**PVTT教师模型评估矩阵**:

| 维度 | 指标 | 目标 | 权重 |
|------|------|------|------|
| 产品身份 | CLIP-I | > 0.85 | 30% |
| 产品身份 | DINO-I | > 0.50 | 20% |
| 视频质量 | FVD | < 100 | 15% |
| 时序一致性 | Frame Consistency | > 0.90 | 15% |
| 背景保持 | BG-PSNR | > 30dB | 10% |
| 背景保持 | BG-SSIM | > 0.90 | 5% |
| 文本对齐 | CLIP-T | > 0.25 | 5% |

---

## 10. PVTT教师模型实施建议

### 10.1 渐进式训练路线

#### 第一阶段: 1.3B模型快速验证

```
目标: 验证管线可行性
模型: Wan2.1-T2V-1.3B
微调: LoRA rank=64 on temporal + cross attention
数据: 10K-50K训练对
硬件: 4×A100 80GB
时间: 1-2周
验收: CLIP-I > 0.75, 基本视频连贯
```

#### 第二阶段: 14B模型基础训练

```
目标: 建立高质量基线
模型: Wan2.1-T2V-14B / VACE-14B
微调: LoRA rank=128 on temporal + cross attention
数据: 100K-300K训练对
硬件: 8×A100 80GB, DeepSpeed ZeRO-3
时间: 4-6周
验收: CLIP-I > 0.80, FVD < 150
```

#### 第三阶段: 14B模型精细调优

```
目标: 达到目标质量
模型: 第二阶段checkpoint
策略: 降低学习率, 增加高质量数据比例
数据: 精选50K-100K高质量数据
硬件: 8×A100 80GB
时间: 2-4周
验收: CLIP-I > 0.85, FVD < 100, BG-PSNR > 30dB
```

### 10.2 参考图像注入方案

#### 推荐方案: IP-Adapter-Plus变体

```
产品图像
    ↓
DINOv2 ViT-L/14 编码器 (冻结)
    ↓
Patch tokens (256×1024)
    ↓
Perceiver Resampler → 16个token
    ↓
解耦交叉注意力注入 DiT每层
```

**训练策略**:
1. 冻结: DiT骨干 + DINOv2编码器
2. 可训练: Perceiver Resampler + 新增交叉注意力K/V投影 + LoRA
3. CFG: 20%概率丢弃参考图像条件

#### 备选方案: ReferenceNet

如果IP-Adapter-Plus身份保持不足:
- 使用DiT的前50%层作为参考分支
- 自注意力K/V拼接
- 训练成本增加~2×

### 10.3 LoRA微调配置

```python
# PVTT教师模型LoRA配置
lora_config = {
    "rank": 128,
    "alpha": 128,
    "dropout": 0.05,
    "target_modules": [
        # 时序自注意力
        "temporal_self_attn.to_q",
        "temporal_self_attn.to_k",
        "temporal_self_attn.to_v",
        "temporal_self_attn.to_out",
        # 交叉注意力 (文本+图像)
        "cross_attn.to_q",
        "cross_attn.to_k",
        "cross_attn.to_v",
        "cross_attn.to_out",
    ],
}
```

### 10.4 训练超参数

| 参数 | 推荐值 |
|------|--------|
| 学习率 | 1e-4 (warmup 1000 steps) |
| 优化器 | AdamW (β1=0.9, β2=0.999) |
| Weight Decay | 0.01 |
| Batch Size | 1 per GPU × 8 GPU × 4 grad accum = 32 |
| 混合精度 | BF16 |
| Gradient Clip | 1.0 |
| 采样器 | Rectified Flow |
| CFG Scale (推理) | 7.0-10.0 |
| 步数 (推理) | 50步 DDIM/DPM++ |

### 10.5 训练损失函数

```
L_total = L_diffusion                    # 扩散损失 (v-prediction或noise prediction)
        + λ_1 * L_reference_recon        # 参考图像重建
        + λ_2 * L_identity               # 身份保持 (CLIP-I/DINO-I)
```

**L_diffusion**: 标准扩散训练损失
```
L_diffusion = E[||v_θ(x_t, t, c) - v_target||²]
```

**L_reference_recon** (可选): 确保参考图像特征被正确编码
**L_identity** (可选): 在训练中直接优化身份保持

### 10.6 数据准备

#### 训练数据格式

```json
{
    "video_path": "path/to/video.mp4",
    "reference_image": "path/to/product.jpg",
    "mask_path": "path/to/mask.mp4",
    "text_prompt": "A video of [product] on a white table",
    "metadata": {
        "resolution": [720, 480],
        "fps": 16,
        "num_frames": 81
    }
}
```

#### 数据增强

1. **参考图像增强**: 随机裁剪、颜色抖动、水平翻转
2. **视频增强**: 随机时间偏移、帧率抖动
3. **掩码增强**: 随机膨胀/侵蚀

### 10.7 工程清单

```
□ 安装DiffSynth-Studio / Diffusers
□ 下载Wan2.1-T2V-14B / VACE-14B权重
□ 配置DeepSpeed ZeRO-3
□ 实现参考图像编码器 (DINOv2 + Perceiver)
□ 实现解耦交叉注意力
□ 数据加载器 (视频 + 参考图像 + 掩码)
□ 训练循环 (LoRA on temporal + cross attention)
□ 评估管线 (CLIP-I, FVD, BG-PSNR)
□ 可视化工具 (TensorBoard / WandB)
□ Checkpoint管理
```

---

## 11. 参考文献

### 视频生成基础模型
1. **Wan2.1**: "Wan: Open and Advanced Large-Scale Video Generative Models" — https://arxiv.org/abs/2503.20314 — https://github.com/Wan-Video/Wan2.1
2. **VACE**: "VACE: All-in-One Video Creation and Editing", ICCV 2025 — https://arxiv.org/abs/2503.07598
3. **CogVideoX**: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" — https://arxiv.org/abs/2408.06072
4. **Open-Sora**: https://github.com/hpcaitech/Open-Sora
5. **SVD**: "Stable Video Diffusion" — https://arxiv.org/abs/2311.15127
6. **DiT**: "Scalable Diffusion Models with Transformers", ICCV 2023 — https://arxiv.org/abs/2212.09748

### 扩散模型基础
7. **DDPM**: Ho et al., NeurIPS 2020 — https://arxiv.org/abs/2006.11239
8. **DDIM**: Song et al., ICLR 2021 — https://arxiv.org/abs/2010.02502
9. **Flow Matching**: Lipman et al., ICLR 2023 — https://arxiv.org/abs/2210.02747
10. **Rectified Flow**: Liu et al. — https://arxiv.org/abs/2209.03003
11. **Score SDE**: Song et al., ICLR 2021 — https://arxiv.org/abs/2011.13456
12. **CFG**: Ho & Salimans — https://arxiv.org/abs/2207.12598

### 参考图像条件注入
13. **IP-Adapter**: Ye et al. — https://arxiv.org/abs/2308.06721 — https://github.com/tencent-ailab/IP-Adapter
14. **InstantID**: Wang et al. — https://arxiv.org/abs/2401.07519
15. **Emu Edit**: Meta AI — https://arxiv.org/abs/2311.10089
16. **ConsistI2V**: — https://arxiv.org/abs/2402.04324
17. **ID-Animator**: — https://arxiv.org/abs/2404.15275

### LoRA微调
18. **LoRA**: Hu et al., ICLR 2022 — https://arxiv.org/abs/2106.09685
19. **QLoRA**: Dettmers et al., NeurIPS 2023 — https://arxiv.org/abs/2305.14314
20. **DoRA**: Liu et al., ICML 2024 — https://arxiv.org/abs/2402.09353

### 视频编辑
21. **AnyV2V**: Ku et al. — https://arxiv.org/abs/2403.14468
22. **TokenFlow**: Geyer et al., ICLR 2024 — https://arxiv.org/abs/2307.10373
23. **FateZero**: Qi et al., ICCV 2023 — https://arxiv.org/abs/2303.09535
24. **Rerender-a-Video**: Yang et al. — https://arxiv.org/abs/2306.07954

### 训练框架
25. **DiffSynth-Studio**: https://github.com/modelscope/DiffSynth-Studio
26. **HuggingFace Diffusers**: https://github.com/huggingface/diffusers
27. **DeepSpeed**: https://github.com/microsoft/DeepSpeed

### 时序建模
28. **AnimateDiff**: Guo et al. — https://arxiv.org/abs/2307.04725

### 评估
29. **FVD**: Unterthiner et al. — https://arxiv.org/abs/1812.01717
30. **CLIP**: Radford et al. — https://arxiv.org/abs/2103.00020
31. **DINOv2**: Oquab et al. — https://arxiv.org/abs/2304.07193

---

**文档版本**: v1.0
**最后更新**: 2026-02-11
**编写**: Claude Code (基于2024-2026最新研究文献)
