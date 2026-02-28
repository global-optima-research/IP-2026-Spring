# Distribution Matching Distillation (DMD) 研究综述

> **作者**: 调研整理
> **日期**: 2025年2月
> **关键词**: 扩散模型蒸馏、DMD、模式崩溃、多样性保持、快速图像生成

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [DMD 系列方法](#2-dmd-系列方法)
   - [2.1 DMD: 原始方法](#21-dmd-原始方法)
   - [2.2 DMD2: 改进版本](#22-dmd2-改进版本)
   - [2.3 DP-DMD: 多样性保持](#23-dp-dmd-多样性保持)
   - [2.4 Phased DMD: 分阶段蒸馏](#24-phased-dmd-分阶段蒸馏)
   - [2.5 ADM: 对抗式分布匹配](#25-adm-对抗式分布匹配)
   - [2.6 DMDR: 结合强化学习](#26-dmdr-结合强化学习)
3. [DMD 的局限性与风险分析](#3-dmd-的局限性与风险分析)
   - [3.1 模式崩溃问题](#31-模式崩溃问题)
   - [3.2 训练不稳定性](#32-训练不稳定性)
   - [3.3 数据集构建开销](#33-数据集构建开销)
   - [3.4 质量-多样性权衡](#34-质量-多样性权衡)
4. [解决方案分类与对比](#4-解决方案分类与对比)
5. [其他相关蒸馏方法](#5-其他相关蒸馏方法)
6. [研究趋势与展望](#6-研究趋势与展望)
7. [参考文献](#7-参考文献)

---

## 1. 背景与动机

### 1.1 扩散模型的推理瓶颈

扩散模型（Diffusion Models）在图像生成领域取得了显著成功，但其 **多步采样过程** 导致推理速度缓慢：

| 模型 | 采样步数 | 生成时间 |
|------|---------|---------|
| Stable Diffusion v1.5 | 50 步 | ~3-5 秒/张 |
| DDPM | 1000 步 | ~分钟级 |
| 单步蒸馏模型 | 1-4 步 | ~50-100 毫秒/张 |

### 1.2 蒸馏的核心目标

将多步教师模型压缩为少步（甚至单步）学生模型，同时保持：
- **生成质量**（FID、IS 等指标）
- **生成多样性**（不同噪声输入产生差异化输出）
- **Prompt 对齐**（文本-图像一致性）

---

## 2. DMD 系列方法

### 2.1 DMD: 原始方法

**论文**: *One-step Diffusion with Distribution Matching Distillation* (CVPR 2024)

#### 核心思想

DMD 通过最小化学生分布与教师分布之间的 **逆向 KL 散度** 来训练单步生成器：

$$\mathcal{L}_{\text{DMD}} = D_{\text{KL}}(p_{\text{fake}}(\bm{x}_\theta) \| p_{\text{real}}(\bm{x}_\theta))$$

其梯度可表示为两个 score function 的差：

$$\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}\left[(s_{\text{fake}}(\bm{z}_t) - s_{\text{real}}(\bm{z}_t)) \nabla_\theta \bm{x}_\theta\right]$$

#### 训练流程

```
1. 预训练扩散模型作为 "真实分布" score 估计器
2. 持续训练另一扩散模型估计 "生成分布" score
3. 两者差值作为梯度方向，指导单步生成器优化
4. 额外使用回归损失（LPIPS）匹配教师多步输出
```

#### 性能指标

| 数据集 | FID | 推理速度 |
|--------|-----|---------|
| ImageNet 64×64 | 2.62 | 20 FPS (FP16) |
| COCO-30k (zero-shot) | 11.49 | 30× faster than SD v1.5 |

---

### 2.2 DMD2: 改进版本

**论文**: *Improved Distribution Matching Distillation for Fast Image Synthesis* (NeurIPS 2024 Oral)

#### 解决的问题

| 原始 DMD 问题 | DMD2 解决方案 |
|--------------|--------------|
| 昂贵的数据集构建（需要教师生成大量噪声-图像对） | 移除回归损失，使用两时间尺度更新规则 |
| Fake critic 估计不准确导致训练不稳定 | Two time-scale update rule |
| 学生质量受限于教师采样路径 | 引入 GAN 损失，在真实数据上训练 |
| 仅支持单步生成 | 训练时模拟推理时的多步采样 |

#### 核心创新

**1. 两时间尺度更新规则**
```python
# Fake critic 更新频率更高
for _ in range(k):  # k > 1
    update_fake_critic()
update_real_critic()
update_generator()
```

**2. GAN 损失集成**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DMD}} + \lambda_{\text{GAN}} \mathcal{L}_{\text{GAN}}$$

#### 性能提升

| 数据集 | DMD FID | DMD2 FID | 提升 |
|--------|---------|----------|------|
| ImageNet-64×64 | 2.62 | **1.28** | 51% |
| COCO-2014 (zero-shot) | 11.49 | **8.35** | 27% |

---

### 2.3 DP-DMD: 多样性保持

**论文**: *Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis* (arXiv 2602.03139)

#### 核心洞察

不同去噪步骤承担不同角色：

| 步骤 | 功能 | 对多样性的影响 |
|------|------|---------------|
| 早期步骤 | 决定全局结构布局 | **关键** |
| 后期步骤 | 细化视觉细节 | 次要 |

#### 技术方案：角色分离 + 梯度阻断

```
噪声 ε → [第一步: Flow Matching 损失] → detach → [后续步骤: DMD 损失] → 输出
         ↑                                        ↑
    保持多样性                                 优化质量
```

**损失函数设计**:
$$\mathcal{L} = \mathcal{L}_{\text{DMD}} + \lambda \mathcal{L}_{\text{Div}}$$

**多样性损失** (Flow Matching):
$$\mathcal{L}_{\text{Div}} = \mathbb{E}_\epsilon\left[\|\bm{v}_\theta(\epsilon,1) - \bm{v}_k^{\text{target}}\|^2\right]$$

#### 优势

- **无额外网络**: 不需要感知网络、判别器或辅助模型
- **计算高效**: 仅在损失计算和梯度传播上做修改
- **效果显著**: 在 DINO/CLIP 多样性指标上明显优于 DMD、DMD-LPIPS、DMD-GAN

---

### 2.4 Phased DMD: 分阶段蒸馏

**论文**: *Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals* (arXiv 2510.27684)

#### 核心洞察

现有 SGTS（Self-Guided Training Scheme）方法会将多步生成器的多样性退化到单步水平。

#### 技术方案

**SNR 分段训练**:
```
完整 SNR 范围: [0, 1]
     ↓ 分割
Phase 1: [0, 0.3]  → Expert 1 (粗结构)
Phase 2: [0.3, 0.7] → Expert 2 (中间层)
Phase 3: [0.7, 1.0] → Expert 3 (细节)
```

**渐进式训练**: 冻结低 SNR 专家，仅训练高 SNR 专家

#### 多样性保持效果

| 方法 | DINOv3 相似度 ↓ | LPIPS 距离 ↑ |
|------|----------------|--------------|
| DMD + SGTS | 0.826 | 0.521 |
| **Phased DMD** | **0.782** | **0.544** |

---

### 2.5 ADM: 对抗式分布匹配

**论文**: *Adversarial Distribution Matching for Diffusion Distillation* (arXiv 2507.18569)

#### 核心思想

用对抗训练替代逆向 KL 散度最小化，从根本上避免 mode-seeking 行为：

```
传统 DMD: min D_KL(p_fake || p_real)  → 固有的 mode-seeking
ADM:      对抗训练 with diffusion discriminator → 更好的 mode coverage
```

#### 技术方案

**混合判别器**:
- Latent space discriminator
- Pixel space discriminator

**DMDX Pipeline**:
```
Stage 1: Distributional loss 预训练 (使用教师 ODE pairs)
Stage 2: ADM 微调 (对抗式分布匹配)
```

---

### 2.6 DMDR: 结合强化学习

**论文**: *Distribution Matching Distillation Meets Reinforcement Learning* (arXiv 2511.13649)

#### 核心创新

将强化学习与 DMD 结合，突破教师模型性能上限：

$$\mathcal{L}_{\text{DMDR}} = \mathcal{L}_{\text{RL}} + \lambda \mathcal{L}_{\text{DMD}}$$

关键发现: **DMD 损失本身是比传统方法更有效的 RL 正则化器**

#### 技术贡献

1. **同时训练**: RL 优化 + 蒸馏（而非顺序执行）
2. **动态分布引导**: Dynamic distribution guidance
3. **动态重噪声采样**: Dynamic renoise sampling

#### 结果

学生模型性能 **超越多步教师模型**

---

## 3. DMD 的局限性与风险分析

### 3.1 模式崩溃问题

#### 根本原因

**逆向 KL 散度的固有特性**:

$$D_{\text{KL}}(q \| p) = \mathbb{E}_q\left[\log \frac{q(x)}{p(x)}\right]$$

| 散度类型 | 行为模式 | 后果 |
|---------|---------|------|
| Forward KL: $D_{\text{KL}}(p \| q)$ | Mode-covering | 覆盖所有模式，但可能模糊 |
| **Reverse KL**: $D_{\text{KL}}(q \| p)$ | **Mode-seeking** | 集中于高密度区域，丢失多样性 |

#### 数学解释

当 $q(x) > 0$ 但 $p(x) \approx 0$ 时，$\log \frac{q(x)}{p(x)} \to +\infty$

因此优化器会 **避免** 在教师分布低密度区域生成样本，导致：
- 概率质量集中在少数高似然样本
- 不同输入噪声生成相似输出（部分记忆化）
- 严重时退化为几乎恒定输出

#### 量化表现

```
同一 prompt，不同随机种子:
- 基础扩散模型: 高度多样化的输出
- DMD 蒸馏模型: 相似的构图、颜色、物体位置
```

---

### 3.2 训练不稳定性

#### 问题来源

1. **Fake critic 估计不准确**
   - 生成分布持续变化
   - Critic 跟踪滞后

2. **两个 score estimator 的对抗性**
   - 类似 GAN 的不稳定动态
   - 梯度震荡

#### 表现症状

- 训练损失剧烈波动
- FID 指标不收敛
- 生成质量突然崩溃

---

### 3.3 数据集构建开销

#### 原始 DMD 的需求

```python
# 需要预先生成大量配对数据
for i in range(100000):  # 大规模数据集
    noise = sample_noise()
    image = teacher.sample(noise, steps=50)  # 耗时!
    save_pair(noise, image)
```

#### 成本分析

| 项目 | 开销 |
|------|------|
| 存储空间 | 数十 GB（取决于分辨率和数量） |
| 计算时间 | 数天（A100 GPU） |
| 更新困难 | 每次改变配置需重新生成 |

---

### 3.4 质量-多样性权衡

#### 核心矛盾

```
       优化质量 (低 FID)
            ↑
            |
            |  ← 存在 trade-off
            |
            ↓
       保持多样性 (高 LPIPS/低 DINO 相似度)
```

#### 现有方法的位置

| 方法 | 质量 | 多样性 | 计算开销 |
|------|------|--------|---------|
| DMD | 高 | **低** | 中 |
| DMD2 | **很高** | 中 | 高 (GAN) |
| DP-DMD | 高 | **高** | **低** |
| Phased DMD | 高 | 高 | 中 |
| ADM | 很高 | 高 | 高 |

---

## 4. 解决方案分类与对比

### 4.1 解决方案分类

```
                    DMD 局限性解决方案
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   损失函数改进       架构/训练改进      理论框架改进
        │                 │                 │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │         │       │         │       │         │
感知损失   对抗损失  多步训练  梯度阻断  f-散度   Score正则化
(LPIPS)   (GAN)    (SGTS)   (DP-DMD) (f-distill)
```

### 4.2 详细对比

| 方法 | 解决的问题 | 技术手段 | 优点 | 缺点 |
|------|-----------|---------|------|------|
| **DMD + LPIPS** | 结构一致性 | 感知损失正则化 | 简单易实现 | GPU 内存开销大 |
| **DMD2 (GAN)** | 训练不稳定 | 两时间尺度 + GAN | 质量很高 | 训练复杂，GAN 不稳定 |
| **DP-DMD** | 模式崩溃 | 角色分离 + 梯度阻断 | 无额外网络，高效 | 需要调参 (K, λ) |
| **Phased DMD** | 多步多样性退化 | SNR 分段训练 | 保持多步优势 | 需要多个专家网络 |
| **ADM** | 逆向 KL 固有问题 | 对抗式分布匹配 | 从根本解决 | 训练复杂 |
| **DMDR** | 受限于教师性能 | RL + DMD 联合 | 可超越教师 | 需要奖励模型 |
| **f-distill** | 散度选择局限 | 通用 f-散度框架 | 理论灵活 | 梯度方差大 |

### 4.3 选择建议

```
需求分析决策树:

是否需要最高质量？
├─ 是 → DMD2 或 ADM
└─ 否 → 继续

是否对多样性要求高？
├─ 是 → DP-DMD 或 Phased DMD
└─ 否 → 原始 DMD 可能足够

是否有计算资源限制？
├─ 是 → DP-DMD (无额外网络)
└─ 否 → DMD2 或 ADM

是否需要超越教师性能？
├─ 是 → DMDR
└─ 否 → 其他方法
```

---

## 5. 其他相关蒸馏方法

### 5.1 一致性模型 (Consistency Models)

**论文**: *Consistency Models* (ICML 2023)

| 特性 | 说明 |
|------|------|
| 核心思想 | 直接学习 PF-ODE 的解映射 |
| 训练方式 | 一致性蒸馏 / 一致性训练 |
| 优势 | 理论优雅，单步生成 |
| 劣势 | 大规模文生图效果验证不足 |

### 5.2 Latent Consistency Models (LCM)

**论文**: *Latent Consistency Models: Synthesizing High-Resolution Images with Few-step Inference*

| 特性 | 说明 |
|------|------|
| 核心改进 | 在潜空间应用一致性蒸馏 |
| 训练成本 | ~32 小时 A100 |
| 采样步数 | 2-4 步 |
| 与 DMD 对比 | LCM 更依赖一致性约束，DMD 更依赖分布匹配 |

### 5.3 Score Distillation Sampling (SDS)

**主要用于**: 3D 生成 (如 DreamFusion)

| 问题 | 解决方案 |
|------|---------|
| Mode-seeking | Variational Score Distillation (VSD) |
| Janus artifact | Entropic Score Distillation (ESD) |
| 模糊输出 | Denoised Score Distillation (DSD) |

---

## 6. 研究趋势与展望

### 6.1 当前研究热点

1. **多样性保持**: DP-DMD、Phased DMD 代表的方向
2. **视频扩散蒸馏**: 将图像蒸馏方法扩展到视频
3. **超越教师**: DMDR 等方法探索学生超越教师的可能
4. **大规模验证**: SDXL、Flux 等更大模型的蒸馏

### 6.2 开放问题

| 问题 | 难点 |
|------|------|
| 统一的质量-多样性理论 | 缺乏明确的数学框架 |
| 自动超参数选择 | K, λ 等参数敏感 |
| 可控生成的蒸馏 | ControlNet 等条件生成的高效蒸馏 |
| 理论分析 | 蒸馏误差的严格界 |

### 6.3 未来方向

1. **自适应蒸馏**: 根据输入动态调整采样步数
2. **多任务蒸馏**: 同时保持生成、编辑、补全能力
3. **硬件协同**: 针对特定硬件优化的蒸馏方法

---

## 7. 参考文献

### 核心 DMD 系列

1. **[DMD]** Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., & Park, T. (2024). *One-step Diffusion with Distribution Matching Distillation*. CVPR 2024.
   - arXiv: [https://arxiv.org/abs/2311.18828](https://arxiv.org/abs/2311.18828)
   - 项目主页: [https://tianweiy.github.io/dmd/](https://tianweiy.github.io/dmd/)

2. **[DMD2]** Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., & Freeman, W. T. (2024). *Improved Distribution Matching Distillation for Fast Image Synthesis*. NeurIPS 2024 Oral.
   - arXiv: [https://arxiv.org/abs/2405.14867](https://arxiv.org/abs/2405.14867)
   - GitHub: [https://github.com/tianweiy/DMD2](https://github.com/tianweiy/DMD2)
   - 项目主页: [https://tianweiy.github.io/dmd2/](https://tianweiy.github.io/dmd2/)

3. **[DP-DMD]** Wu, T., Li, R., Zhang, L., & Ma, K. (2025). *Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis*.
   - arXiv: [https://arxiv.org/abs/2602.03139](https://arxiv.org/abs/2602.03139)

4. **[Phased DMD]** *Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals*.
   - arXiv: [https://arxiv.org/abs/2510.27684](https://arxiv.org/abs/2510.27684)

5. **[ADM]** *Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis*.
   - arXiv: [https://arxiv.org/abs/2507.18569](https://arxiv.org/abs/2507.18569)

6. **[DMDR]** *Distribution Matching Distillation Meets Reinforcement Learning*.
   - arXiv: [https://arxiv.org/abs/2511.13649](https://arxiv.org/abs/2511.13649)

### 多样性与模式崩溃相关

7. **[Diversity Distillation]** Gandikota, R., & Bau, D. (2025). *Distilling Diversity and Control in Diffusion Models*.
   - arXiv: [https://arxiv.org/abs/2503.10637](https://arxiv.org/abs/2503.10637)
   - 项目主页: [https://distillation.baulab.info/](https://distillation.baulab.info/)

8. **[Taming Mode Collapse]** Wang, et al. (2024). *Taming Mode Collapse in Score Distillation for Text-to-3D Generation*. CVPR 2024.
   - arXiv: [https://arxiv.org/abs/2401.00909](https://arxiv.org/abs/2401.00909)

### 一致性模型相关

9. **[Consistency Models]** Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). *Consistency Models*. ICML 2023.

10. **[LCM]** Luo, S., Tan, Y., Huang, L., Li, J., & Zhao, H. (2023). *Latent Consistency Models: Synthesizing High-Resolution Images with Few-step Inference*.
    - 项目主页: [https://latent-consistency-models.github.io/](https://latent-consistency-models.github.io/)

11. **[MLCM]** *MLCM: Multistep Consistency Distillation of Latent Diffusion Model*.
    - arXiv: [https://arxiv.org/abs/2406.05768](https://arxiv.org/abs/2406.05768)

### Score Distillation 相关

12. **[SDS/DreamFusion]** Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). *DreamFusion: Text-to-3D using 2D Diffusion*. ICLR 2023.

13. **[Rethinking Score Distillation]** *Rethinking Score Distillation as a Bridge Between Image Distributions*. NeurIPS 2024.
    - 论文: [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/3b62bca132cf5c8973b09a2fc6dc8ca6-Paper-Conference.pdf)

### 综述与分析

14. **[EmergentMind DMD Topic]** Distribution Matching Distillation 专题整理
    - [https://www.emergentmind.com/topics/distribution-matching-distillation-dmd](https://www.emergentmind.com/topics/distribution-matching-distillation-dmd)

15. **[The Paradox of Diffusion Distillation]** Dieleman, S. (2024). Blog post.
    - [https://sander.ai/2024/02/28/paradox.html](https://sander.ai/2024/02/28/paradox.html)

---

## 附录 A: 关键公式汇总

### A.1 DMD 基础损失
$$\mathcal{L}_{\text{DMD}} = D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}})$$

### A.2 DMD 梯度
$$\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}\left[(s_{\text{fake}}(\bm{z}_t) - s_{\text{real}}(\bm{z}_t)) \nabla_\theta \bm{x}_\theta\right]$$

### A.3 DP-DMD 组合损失
$$\mathcal{L} = \mathcal{L}_{\text{DMD}} + \lambda \mathcal{L}_{\text{Div}}$$

### A.4 Flow Matching 多样性损失
$$\mathcal{L}_{\text{Div}} = \mathbb{E}_\epsilon\left[\|\bm{v}_\theta(\epsilon,1) - \bm{v}_k^{\text{target}}\|^2\right]$$

---

## 附录 B: 实验指标说明

| 指标 | 含义 | 方向 |
|------|------|------|
| FID | Fréchet Inception Distance, 生成质量 | ↓ 越低越好 |
| IS | Inception Score, 生成质量与多样性 | ↑ 越高越好 |
| LPIPS | 感知相似度距离 | 作为多样性指标时 ↑ 越高越好 |
| DINO 相似度 | 语义相似度 | 作为多样性指标时 ↓ 越低越好 |
| CLIP Score | 文本-图像对齐 | ↑ 越高越好 |

---

*文档最后更新: 2025年2月*
