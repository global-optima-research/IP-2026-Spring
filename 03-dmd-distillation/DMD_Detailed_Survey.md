# Distribution Matching Distillation (DMD) 详细研究综述

> **目标读者**: 希望深入理解 DMD 原理及相关工作的研究者
> **日期**: 2025年2月

---

## 目录

- [第一部分：基础知识](#第一部分基础知识)
  - [1. 扩散模型基础](#1-扩散模型基础)
  - [2. Score Function 详解](#2-score-function-详解)
  - [3. 为什么需要蒸馏](#3-为什么需要蒸馏)
- [第二部分：DMD 核心原理](#第二部分dmd-核心原理)
  - [4. DMD 的核心思想](#4-dmd-的核心思想)
  - [5. DMD 的数学原理](#5-dmd-的数学原理)
  - [6. DMD 的完整训练流程](#6-dmd-的完整训练流程)
- [第三部分：DMD 系列工作详解](#第三部分dmd-系列工作详解)
  - [7. DMD2：改进版本](#7-dmd2改进版本)
  - [8. DP-DMD：多样性保持](#8-dp-dmd多样性保持)
  - [9. Phased DMD：分阶段蒸馏](#9-phased-dmd分阶段蒸馏)
  - [10. ADM：对抗式分布匹配](#10-adm对抗式分布匹配)
  - [11. DMDR：结合强化学习](#11-dmdr结合强化学习)
- [第四部分：总结与对比](#第四部分总结与对比)
- [参考文献](#参考文献)

---

# 第一部分：基础知识

## 1. 扩散模型基础

### 1.1 什么是扩散模型？

扩散模型是一类生成模型，其核心思想可以用一个简单的比喻来理解：

> **想象你有一张清晰的照片，你不断往上面撒盐（加噪声），最终照片变成一片模糊的噪点。扩散模型学习的就是"如何一步步把盐去掉，恢复原来的照片"。**

扩散模型包含两个过程：

```
前向过程（加噪）          逆向过程（去噪）
清晰图像 ──────────────> 纯噪声
   x₀    → x₁ → x₂ → ... → xₜ
         <──────────────
         模型学习这个过程
```

### 1.2 前向过程（Forward Process）

前向过程是确定性的，逐步向数据添加高斯噪声：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中：
- $x_0$ 是原始清晰图像
- $x_t$ 是第 $t$ 步的噪声图像
- $\bar{\alpha}_t$ 是噪声调度参数（随 $t$ 增大而减小）
- $\epsilon$ 是标准高斯噪声

**直观理解**：$t$ 越大，$\sqrt{\bar{\alpha}_t}$ 越小，原始图像信号越弱；$\sqrt{1 - \bar{\alpha}_t}$ 越大，噪声越强。

### 1.3 逆向过程（Reverse Process）

逆向过程是模型需要学习的，从噪声逐步恢复图像：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

其中 $\epsilon_\theta(x_t, t)$ 是神经网络预测的噪声。

### 1.4 训练目标

扩散模型的训练非常简单——**预测噪声**：

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**直观理解**：给模型一张加了噪声的图片 $x_t$，让它猜测我们加了什么噪声 $\epsilon$。

---

## 2. Score Function 详解

### 2.1 什么是 Score Function？

**Score Function（得分函数）** 是概率分布的对数梯度：

$$s(x) = \nabla_x \log p(x)$$

**直观理解**：Score function 告诉你"从当前位置 $x$，往哪个方向走可以到达概率更高的地方"。

```
概率分布 p(x) 的等高线图：

        低概率区
           ↓
    ─────────────────
   │    ○  ○  ○      │
   │   ○  ●●●  ○     │  ● = 高概率区（如真实图像）
   │    ○  ○  ○      │  ○ = 中等概率区
   │                 │  箭头 = Score function 的方向
    ─────────────────
           ↑
        低概率区

Score function 在每个点都指向概率增加最快的方向
```

### 2.2 Score Function 与噪声预测的关系

有一个关键的数学关系：

$$s(x_t) = \nabla_{x_t} \log p(x_t) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

这意味着：**预测噪声 $\epsilon$ 等价于估计 score function！**

所以扩散模型的噪声预测网络 $\epsilon_\theta(x_t, t)$ 实际上在估计：

$$s_\theta(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

### 2.3 为什么 Score Function 重要？

Score function 的方向告诉我们如何将样本"推向"数据分布的高密度区域：

```
噪声图像          Score的方向           真实图像分布
    ●  ──────────────────────────>  ●●●●
 (当前位置)      (去噪方向)          (目标区域)
```

---

## 3. 为什么需要蒸馏？

### 3.1 扩散模型的推理瓶颈

扩散模型生成一张图像需要 **多次迭代去噪**：

| 模型 | 采样步数 | 每张图生成时间 |
|------|---------|--------------|
| DDPM | 1000 步 | 数分钟 |
| DDIM | 50-100 步 | 数秒 |
| Stable Diffusion | 20-50 步 | 3-10 秒 |

对于实时应用（如视频生成、交互式编辑），这是不可接受的。

### 3.2 蒸馏的目标

**知识蒸馏**：将"教师模型"（多步扩散模型）的知识压缩到"学生模型"（少步/单步生成器）。

```
教师模型（50步）                    学生模型（1-4步）
噪声 → [去噪] → [去噪] → ... → 图像    噪声 → [一步生成] → 图像
        慢但质量高                           快但需要蒸馏
```

### 3.3 蒸馏的挑战

核心问题：**如何让单步生成的图像分布与多步生成的分布一致？**

```
目标：
p_student(x) ≈ p_teacher(x)

即：学生生成的图像的统计特性要和教师一样
```

---

# 第二部分：DMD 核心原理

## 4. DMD 的核心思想

### 4.1 直观理解 DMD

DMD 的核心思想用一句话概括：

> **训练两个"裁判"（扩散模型），一个评价"真实图像有多真实"，另一个评价"生成图像有多像生成的"，两个裁判的评分之差就是学生模型的优化方向。**

```
                    ┌─────────────────────┐
                    │   单步生成器 G_θ     │
                    │  (学生，要训练的)    │
                    └──────────┬──────────┘
                               │
                          生成图像 x_θ
                               │
                    ┌──────────┴──────────┐
                    ▼                      ▼
          ┌─────────────────┐    ┌─────────────────┐
          │  Real Diffusion │    │  Fake Diffusion │
          │   (真实裁判)     │    │   (虚假裁判)     │
          │ 在真实图像上训练 │    │ 在生成图像上训练 │
          └────────┬────────┘    └────────┬────────┘
                   │                      │
              s_real(x)              s_fake(x)
                   │                      │
                   └──────────┬───────────┘
                              │
                        梯度 = s_fake - s_real
                              │
                              ▼
                    "让生成图像更真实、更不像生成的"
```

### 4.2 两个扩散模型的角色

| 模型 | 训练数据 | 作用 | 是否更新 |
|------|---------|------|---------|
| **Real Diffusion** | 真实图像（或教师生成的高质量图像） | 告诉生成图像"离真实还有多远" | 固定不变 |
| **Fake Diffusion** | 学生生成器当前生成的图像 | 告诉生成图像"你有多像生成的" | 持续更新 |

### 4.3 为什么需要两个模型？

**只用 Real Diffusion 的问题**：
- 只知道"真实是什么样"，但不知道"生成的问题在哪"
- 类似于只看标准答案，但不知道自己的答案错在哪

**加入 Fake Diffusion 的好处**：
- 知道生成图像的"缺陷特征"在哪
- 梯度方向：远离"生成的特征"，靠近"真实的特征"

```
梯度 = s_fake(x) - s_real(x)
     = "像生成的程度" - "像真实的程度"

优化方向：减小这个差值
→ 让生成图像更像真实的，更不像生成的
```

---

## 5. DMD 的数学原理

### 5.1 分布匹配的目标

DMD 的目标是最小化两个分布之间的 **KL 散度**：

$$\mathcal{L}_{\text{DMD}} = D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}})$$

其中：
- $p_{\text{fake}}$：学生生成器产生的图像分布
- $p_{\text{real}}$：目标分布（教师模型生成的图像分布）

### 5.2 KL 散度的梯度

KL 散度对生成器参数 $\theta$ 的梯度为：

$$\nabla_\theta D_{\text{KL}} = \mathbb{E}_{x \sim p_{\text{fake}}}\left[\nabla_\theta \log \frac{p_{\text{fake}}(x)}{p_{\text{real}}(x)}\right]$$

利用 score function 的定义 $s(x) = \nabla_x \log p(x)$，可以推导出：

$$\nabla_\theta \mathcal{L}_{\text{DMD}} = \mathbb{E}_{x_\theta}\left[(s_{\text{fake}}(x_t) - s_{\text{real}}(x_t)) \cdot \nabla_\theta x_\theta\right]$$

### 5.3 梯度的直观理解

```
s_fake(x_t) - s_real(x_t) 的含义：

s_fake(x_t)  = "从 x_t 出发，往哪走更像生成图像"
s_real(x_t)  = "从 x_t 出发，往哪走更像真实图像"

差值 = "两个方向的差异"

优化时取负梯度：
-（s_fake - s_real）= s_real - s_fake
= "往真实方向走" - "往生成方向走"
= "离开生成，走向真实"
```

### 5.4 为什么用逆向 KL 散度？

DMD 使用的是 **逆向 KL 散度** $D_{\text{KL}}(q \| p)$，而非正向 KL $D_{\text{KL}}(p \| q)$：

| 散度类型 | 公式 | 行为特点 |
|---------|------|---------|
| 正向 KL | $D_{\text{KL}}(p_{\text{real}} \| p_{\text{fake}})$ | **Mode-covering**：尝试覆盖所有模式，可能导致模糊 |
| 逆向 KL | $D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}})$ | **Mode-seeking**：集中于高概率区域，图像清晰但可能丢失多样性 |

**DMD 选择逆向 KL 的原因**：
1. 梯度容易从生成样本估计（不需要从真实分布采样）
2. 生成的图像质量更高（不会模糊）

**但代价是**：可能导致模式崩溃（后面详述）

---

## 6. DMD 的完整训练流程

### 6.1 训练算法伪代码

```python
# 初始化
G_θ = Generator()           # 单步生成器（要训练的学生）
D_real = Diffusion()        # 真实分布的扩散模型（预训练，固定）
D_fake = Diffusion()        # 生成分布的扩散模型（持续训练）

# 预计算回归目标（可选）
regression_pairs = []
for i in range(N):
    z = sample_noise()
    x = Teacher.sample(z, steps=50)  # 教师50步生成
    regression_pairs.append((z, x))

# 训练循环
for iteration in range(total_iterations):
    # ===== 第一步：更新 Fake Diffusion =====
    x_fake = G_θ(sample_noise())           # 生成假图像
    loss_fake_diff = train_diffusion(D_fake, x_fake)  # 在假图像上训练

    # ===== 第二步：计算分布匹配梯度 =====
    z = sample_noise()
    x_θ = G_θ(z)                           # 生成图像

    # 对生成图像加噪
    t = sample_timestep()
    ε = sample_noise()
    x_t = add_noise(x_θ, t, ε)

    # 用两个扩散模型预测 score
    s_real = D_real.score(x_t, t)          # 真实分布的 score
    s_fake = D_fake.score(x_t, t)          # 生成分布的 score

    # 分布匹配梯度
    grad_dm = (s_fake - s_real)

    # ===== 第三步：计算回归损失（可选）=====
    z_reg, x_target = sample_pair(regression_pairs)
    x_pred = G_θ(z_reg)
    loss_reg = LPIPS(x_pred, x_target)     # 感知损失

    # ===== 第四步：更新生成器 =====
    loss_total = loss_dm + λ * loss_reg
    G_θ.update(loss_total)
```

### 6.2 训练流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                        DMD 训练流程                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐     ┌─────────────┐     ┌─────────────────────┐    │
│  │ 随机噪声 │────>│ 单步生成器 G │────>│ 生成图像 x_θ        │    │
│  │    z    │     │    (学生)    │     │                     │    │
│  └─────────┘     └─────────────┘     └──────────┬──────────┘    │
│                                                  │               │
│                                    ┌─────────────┼─────────────┐ │
│                                    │             │             │ │
│                                    ▼             ▼             │ │
│                            ┌────────────┐ ┌────────────┐       │ │
│                            │  加噪声    │ │ 更新 D_fake│       │ │
│                            │  x_θ → x_t │ │ (持续训练) │       │ │
│                            └─────┬──────┘ └────────────┘       │ │
│                                  │                             │ │
│                    ┌─────────────┴─────────────┐               │ │
│                    ▼                           ▼               │ │
│            ┌─────────────┐             ┌─────────────┐         │ │
│            │  D_real     │             │  D_fake     │         │ │
│            │ (预训练固定) │             │ (持续更新)  │         │ │
│            │ 预测 s_real │             │ 预测 s_fake │         │ │
│            └──────┬──────┘             └──────┬──────┘         │ │
│                   │                           │                │ │
│                   └─────────────┬─────────────┘                │ │
│                                 │                              │ │
│                                 ▼                              │ │
│                   ┌─────────────────────────┐                  │ │
│                   │ 梯度 = s_fake - s_real  │                  │ │
│                   │ 更新生成器 G_θ          │                  │ │
│                   └─────────────────────────┘                  │ │
│                                                                │ │
│  可选：回归损失 L_reg = LPIPS(G(z), Teacher_output)            │ │
│                                                                │ │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 回归损失的作用

回归损失让学生生成器的输出在"大尺度结构"上与教师一致：

```
回归损失：L_reg = LPIPS(G_θ(z), T(z))

其中：
- G_θ(z)：学生单步生成的图像
- T(z)：教师50步生成的图像
- LPIPS：感知相似度损失（比 L2 更关注语义结构）

作用：
- 稳定训练（防止早期发散）
- 保证大尺度结构一致（构图、主体位置）
- 缺点：需要预先生成大量配对数据
```

### 6.4 DMD 的局限性

| 局限性 | 原因 | 后果 |
|--------|------|------|
| **模式崩溃** | 逆向 KL 的 mode-seeking 特性 | 生成多样性低，不同噪声输入产生相似输出 |
| **训练不稳定** | Fake Diffusion 估计不准确 | 梯度震荡，训练发散 |
| **数据集开销** | 需要预计算噪声-图像配对 | 存储和计算成本高 |
| **依赖回归损失** | 需要教师轨迹监督 | 学生质量受限于教师 |

---

# 第三部分：DMD 系列工作详解

## 7. DMD2：改进版本

**论文**: *Improved Distribution Matching Distillation for Fast Image Synthesis*
**会议**: NeurIPS 2024 Oral
**作者**: Tianwei Yin 等（MIT, Adobe）

### 7.1 解决的问题

DMD2 针对原始 DMD 的三个核心问题提出改进：

| 原始 DMD 问题 | 具体表现 | DMD2 解决方案 |
|--------------|---------|--------------|
| 回归损失需要预计算数据 | 需要教师生成数十万图像对 | 完全移除回归损失 |
| Fake Diffusion 估计不准 | 生成分布变化快，critic 跟不上 | 两时间尺度更新规则 |
| 学生受限于教师采样路径 | 只能模仿，不能超越 | 引入 GAN 损失 |

### 7.2 核心改进一：两时间尺度更新规则

**问题分析**：Fake Diffusion 需要估计当前生成器的输出分布，但生成器在不断更新，导致 Fake Diffusion 的估计总是"过时"的。

**解决方案**：让 Fake Diffusion 更新得更频繁

```python
# 两时间尺度更新
for iteration in range(total_iterations):
    # Fake Diffusion 更新 K 次（K > 1）
    for _ in range(K):
        x_fake = G_θ(sample_noise())
        update_fake_diffusion(D_fake, x_fake)

    # 生成器只更新 1 次
    update_generator(G_θ, D_real, D_fake)
```

**直观理解**：
```
假设 K = 5：

时间 →
Fake Diffusion:  更新 更新 更新 更新 更新 | 更新 更新 更新 更新 更新 | ...
Generator:                           更新 |                        更新 | ...

这样 Fake Diffusion 有更多时间"追上"生成器的分布变化
```

### 7.3 核心改进二：GAN 损失

**动机**：即使 Fake Diffusion 估计准确，它仍然只反映"生成图像的分布"，而非"真实图像的分布"。

**解决方案**：引入 GAN 判别器，直接区分生成图像和真实图像

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DMD}} + \lambda_{\text{GAN}} \mathcal{L}_{\text{GAN}}$$

```python
# GAN 损失
D_gan = Discriminator()

# 判别器损失
loss_D = -log(D_gan(x_real)) - log(1 - D_gan(G_θ(z)))

# 生成器损失（加入 GAN 项）
loss_G = loss_DMD + λ * (-log(D_gan(G_θ(z))))
```

**好处**：
1. 可以在真实图像上训练，不仅限于教师生成的图像
2. 弥补教师 score 估计的不完美
3. 提升整体生成质量

### 7.4 核心改进三：多步采样支持

**问题**：原始 DMD 只能训练单步生成器。但有时 2-4 步可以获得更好的质量。

**解决方案**：训练时模拟推理时的多步过程

```python
# 多步训练（以2步为例）
def train_step():
    z = sample_noise()

    # 第一步
    x_1 = G_θ.step1(z)

    # 第二步（关键：用第一步的实际输出，而非 detach）
    x_2 = G_θ.step2(x_1)  # 梯度可以传回 step1

    # 计算损失时使用最终输出 x_2
    loss = compute_DMD_loss(x_2, D_real, D_fake)
```

### 7.5 DMD2 完整算法

```
Algorithm: DMD2 Training

输入：预训练扩散模型 D_real，生成器 G_θ
超参数：K（fake model 更新频率），λ_GAN

1. 初始化 D_fake = copy(D_real)
2. 初始化 GAN 判别器 D_gan

3. for iteration = 1 to N do:

   4. // 更新 Fake Diffusion（K 次）
   5. for k = 1 to K do:
   6.     z ~ N(0, I)
   7.     x_fake = G_θ(z).detach()  // 不传梯度到 G
   8.     t ~ Uniform(1, T)
   9.     ε ~ N(0, I)
   10.    x_t = α_t * x_fake + σ_t * ε
   11.    loss_fake = ||D_fake(x_t, t) - ε||²
   12.    更新 D_fake

   13. // 更新 GAN 判别器
   14. x_real ~ 真实数据
   15. x_fake = G_θ(z).detach()
   16. loss_D = -log(D_gan(x_real)) - log(1 - D_gan(x_fake))
   17. 更新 D_gan

   18. // 更新生成器
   19. z ~ N(0, I)
   20. x_θ = G_θ(z)
   21. t ~ Uniform(1, T)
   22. x_t = 加噪(x_θ, t)
   23. s_real = D_real.score(x_t, t)
   24. s_fake = D_fake.score(x_t, t)
   25. loss_DMD = (s_fake - s_real) · ∇x_θ
   26. loss_GAN = -log(D_gan(x_θ))
   27. loss_total = loss_DMD + λ_GAN * loss_GAN
   28. 更新 G_θ

输出：训练好的单步生成器 G_θ
```

### 7.6 DMD2 性能

| 数据集 | DMD FID | DMD2 FID | 提升 |
|--------|---------|----------|------|
| ImageNet-64×64 | 2.62 | **1.28** | 51% |
| COCO-2014 (zero-shot) | 11.49 | **8.35** | 27% |

**关键突破**：DMD2 的学生模型在某些指标上 **超越了教师模型**！

---

## 8. DP-DMD：多样性保持

**论文**: *Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis*
**arXiv**: 2602.03139
**作者**: Tianhe Wu 等

### 8.1 核心问题：模式崩溃

DMD 系列方法面临的最大问题是 **模式崩溃（Mode Collapse）**：

```
理想情况：                          实际情况（模式崩溃）：
不同噪声 → 不同风格的猫              不同噪声 → 几乎一样的猫

z₁ → 🐱 (黑猫)                      z₁ → 🐱 (橘猫)
z₂ → 🐱 (白猫)                      z₂ → 🐱 (橘猫)  ← 几乎一样！
z₃ → 🐱 (橘猫)                      z₃ → 🐱 (橘猫)  ← 几乎一样！
```

### 8.2 模式崩溃的根本原因

**数学原因**：逆向 KL 散度的 mode-seeking 特性

$$D_{\text{KL}}(q \| p) = \mathbb{E}_q\left[\log \frac{q(x)}{p(x)}\right]$$

当 $q(x) > 0$ 但 $p(x) \approx 0$ 时，$\log \frac{q(x)}{p(x)} \to +\infty$

**后果**：优化器会让 $q$ 避免在 $p$ 的低密度区域生成样本

```
p_real 的分布（多峰）：

    ▲
    │    ╱╲      ╱╲      ╱╲
    │   ╱  ╲    ╱  ╲    ╱  ╲
    │  ╱    ╲  ╱    ╲  ╱    ╲
    │ ╱      ╲╱      ╲╱      ╲
    └────────────────────────────>
      模式1   模式2   模式3

逆向 KL 优化后的 q_fake：

    ▲
    │           ╱╲
    │          ╱  ╲
    │         ╱    ╲
    │        ╱      ╲
    └────────────────────────────>
              只剩一个模式！
```

### 8.3 DP-DMD 的核心洞察

**发现**：扩散生成过程中，不同阶段有不同作用

| 阶段 | 时间步 | 作用 | 对多样性的影响 |
|------|--------|------|---------------|
| 早期（低 SNR） | t ≈ T | 决定全局结构、构图 | **决定性** |
| 后期（高 SNR） | t ≈ 0 | 细化纹理、细节 | 较小 |

```
生成过程可视化：

t=T (纯噪声)     t=T/2 (粗略形状)    t=0 (最终图像)
    ░░░░░░           ▓▓░░░░              🐱
    ░░░░░░    →      ░▓▓▓░░      →
    ░░░░░░           ░░▓▓░░

    ↑                   ↑                 ↑
 多样性源头          结构确定           细节优化
（在这里做对）    （在这里已定型）
```

### 8.4 DP-DMD 的解决方案：角色分离

**核心思想**：给不同步骤分配不同的训练目标

```
第一步（决定多样性）          后续步骤（优化质量）
━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━
   Flow Matching 损失            DMD 损失
   锚定到教师中间状态            标准分布匹配
   保持结构多样性                优化图像质量
```

### 8.5 技术实现：梯度阻断

**关键机制**：在第一步输出后切断梯度

```python
def dp_dmd_forward(noise, teacher, student):
    # 第一步：使用 Flow Matching 损失
    z_1 = student.step1(noise)

    # 🔑 关键：阻断梯度！
    z_1_detached = z_1.detach()  # stop gradient

    # 后续步骤：使用 DMD 损失
    z_final = student.remaining_steps(z_1_detached)

    return z_1, z_final
```

**为什么要阻断梯度？**

```
不阻断梯度的情况：

DMD 损失 → 梯度传播 → step2 → step1
                              ↑
                    DMD 的 mode-seeking 也影响 step1
                    → 第一步也被"污染"，多样性下降

阻断梯度的情况：

DMD 损失 → 梯度传播 → step2 → ✂️ (cut) → step1 ← Flow Matching 损失
                                              ↑
                              第一步只受 Flow Matching 影响
                              → 多样性得以保持
```

### 8.6 Flow Matching 多样性损失

**目标**：让学生第一步的输出"锚定"到教师的中间状态

$$\mathcal{L}_{\text{Div}} = \mathbb{E}_\epsilon\left[\|v_\theta(\epsilon, 1) - v_k^{\text{target}}\|^2\right]$$

其中目标速度为：

$$v_k^{\text{target}} = \frac{\epsilon - z_{t_k}}{1 - t_k}$$

**直观理解**：
```
教师模型在 k 步后的状态 z_k
    ↓
计算从纯噪声 ε 到 z_k 需要的"速度"
    ↓
让学生第一步的输出速度与之匹配
    ↓
学生第一步就能到达与教师类似的"结构状态"
    ↓
保持了教师的结构多样性
```

### 8.7 DP-DMD 完整算法

```
Algorithm: DP-DMD Training

输入：教师扩散模型 T，学生生成器 G_θ
超参数：K（锚定步数），λ（权重系数）

1. for iteration = 1 to N do:

   2. // 采样噪声
   3. ε ~ N(0, I)

   4. // 教师前向 K 步，获得中间状态
   5. z_k = T.forward_k_steps(ε, K)

   6. // 计算目标速度
   7. v_target = (ε - z_k) / (1 - t_k)

   8. // 学生第一步
   9. v_1 = G_θ.velocity(ε, t=1)
   10. z_1 = G_θ.step1(ε)

   11. // 多样性损失
   12. L_div = ||v_1 - v_target||²

   13. // 🔑 梯度阻断
   14. z_1_detached = z_1.detach()

   15. // 学生后续步骤
   16. z_final = G_θ.remaining_steps(z_1_detached)

   17. // DMD 损失（标准流程）
   18. L_dmd = compute_dmd_loss(z_final)

   19. // 总损失
   20. L_total = L_dmd + λ * L_div

   21. // 更新（注意：L_div 只更新 step1，L_dmd 只更新后续步骤）
   22. 更新 G_θ

输出：多样性保持的少步生成器 G_θ
```

### 8.8 DP-DMD 的优势

| 特性 | DP-DMD | DMD + LPIPS | DMD + GAN |
|------|--------|-------------|-----------|
| 额外网络 | ❌ 无 | ✅ VGG | ✅ Discriminator |
| GPU 内存 | 低 | 高 | 中 |
| 训练稳定性 | 高 | 中 | 低 |
| 多样性保持 | **优秀** | 一般 | 一般 |
| 实现复杂度 | **简单** | 简单 | 复杂 |

### 8.9 消融实验关键发现

**1. 锚定步数 K 的影响**：
```
K=1:  多样性较低，结构过于简单
K=5:  平衡点，推荐默认值
K=10: 多样性最高，但质量略降
K=30: 多样性饱和，计算开销增加
```

**2. 权重系数 λ 的影响**：
```
λ=0.01: 多样性提升有限
λ=0.05: 推荐值，平衡多样性和质量
λ=0.10: 多样性最高，质量轻微下降
```

**3. 梯度阻断的必要性**：
```
无梯度阻断：多样性随训练快速下降
有梯度阻断：多样性始终保持在较高水平
```

---

## 9. Phased DMD：分阶段蒸馏

**论文**: *Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals*
**arXiv**: 2510.27684

### 9.1 解决的问题

现有多步蒸馏方法（如 SGTS）的问题：

```
SGTS（随机梯度截断策略）：
训练时随机选择在哪一步终止

问题：有时只训练 1 步就终止
    → 多步生成器退化成单步生成器的质量
    → 多样性大幅下降
```

### 9.2 核心思想：SNR 分段

将信噪比（SNR）范围分成多个区间，每个区间训练一个专家：

```
完整 SNR 范围: [0, 1]
         │
         ▼ 分割
┌─────────────────────────────────────────┐
│ Phase 1: [0, 0.3]   → Expert 1 (粗结构) │
│ Phase 2: [0.3, 0.7] → Expert 2 (中间层) │
│ Phase 3: [0.7, 1.0] → Expert 3 (细节)   │
└─────────────────────────────────────────┘
```

### 9.3 子区间内的 Score Matching

对于中间时间步 $s$ 的训练目标：

$$\mathcal{J}_{\text{flow}}(\theta) = \mathbb{E}\left[\text{clamp}\left(\frac{1}{\sigma^2_{t|s}}\right) \left\|\sigma_{t|s}\psi_\theta(x_t) - \text{target}\right\|^2\right]$$

其中 $t$ 从子区间 $(s, 1)$ 采样，clamp 函数防止数值不稳定。

### 9.4 渐进式训练流程

```
训练流程：

Phase 1: 训练 Expert 1（粗结构）
    └── 冻结 Expert 1

Phase 2: 使用 Expert 1 的输出，训练 Expert 2（中间细节）
    └── 冻结 Expert 1, Expert 2

Phase 3: 使用 Expert 1,2 的输出，训练 Expert 3（精细细节）
    └── 冻结所有 Experts

推理时：
    噪声 → Expert 1 → Expert 2 → Expert 3 → 最终图像
```

### 9.5 为什么能保持多样性？

**关键洞察**：低 SNR 阶段（Phase 1）决定了图像的整体构图

```
Phase 1 训练完成后冻结
    ↓
后续 Phase 只优化细节，不改变构图
    ↓
构图的多样性被"锁定"保留
    ↓
即使后续训练很长，多样性也不会下降
```

### 9.6 Phased DMD 完整算法

```
Algorithm: Phased DMD Training

输入：教师模型 T，时间步划分 [t_0=0, t_1, ..., t_K=1]

1. for phase k = 1 to K do:

   2. // 使用已训练的专家生成中间状态
   3. for each training sample:
   4.     z ~ N(0, I)
   5.     x_{t_{k-1}} = Expert_1...Expert_{k-1}(z)  // 前面专家的输出

   6. // 训练当前阶段的 Fake Diffusion
   7. for iteration = 1 to M:
   8.     t ~ Uniform(t_{k-1}, t_k)  // 只在当前子区间采样
   9.     扩散 x_{t_{k-1}} 到 x_t
   10.    训练 F_{θ_k} 预测 score

   11. // 训练当前专家
   12. for iteration = 1 to N:
   13.    计算分布匹配梯度（在子区间内）
   14.    更新 Expert_k

   15. // 冻结当前专家
   16. Freeze(Expert_k)

输出：K 个专家组成的少步生成器
```

### 9.7 实验结果

**视频生成多样性对比**：

| 方法 | DINOv3 相似度 ↓ | LPIPS 距离 ↑ |
|------|----------------|--------------|
| DMD + SGTS | 0.826 | 0.521 |
| **Phased DMD** | **0.782** | **0.544** |

**运动质量对比**：

| 方法 | 光流分数 ↑ | 动态程度 ↑ |
|------|-----------|-----------|
| SGTS | 3.23 | 65.45% |
| **Phased DMD** | **7.57** | **74.55%** |

**规模验证**：成功蒸馏 280 亿参数的 Wan2.2 模型（最大规模验证）

---

## 10. ADM：对抗式分布匹配

**论文**: *Adversarial Distribution Matching for Diffusion Distillation*
**arXiv**: 2507.18569

### 10.1 核心问题：逆向 KL 的根本缺陷

ADM 认为，之前的方法都是在"修补"逆向 KL 的问题，而没有从根本上解决：

```
DMD:     逆向 KL + 回归损失          → 仍有模式崩溃
DMD2:    逆向 KL + GAN 损失          → 部分缓解
DP-DMD:  逆向 KL + 梯度阻断          → 部分缓解

ADM 的问题：能否完全不用逆向 KL？
```

### 10.2 ADM 的解决方案：用对抗损失替代 KL

**核心思想**：用 GAN 的对抗损失完全替代 KL 散度

```
DMD 损失:  min D_KL(p_fake || p_real)

ADM 损失:  min_G max_D [D 区分真假的能力]
           ↓
        最小化 Total Variation Distance
```

**为什么 TVD 更好？**

| 散度类型 | 对称性 | 有界性 | Mode-seeking |
|---------|--------|--------|--------------|
| 逆向 KL | ❌ 不对称 | ❌ 可能无穷 | ✅ 严重 |
| **TVD** | ✅ 对称 | ✅ [0,1] | ❌ 无 |

### 10.3 扩散判别器架构

ADM 不使用普通的 CNN 判别器，而是使用 **扩散模型作为判别器**：

```
传统 GAN 判别器：
    图像 x → CNN → 真/假概率

ADM 扩散判别器：
    带噪图像 x_t, 时间步 t → 扩散 UNet → Score 预测
                                 ↓
                          在多个尺度评价真假
```

**优势**：
1. 利用扩散模型的多尺度特征
2. 在不同噪声水平评价（更全面）
3. 可以复用预训练的扩散模型权重

### 10.4 双空间判别器

ADM 使用两个判别器：

```
┌─────────────────────────────────────────────────────┐
│              ADM 双空间判别器                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │ Latent 空间判别器 │    │ Pixel 空间判别器 │        │
│  │   D_τ1 (85%)    │    │   D_τ2 (15%)    │        │
│  │                 │    │                 │        │
│  │ 基于教师扩散模型  │    │ 基于 SAM 编码器  │        │
│  │ 评价 latent 特征 │    │ 评价像素细节     │        │
│  └─────────────────┘    └─────────────────┘        │
│           │                      │                 │
│           └──────────┬───────────┘                 │
│                      │                             │
│              加权组合损失                           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 10.5 DMDX 两阶段流水线

```
Stage 1: 对抗蒸馏预训练 (ADP)
├── 在离线收集的 ODE pairs 上训练
├── 使用立方时间步调度（偏向高噪声区域）
├── 双空间判别器
└── 输出：初步对齐的学生模型

         ↓

Stage 2: ADM 微调
├── 单一 latent 判别器
├── 在完整概率流上优化
├── Hinge 损失
└── 输出：最终的高质量学生模型
```

### 10.6 ADM 损失函数

**生成器损失**：
$$\mathcal{L}_G(\theta) = -\mathbb{E}\left[D_\tau(x_{t-\Delta t}^{\text{fake}}, t-\Delta t)\right]$$

**判别器损失**（Hinge 形式）：
$$\mathcal{L}_D = \mathbb{E}\left[\max(0, 1 + D(x^{\text{fake}}))\right] + \mathbb{E}\left[\max(0, 1 - D(x^{\text{real}}))\right]$$

### 10.7 ADM vs DMD 系列对比

| 方面 | DMD/DMD2 | ADM |
|------|----------|-----|
| 主要损失 | 逆向 KL 散度 | Total Variation (Hinge GAN) |
| 模式崩溃处理 | 额外正则化器缓解 | 从根本上避免（TVD 无 mode-seeking） |
| 散度度量 | 预定义，固定 | 可学习，数据驱动 |
| 分布重叠假设 | 需要大量重叠 | 可处理小重叠（TVD 有界） |
| 预训练必要性 | 可选 | 关键（单步时必须） |

### 10.8 理论优势

**关键定理**：当判别器足够强且训练充分时，Hinge 损失在收敛时最小化 TVD。

**实验验证**：即使不直接优化 DMD 损失，ADM 训练过程中 DMD 损失仍然稳定下降（图3），说明判别器隐式地包含了 KL 最小化，同时避免了其病态行为。

---

## 11. DMDR：结合强化学习

**论文**: *Distribution Matching Distillation Meets Reinforcement Learning*
**arXiv**: 2511.13649

### 11.1 解决的问题

所有之前的蒸馏方法都有一个共同限制：

> **学生模型的性能上限是教师模型**

因为学生是在"模仿"教师，怎么可能超越教师呢？

### 11.2 DMDR 的洞察

**关键发现**：DMD 损失本身可以作为 RL 的正则化器

```
传统 RL 微调扩散模型：
    Loss = RL_reward_loss + λ * KL_regularization
    问题：KL 正则化可能不够有效

DMDR：
    Loss = RL_reward_loss + λ * DMD_loss
    优势：DMD 损失提供更强的分布约束
```

### 11.3 DMDR 的框架

```
┌─────────────────────────────────────────────────────┐
│                    DMDR 框架                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐                                    │
│  │  奖励模型    │ ← 评价生成图像质量/对齐度           │
│  │  R(x, text) │                                    │
│  └──────┬──────┘                                    │
│         │                                           │
│         ▼                                           │
│  ┌─────────────┐                                    │
│  │  RL 损失    │ L_RL = -E[R(G_θ(z), text)]        │
│  └──────┬──────┘                                    │
│         │                                           │
│         │    ┌─────────────┐                        │
│         │    │  DMD 损失    │ 作为正则化器           │
│         │    └──────┬──────┘                        │
│         │           │                               │
│         ▼           ▼                               │
│  ┌─────────────────────────┐                        │
│  │ L_total = L_RL + λ*L_DMD │                        │
│  └─────────────────────────┘                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 11.4 技术创新

**1. 同时训练（而非顺序训练）**：
```
传统方法：先蒸馏 → 再 RL 微调（两阶段）
DMDR：    蒸馏 + RL 同时进行（单阶段）
```

**2. 动态分布引导**：
```python
# 根据训练进度调整 DMD 和 RL 损失的权重
λ_t = schedule(iteration)  # 随训练递减
loss = L_RL + λ_t * L_DMD
```

**3. 动态重噪声采样**：
```python
# 训练时动态调整噪声注入策略
# 早期：更多噪声，鼓励探索
# 后期：更少噪声，精细优化
```

### 11.5 DMDR 的结果

**关键突破**：学生模型性能 **超越多步教师模型**

| 指标 | 教师 (50步) | 学生 (1步) |
|------|------------|-----------|
| 人类偏好率 | 47% | **53%** |
| CLIP Score | 0.31 | **0.33** |
| 美学评分 | 5.8 | **6.1** |

---

# 第四部分：总结与对比

## 方法对比总表

| 方法 | 解决的核心问题 | 主要技术 | 优点 | 缺点 |
|------|--------------|---------|------|------|
| **DMD** | 快速生成 | 分布匹配 + 回归损失 | 首创框架 | 模式崩溃、需要预计算 |
| **DMD2** | 训练不稳定 | 两时间尺度 + GAN | 质量高、超越教师 | 训练复杂 |
| **DP-DMD** | 模式崩溃 | 角色分离 + 梯度阻断 | 简单高效、无额外网络 | 需要调参 |
| **Phased DMD** | 多步多样性退化 | SNR 分段训练 | 保持多步优势 | 需要多个专家 |
| **ADM** | 逆向 KL 根本缺陷 | 对抗式分布匹配 | 从根本解决 | 训练复杂 |
| **DMDR** | 受限于教师 | RL + DMD 联合 | 可超越教师 | 需要奖励模型 |

## 选择建议

```
你的需求是什么？
│
├── 追求最高质量 → DMD2 或 ADM
│
├── 追求高多样性 → DP-DMD 或 Phased DMD
│
├── 计算资源有限 → DP-DMD（无额外网络）
│
├── 想超越教师性能 → DMDR
│
└── 刚入门，想简单实现 → 原始 DMD
```

---

# 参考文献

## DMD 核心系列

1. **[DMD]** Yin, T., et al. *One-step Diffusion with Distribution Matching Distillation*. CVPR 2024.
   - [arXiv](https://arxiv.org/abs/2311.18828) | [项目主页](https://tianweiy.github.io/dmd/)

2. **[DMD2]** Yin, T., et al. *Improved Distribution Matching Distillation for Fast Image Synthesis*. NeurIPS 2024 Oral.
   - [arXiv](https://arxiv.org/abs/2405.14867) | [GitHub](https://github.com/tianweiy/DMD2)

3. **[DP-DMD]** Wu, T., et al. *Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis*.
   - [arXiv](https://arxiv.org/abs/2602.03139)

4. **[Phased DMD]** *Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals*.
   - [arXiv](https://arxiv.org/abs/2510.27684)

5. **[ADM]** *Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis*.
   - [arXiv](https://arxiv.org/abs/2507.18569)

6. **[DMDR]** *Distribution Matching Distillation Meets Reinforcement Learning*.
   - [arXiv](https://arxiv.org/abs/2511.13649)

## 扩散模型基础

7. **[DDPM]** Ho, J., et al. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.

8. **[Score Matching]** Song, Y., et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021.

## 相关蒸馏方法

9. **[Consistency Models]** Song, Y., et al. *Consistency Models*. ICML 2023.

10. **[LCM]** Luo, S., et al. *Latent Consistency Models*.
    - [项目主页](https://latent-consistency-models.github.io/)

## 教程与综述

11. **[Annotated Diffusion]** HuggingFace Blog.
    - [链接](https://huggingface.co/blog/annotated-diffusion)

12. **[EmergentMind DMD Topic]** 综合整理.
    - [链接](https://www.emergentmind.com/topics/distribution-matching-distillation-dmd)

---

*文档最后更新: 2025年2月*
