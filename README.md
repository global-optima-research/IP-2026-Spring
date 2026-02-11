# IP-2026-Spring: Video Editing 技术调研

> **团队内部调研文档** | Global Optima Research
> 本文档旨在帮助团队成员全面了解 Video Editing 领域的定义、核心任务、关键技术和前沿进展。

---

## 目录

- [1. 什么是 Video Editing](#1-什么是-video-editing)
  - [1.1 定义与范围](#11-定义与范围)
  - [1.2 Video Editing 的核心任务分类](#12-video-editing-的核心任务分类)
  - [1.3 与相关领域的关系](#13-与相关领域的关系)
- [2. 文本引导的视频编辑 (Text-Guided Video Editing)](#2-文本引导的视频编辑-text-guided-video-editing)
  - [2.1 任务定义](#21-任务定义)
  - [2.2 技术路线分类](#22-技术路线分类)
  - [2.3 核心技术详解](#23-核心技术详解)
  - [2.4 代表性方法](#24-代表性方法)
  - [2.5 最新进展 (2025-2026)](#25-最新进展-2025-2026)
- [3. 视频修复 (Video Inpainting)](#3-视频修复-video-inpainting)
  - [3.1 任务定义](#31-任务定义)
  - [3.2 经典方法](#32-经典方法)
  - [3.3 深度学习方法](#33-深度学习方法)
  - [3.4 扩散模型方法](#34-扩散模型方法)
- [4. 视频风格迁移 (Video Style Transfer)](#4-视频风格迁移-video-style-transfer)
  - [4.1 任务定义与分类](#41-任务定义与分类)
  - [4.2 代表性方法](#42-代表性方法)
  - [4.3 扩散模型时代的风格迁移](#43-扩散模型时代的风格迁移)
- [5. 视频生成基础模型 (Video Generation Foundation Models)](#5-视频生成基础模型-video-generation-foundation-models)
  - [5.1 技术路线演进](#51-技术路线演进)
  - [5.2 GAN 时代](#52-gan-时代)
  - [5.3 自回归/Token 时代](#53-自回归token-时代)
  - [5.4 扩散模型 + UNet 时代](#54-扩散模型--unet-时代)
  - [5.5 Diffusion Transformer (DiT) 时代](#55-diffusion-transformer-dit-时代)
  - [5.6 生成模型如何应用于视频编辑](#56-生成模型如何应用于视频编辑)
- [6. 视频到视频翻译 (Video-to-Video Translation)](#6-视频到视频翻译-video-to-video-translation)
- [7. 相关辅助任务](#7-相关辅助任务)
  - [7.1 视频超分辨率 (Video Super-Resolution)](#71-视频超分辨率-video-super-resolution)
  - [7.2 视频帧插值 (Video Frame Interpolation)](#72-视频帧插值-video-frame-interpolation)
  - [7.3 视频着色 (Video Colorization)](#73-视频着色-video-colorization)
  - [7.4 视频抠图与分割 (Video Matting & Segmentation)](#74-视频抠图与分割-video-matting--segmentation)
- [8. 商业产品与系统](#8-商业产品与系统)
- [9. 数据集与评估基准](#9-数据集与评估基准)
  - [9.1 常用数据集](#91-常用数据集)
  - [9.2 评估指标](#92-评估指标)
  - [9.3 专用评测基准](#93-专用评测基准)
- [10. 开源模型与框架](#10-开源模型与框架)
- [11. 关键挑战与未来方向](#11-关键挑战与未来方向)
- [12. 参考文献](#12-参考文献)

---

## 1. 什么是 Video Editing

### 1.1 定义与范围

**Video Editing（视频编辑）** 是指对已有视频内容进行修改、变换或增强的技术，核心目标是在改变视频视觉内容的同时，保持时间维度上的一致性（temporal consistency），即编辑后的视频在帧与帧之间不产生闪烁、跳变或不自然的视觉断裂。

从算法角度看，Video Editing 不同于传统剪辑软件中的"剪切-拼接"操作，它更关注**像素级别的视觉内容变换**，例如：
- 改变视频中物体的颜色、纹理或形状
- 替换或移除视频中的特定对象
- 改变整体视觉风格（如转为水彩画风格）
- 修改场景背景
- 填补视频中的缺损区域

### 1.2 Video Editing 的核心任务分类

| 任务类别 | 描述 | 输入/条件 | 典型应用场景 |
|---------|------|----------|------------|
| **文本引导编辑** | 根据文本描述修改视频内容 | 源视频 + 文本提示 | "将红色汽车变为蓝色" |
| **视频修复 (Inpainting)** | 填补视频中缺损或被遮挡的区域 | 源视频 + 掩码 | 去水印、去除行人 |
| **风格迁移** | 改变视频的整体或局部视觉风格 | 源视频 + 风格参考 | 将实拍转为动画风格 |
| **视频到视频翻译** | 将一种视觉域的视频转换为另一种 | 语义图/边缘图 + 文本 | 语义分割图转真实街景 |
| **超分辨率** | 提升视频空间分辨率 | 低分辨率视频 | 老视频修复、4K升级 |
| **帧插值** | 在已有帧之间生成中间帧 | 相邻帧对 | 慢动作、帧率提升 |
| **视频着色** | 为灰度视频添加色彩 | 灰度视频 (+ 参考) | 黑白影片修复 |
| **视频抠图/分割** | 精确分离前景与背景 | 视频 (+ 提示) | 绿幕替换、特效合成 |

### 1.3 与相关领域的关系

```
                    ┌─────────────────────┐
                    │   Video Generation  │  从零生成视频
                    │  (Text/Image→Video) │
                    └─────────┬───────────┘
                              │ 技术共享
                    ┌─────────▼───────────┐
                    │    Video Editing     │  修改已有视频
                    │ (Video→Edited Video) │
                    └─────────┬───────────┘
                              │ 依赖
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │ Video Under-  │  │ Image Editing │  │ Video Enhance │
  │  standing     │  │ (基础技术)     │  │  (增强任务)    │
  │ (分割/跟踪)   │  │               │  │ (超分/插帧)    │
  └───────────────┘  └───────────────┘  └───────────────┘
```

Video Editing 与 Video Generation 共享大量底层技术（如扩散模型、时序注意力机制），但关键区别在于：Video Generation 从噪声或文本生成全新视频，而 Video Editing 必须在保留源视频结构、运动和未编辑区域的前提下进行修改。

---

## 2. 文本引导的视频编辑 (Text-Guided Video Editing)

### 2.1 任务定义

**文本引导的视频编辑** 是当前 Video Editing 领域最活跃的研究方向。给定一段源视频和一个文本提示（目标描述或编辑指令），系统生成一段编辑后的视频，使其反映文本描述的变化，同时保持源视频的结构布局、运动动态和时间一致性。

**编辑类型包括：**

| 编辑类型 | 示例 |
|---------|------|
| 风格编辑 | "将视频转为水彩画风格" |
| 属性编辑 | "将汽车从红色改为蓝色" |
| 物体替换 | "将猫替换为狗" |
| 背景编辑 | "将背景改为雪景" |
| 主体驱动编辑 | 基于参考图像替换人物身份 |
| 全局 vs 局部编辑 | 全帧风格变化 vs 区域遮罩编辑 |

### 2.2 技术路线分类

#### 按训练需求分类

| 类别 | 代表方法 | 特点 |
|-----|---------|------|
| **Zero-shot (无需训练)** | FateZero, TokenFlow, RAVE, FRESCO, Text2Video-Zero, VidToMe | 直接利用预训练 T2I/T2V 模型，无需微调。计算成本低，但可能存在时空失真 |
| **One-shot (单样本微调)** | Tune-A-Video, ControlVideo | 在单个视频-文本对上微调。学习视频特定的时空模式 |
| **Training-based (训练式)** | InstructVid2Vid, InsV2V, VIVA, InsViE-1M | 在大规模视频编辑数据集上训练。泛化能力更强 |

#### 按技术基础分类

```
文本引导视频编辑
├── 基于 T2I 模型扩展
│   ├── 注意力操控类: FateZero, TokenFlow, VidToMe
│   ├── ControlNet 引导类: ControlVideo, LOVECon, CCEdit
│   └── 光流引导类: FLATTEN, FRESCO, Rerender-A-Video
├── 基于 T2V 模型
│   ├── VideoDirector (CVPR 2025)
│   └── Rectified Flow 编辑 (Wan-Edit, Pyramid-Edit)
├── 指令式编辑
│   ├── InstructVid2Vid, InsV2V
│   ├── InsViE-1M (ICCV 2025)
│   └── VIVA (VLM 引导 + 奖励优化)
└── I2V 传播式
    └── AnyV2V (编辑首帧 + I2V 传播)
```

### 2.3 核心技术详解

#### 2.3.1 DDIM Inversion (DDIM 反演)

扩散模型视频编辑的起点。DDIM 反演将干净的视频帧确定性地映射回其对应的噪声潜变量，反转去噪过程。反演得到的潜变量作为编辑去噪的起始点，比随机高斯噪声更好地保留了原始视频的结构和细节。

- **Null-text Inversion**: 在每个去噪步骤优化无条件嵌入以实现近乎完美的重建
- **Multi-frame Null-text Optimization** (VideoDirector, CVPR 2025): 联合优化时序线索

**局限**: 多步反演累积误差；计算昂贵；2025年的 Rectified Flow 方法完全绕过了反演。

#### 2.3.2 注意力注入与操控 (Attention Injection & Manipulation)

几乎所有基于扩散模型的视频编辑方法的核心技术：

| 技术 | 原理 | 代表方法 |
|-----|------|---------|
| **Cross-attention 控制** | 在文本 token 和空间特征之间的注意力层中，替换/混合/调制注意力图 | FateZero, Ground-A-Video |
| **Self-attention 注入** | 将源视频 DDIM 反演时的自注意力特征存储并注入编辑过程 | Pix2Video, FateZero |
| **Cross-frame attention** | 每帧关注参考帧（通常是第一帧）的特征 | Text2Video-Zero, vid2vid-zero |
| **时空注意力** | 将 2D 空间注意力扩展到时间维度 | Tune-A-Video, FateZero |

#### 2.3.3 时间一致性机制 (Temporal Consistency Mechanisms)

这是视频编辑的**核心挑战**。多种策略已被开发：

| 策略 | 原理 | 代表方法 |
|-----|------|---------|
| **Token 传播** | 基于帧间最近邻对应关系传播扩散特征 | TokenFlow (ICLR 2024) |
| **Token 合并** | 跨帧合并自注意力 token | VidToMe (CVPR 2024) |
| **光流引导** | 用光流引导注意力或特征变换 | FLATTEN, FRESCO (CVPR 2024) |
| **随机噪声混洗** | 跨帧混洗噪声模式创建隐式时空耦合 | RAVE |
| **I2V 模型传播** | 利用 I2V 模型的时序特征注入传播首帧编辑 | AnyV2V (TMLR 2024) |
| **帧间一致性损失** | 训练时惩罚连续帧的视觉不一致 | InstructVid2Vid |
| **跨窗口注意力** | 长视频分窗口处理，相邻窗口帧互相关注 | LOVECon |

#### 2.3.4 ControlNet 适配视频

ControlNet 提供额外的结构条件信号（Canny 边缘、深度图、姿态骨架等）来引导扩散过程，保持源视频结构：

- **ControlVideo**: 使用 ControlNet 逐帧强制结构保真
- **LOVECon**: 在 ControlNet 之上构建完整长视频编辑流程
- **CCEdit**: 三叉网络，分离结构分支（ControlNet）和外观分支

### 2.4 代表性方法

#### 开创性工作

**Tune-A-Video** (ICCV 2023) `arXiv:2212.11565`
- 作者: Wu et al. (NUS Show Lab / Tencent)
- 开创 One-Shot Video Tuning 范式
- 将预训练 T2I 模型的空间自注意力转换为稀疏因果时空注意力
- 推理时用 DDIM 反演获取保结构的初始潜变量

**FateZero** (ICCV 2023 Oral) `arXiv:2303.09535`
- 作者: Qi et al. (HKUST / Tencent AI Lab)
- 首个零样本文本驱动视频编辑框架
- 三个核心创新: 中间注意力捕获、自注意力融合、时空注意力改造

**TokenFlow** (ICLR 2024) `arXiv:2307.10373`
- 作者: Geyer et al. (Weizmann Institute)
- 关键洞察: 扩散特征空间中的一致性可确保输出视频的一致性
- 基于帧间最近邻对应关系传播扩散 token

#### 2024 年重要工作

| 方法 | 会议 | 核心创新 |
|-----|------|---------|
| **FRESCO** | CVPR 2024 | 基于光流和特征匹配建立时空对应关系 |
| **VidToMe** | CVPR 2024 | 跨帧 token 合并，降低内存消耗同时提升一致性 |
| **AnyV2V** | TMLR 2024 | 编辑首帧 + I2V 传播的两步范式 |
| **InsV2V** | ICLR 2024 | 40万+合成编辑对训练；LVSC 长视频处理 |

### 2.5 最新进展 (2025-2026)

**VideoDirector** (CVPR 2025) `arXiv:2411.17592`
- 首个有效利用 T2V 模型（而非 T2I）进行精确视频编辑
- Spatial-Temporal Decoupled Guidance (STDG) 分离空间和时间控制
- 解决了 T2I→T2V 扩展中的严重伪影问题

**VIVA** (2025) `arXiv:2512.16906`
- VLM 引导的指令式编辑 + Diffusion Transformer 骨干
- Edit-GRPO 奖励优化提升指令遵循、内容保真和视觉美学

**InsViE-1M** (ICCV 2025) `arXiv:2503.20287`
- 100 万三元组指令式视频编辑数据集
- 多阶段流水线 + GPT-4o 质量过滤

**Rectified Flow 编辑范式** (2025-2026)
- 从 DDIM 反演转向 Rectified Flow，完全消除反演开销
- Wan-Edit、Pyramid-Edit 在 FiVE-Bench 上显著优于扩散方法
- **训练免费、反演免费、超参数不敏感**

**关键趋势总结:**
1. 从 T2I 扩展到原生 T2V 编辑
2. 从扩散到 Flow Matching
3. VLM + 扩散模型融合
4. 训练数据规模从 40 万到 100 万
5. 评估体系从简单 CLIP 分数到多维 LMM 评估

---

## 3. 视频修复 (Video Inpainting)

### 3.1 任务定义

**视频修复 (Video Inpainting)** 是指用合理内容填充视频帧中缺失、损坏或被刻意遮蔽的区域，要求生成内容在空间上连贯（单帧自然），在时间上一致（跨帧平滑无闪烁）。

**应用场景:** 物体移除、水印/logo 去除、视频补全、视频稳像后边缘填补、视频外扩 (outpainting)。

### 3.2 经典方法

| 方法 | 年份 | 核心思路 |
|-----|------|---------|
| **Wexler et al.** | 2007 | 全局 3D 时空 patch 优化（奠基性工作） |
| **Newson et al.** | 2014 | 3D PatchMatch + 仿射运动补偿（经典方法标杆） |
| **Huang et al.** | 2016 | 联合优化光流和颜色以确保时间一致性 |

**经典方法的局限:** 计算极其昂贵、无法生成全新内容、大面积缺失表现差、缺乏语义理解。

### 3.3 深度学习方法

#### CNN 时代 (2019)

| 方法 | 会议 | 核心创新 |
|-----|------|---------|
| **Deep Video Inpainting (DVI)** | CVPR 2019 | 多尺度编解码器 + ConvLSTM 时序记忆模块 |
| **Deep Flow-Guided (DFGVI)** | CVPR 2019 | 开创"光流补全→像素传播→内容幻觉"三阶段范式 |
| **Copy-and-Paste Net (CPNet)** | ICCV 2019 | 学习从参考帧复制对应内容并粘贴到目标帧 |
| **Onion-Peel Net (OPN)** | ICCV 2019 | 从边界向内逐层填充（洋葱剥皮策略） |

#### Transformer 时代 (2020-2023)

**STTN** (ECCV 2020) — 首个 Transformer 视频修复方法
- 多尺度 patch 级时空注意力，同时填充所有输入帧
- 时空对抗损失提升感知质量

**FuseFormer** (ICCV 2021) — 解决 STTN 的模糊边缘问题
- Soft Split / Soft Composition 允许重叠 patch，实现子 patch 级信息交互

**E2FGVI** (CVPR 2022) — 首个端到端光流引导框架
- 三个联合训练模块: 光流补全、特征传播、时序焦点 Transformer
- 处理速度 0.12 秒/帧，比 FGVC 快近 15 倍
- CVPR 2022 NTIRE 视频修复挑战赛冠军

**ProPainter** (ICCV 2023) — 非扩散类 SOTA
- **双域传播**: 图像域 warp（全局对应）+ 特征域 warp（局部对应，可变形卷积）
- **掩码引导稀疏 Transformer**: 仅对掩码区域内/附近的 token 进行注意力计算
- PSNR 比前代提升 1.46 dB，广泛被认为是非扩散方法的 SOTA

#### 光流引导方法演进

```
DFGVI (CVPR 2019) → FGVC (ECCV 2020) → FGT (ECCV 2022) → E2FGVI (CVPR 2022) → FGT++ (TPAMI 2023) → ProPainter (ICCV 2023)
  预训练光流          光流边缘引导         光流引导注意力       端到端联合训练        深度光流集成         双域传播+稀疏Transformer
```

### 3.4 扩散模型方法

| 方法 | 年份/会议 | 核心创新 |
|-----|----------|---------|
| **AVID** | CVPR 2024 | Stable Diffusion + 时序注意力; Temporal MultiDiffusion 支持任意长度 |
| **FFF-VDI** | 2024 | 修复首帧 + I2V 扩散传播，无需光流估计 |
| **FloED** | 2024 | 双分支: 光流恢复 + 多尺度光流适配器引导扩散 |
| **CoCoCo** | AAAI 2025 | 文本引导视频修复，融合修复与编辑 |
| **DiffuEraser** | 2025 | ProPainter 输出作为扩散模型的先验/初始化 |
| **VideoPainter** | SIGGRAPH 2025 | 轻量上下文编码器 (6% 参数) 即插即用于任意视频 DiT; VPData 39万+ 片段 |

**VideoPainter** 代表当前扩散视频修复的 SOTA，关键创新:
- 双分支框架: 上下文编码器处理带掩码视频，注入骨干感知的背景上下文线索
- 即插即用到不同预训练 DiT 骨干，无需重新训练
- 引入 VPData/VPBench: 最大视频修复数据集（39万+ 片段）

---

## 4. 视频风格迁移 (Video Style Transfer)

### 4.1 任务定义与分类

将视频的视觉外观转换为目标风格，同时保持语义内容和时间一致性。

| 类别 | 描述 | 示例 |
|-----|------|------|
| **艺术风格迁移** | 将视频渲染为绘画/艺术风格 | Van Gogh 星空风格 |
| **写实风格迁移** | 改变光照、色调、天气等，保持照片真实感 | 白天→黑夜 |

### 4.2 代表性方法

#### 优化类方法

**Artistic Style Transfer for Videos** (Ruder et al., 2016) `arXiv:1604.08610`
- 将 Gatys 的图像风格迁移扩展到视频
- 引入光流 warp 的时序一致性损失: `L_temporal = Σ_t ‖M_t · (O_t - W(O_{t-1}, F))‖²`
- 局限: 极慢（每帧数分钟）

#### 前馈类方法

| 方法 | 核心思路 |
|-----|---------|
| **ReReVST** (IEEE TIP 2020) | 松弛+正则化，平衡逐帧风格化质量与时序平滑 |
| **MCCNet** (2022) | 多通道关联网络，特征空间融合天然保证时序一致 |
| **ArtFlow** (CVPR 2021) | 可逆神经流 (Normalizing Flows)，避免内容泄露 |

#### Transformer 类方法

| 方法 | 核心创新 |
|-----|---------|
| **StyTr²** (CVPR 2022) | 双 Transformer 编码器 + 内容感知位置编码 (CAPE) |
| **AdaAttN** (ICCV 2021) | 自适应注意力归一化，浅层+深层特征联合关注 |

### 4.3 扩散模型时代的风格迁移

| 方法 | 年份/会议 | 核心创新 |
|-----|----------|---------|
| **StyleCrafter** | ACM TOG 2024 | 参考增强适配器 + 解耦风格/内容学习 |
| **BIVDiff** | CVPR 2024 | 桥接图像扩散（高质量）和视频扩散（时序一致），训练免费 |
| **HiCAST** | CVPR 2024 | 多语义层级风格适配器 + 和谐一致性损失 |
| **StyleID** | CVPR 2024 Highlight | 训练免费，直接在扩散自注意力中注入风格 |
| **UniVST** | TPAMI 2025 | 训练免费局部化风格迁移 + 滑动窗口一致性平滑 |
| **StyleMaster** | CVPR 2025 | 运动适配器 + 灰度瓦片 ControlNet + 幻觉数据集监督 |
| **U-StyDiT** | 2025 | DiT 架构（替代 UNet）实现超高质量风格迁移 |
| **PickStyle** | 2025 | CS-CFG: 独立控制风格保真和内容保持的引导因子 |
| **TeleStyle** | 2026 | 统一于 Qwen 生态，无需风格特定 LoRA |

**关键趋势:**
- 训练免费方法占主导
- DiT 正在替代 UNet 成为下一代骨干
- 从全局风格迁移走向局部化精细控制
- 集成到大型多模态编辑流水线中

---

## 5. 视频生成基础模型 (Video Generation Foundation Models)

### 5.1 技术路线演进

```
2017-2021: GAN 时代
  MoCoGAN → DVD-GAN → DIGAN
  局限: 模式崩溃, 低分辨率, 短时长

2021-2022: 自回归/Token 时代
  VideoGPT → TATS → NUWA → MAGVIT V2
  将语言模型范式引入视频生成

2022-2023: 扩散 + UNet 时代
  VDM → Make-A-Video → Imagen Video → Video LDM → SVD
  关键模式: 2D UNet 膨胀 + 分解式时空注意力

2024-至今: Diffusion Transformer (DiT) 时代
  Latte → Sora → CogVideoX → Kling → HunyuanVideo → Open-Sora 2.0 → Veo 3
  关键模式: 时空 patch + 完整 3D 注意力 + 3D VAE
  前沿: MoE, 音视频联合生成, 消费级 GPU 推理
```

### 5.2 GAN 时代

| 模型 | 年份 | 核心思路 |
|-----|------|---------|
| **MoCoGAN** | CVPR 2018 | 将潜空间分解为内容子空间和运动子空间; GRU 生成运动码 |
| **DVD-GAN** | 2019 | 双判别器（空间+时序）; 首个在复杂数据集生成 256×256 视频 |
| **DIGAN** | ICLR 2022 | 隐式神经表示，连续时间视频生成 |

### 5.3 自回归/Token 时代

| 模型 | 年份 | 核心思路 |
|-----|------|---------|
| **VideoGPT** | 2021 | 3D VQ-VAE 离散化 + GPT-2 自回归建模 |
| **TATS** | ECCV 2022 | 时间不变 VQGAN + 时间敏感滑动窗口 Transformer |
| **VideoPoet** | ICML 2024 Best Paper | 统一 LLM 骨干处理视频/音频/文本; MAGVIT V2 tokenizer; 万亿 token 训练 |
| **MAGVIT V2** | 2023 | Lookup-Free Quantization (LFQ)，有效码本 26万+; 证明自回归 LM 可匹配扩散模型 |

### 5.4 扩散模型 + UNet 时代

**Video Diffusion Models (VDM)** (NeurIPS 2022) `arXiv:2204.03458`
- 将 2D 图像扩散 UNet 扩展到 3D
- 建立了**分解式空间-时序注意力**范式: 空间自注意力块后接时序自注意力块

**Make-A-Video** (Meta, 2022) `arXiv:2209.14792`
- 从预训练 T2I 模型出发，冻结空间层，仅插入和训练时序层
- **不需要文本-视频配对数据**，开创了极具影响力的"膨胀+微调"范式

**Imagen Video** (Google, 2022) `arXiv:2210.02303`
- 级联 7 个扩散模型: 基础生成→时序超分→空间超分
- 最终输出: 1280×768, 24fps, 128帧

**Stable Video Diffusion (SVD)** (Stability AI, 2023) `arXiv:2311.15127`
- 基于 SD 2.1 UNet，插入时序卷积和时序注意力层
- 15.2 亿参数，其中约 43% 用于时序处理
- 三阶段训练: 图像预训练 → 视频预训练（系统化过滤）→ 高质量微调
- 支持 Camera Motion LoRA

### 5.5 Diffusion Transformer (DiT) 时代

| 模型 | 参数量 | 组织 | 关键创新 |
|-----|-------|------|---------|
| **Sora** | 未公开 (估>3B) | OpenAI | 时空 patch + DiT; 世界模拟器; 可变分辨率/时长/比例 |
| **CogVideoX** | 2B / 5B | 智谱 AI | 3D VAE (8×空间, 4×时间); Expert AdaLN; 完整 3D 注意力 |
| **Kling** | 未公开 | 快手 | DiT + 3D VAE; 2 分钟, 1080p, 30fps; Kling O1 统一多模态 |
| **HunyuanVideo** | 13B | 腾讯 | 3D Causal VAE; 双流→单流混合 Transformer; MLLM 文本编码器 |
| **HunyuanVideo 1.5** | 8.3B | 腾讯 | SSTA 注意力 (1.87× 加速); 14GB VRAM 消费级推理 |
| **Open-Sora 2.0** | 11B | HPC-AI Tech | MMDiT + 完整 3D 注意力; 训练成本仅 $20 万; 完全开源 |
| **Veo 3** | 未公开 | Google | 首个原生音视频联合生成; Gemini 重标注 |
| **Wan 2.1** | 1.3B / 14B | 阿里 | 最低 8.19GB VRAM; 高度可定制 LoRA |
| **Wan 2.2** | 14B (MoE) | 阿里 | 首个开源视频 MoE 扩散架构 |

**核心技术组件:**

**时序注意力机制:**
| 类型 | 原理 | 代表模型 |
|-----|------|---------|
| 分解式时空注意力 | 先空间注意力，再时序注意力 | VDM, SVD, Make-A-Video |
| 完整 3D 时空注意力 | 所有 token 跨时空互相关注 | Sora, CogVideoX, Kling |
| SSTA | 剪枝冗余 KV 块 + 滑动瓦片 | HunyuanVideo 1.5 |

**3D VAE 设计:**
| 模型 | 空间压缩 | 时间压缩 |
|-----|---------|---------|
| CogVideoX | 8× | 4× |
| HunyuanVideo | 16× | 4× |
| SVD | 8× (2D VAE) | 无 |

### 5.6 生成模型如何应用于视频编辑

| 编辑范式 | 原理 | 代表方法 |
|---------|------|---------|
| **SDEdit** | 对源视频加噪到中间步→用目标文本去噪 | SDEdit, NC-SDEdit |
| **DDIM 反演+编辑** | 反演源视频→修改文本条件→重新去噪 | Null-text Inversion, FateZero |
| **ControlNet 控制** | 提取源视频的结构条件引导生成 | ControlVideo, Ctrl-Adapter |
| **I2V 传播** | 编辑首帧→用 I2V 模型传播到全视频 | AnyV2V, FFF-VDI |
| **V2V 直接编辑** | One-shot 微调或注意力操控 | Tune-A-Video, TokenFlow |
| **Rectified Flow** | Flow Matching 直接扰动流轨迹，无需反演 | Wan-Edit, Pyramid-Edit |

---

## 6. 视频到视频翻译 (Video-to-Video Translation)

| 方法 | 年份/会议 | 核心思路 |
|-----|----------|---------|
| **vid2vid** | NeurIPS 2018 | 序列生成器 + 时序判别器 + FlowNet; 语义图→街景/边缘图→人脸 |
| **World-Consistent vid2vid** | ECCV 2020 | 维护持久 3D 场景表示，投影为引导图像确保全局一致 |
| **Few-shot vid2vid** | NeurIPS 2019 | 少样本测试时适配，Adaptive SPADE |
| **face-vid2vid** | CVPR 2021 | 3D 关键点解耦头部姿态与面部表情，自由视角 |
| **FlowVid** | CVPR 2024 | 联合空间条件和光流，优雅处理光流误差; 比 TokenFlow 快 10.5× |
| **CoDeF** | CVPR 2024 | 规范内容场 + 时间变形场; 完美时序一致（构造保证） |

---

## 7. 相关辅助任务

### 7.1 视频超分辨率 (Video Super-Resolution)

| 方法 | 年份/会议 | 核心创新 |
|-----|----------|---------|
| **BasicVSR** | CVPR 2021 | 系统研究 VSR 四要素（传播、对齐、聚合、上采样）; 双向循环 + 光流对齐 |
| **BasicVSR++** | CVPR 2022 | 二阶网格传播 + 光流引导可变形对齐; 多项挑战赛冠军 |
| **VRT** | 2022 | 视频恢复 Transformer; 时序互惠自注意力 (TRSA); 5 个任务 14 个基准 SOTA |
| **RVRT** | NeurIPS 2022 | 循环+并行混合; 引导可变形注意力 (GDA); 效率显著优于 VRT |
| **DAM-VSR** | SIGGRAPH 2025 | 外观与运动解耦的视频超分 |

### 7.2 视频帧插值 (Video Frame Interpolation)

| 方法 | 年份/会议 | 核心创新 |
|-----|----------|---------|
| **RIFE** | ECCV 2022 | IFNet 直接估计中间光流; 特权蒸馏; 实时 (30+ FPS@720p) |
| **IFRNet** | CVPR 2022 | 联合精炼中间光流和中间特征 |
| **AMT** | CVPR 2023 | 全对双向关联体 + 多组精细光流场; 比 IFRNet 提升 0.17dB，FLOPs 仅 60% |
| **EMA-VFI** | CVPR 2023 | 帧间注意力同时提取运动和外观，注意力图双重利用 |
| **VFIMamba** | NeurIPS 2024 | Mamba (SSM) 应用于帧插值，线性复杂度 |

### 7.3 视频着色 (Video Colorization)

| 方法 | 核心创新 |
|-----|---------|
| **DeOldify** | 自注意力 GAN + 渐进式训练，广泛使用的实用工具 |
| **TCVC** (2024) | 双向深层特征传播 + 自正则化学习 (无需真值) |
| **BiSTNet** (TPAMI 2024) | 语义先验引导对应 + 双向时序特征融合; NTIRE 2023 冠军 |
| **ColorMNet** (ECCV 2024) | 记忆库存储已着色帧特征引导当前帧 |

### 7.4 视频抠图与分割 (Video Matting & Segmentation)

**视频抠图 (Matting):**
| 方法 | 年份/会议 | 核心创新 |
|-----|----------|---------|
| **MODNet** | AAAI 2022 | 三子目标分解（语义/细节/融合）; trimap-free; 实时 |
| **RVM** | 2021 | ConvGRU 循环; 无辅助输入; 实时鲁棒 |
| **MatAnyone** | CVPR 2025 | 一致性记忆传播，长视频稳定抠图 |
| **OAVM** | 2025 | 物体级理解赋能抠图; trimap-free SOTA |

**视频分割 (Segmentation):**
| 方法 | 核心创新 |
|-----|---------|
| **SAM** (Meta, 2023) | 提示式分割基础模型; SA-1B 数据集 (10 亿掩码); 零样本泛化 |
| **SAM 2** (Meta, ICLR 2025) | 统一图像+视频; 流式记忆架构; 比 SAM 准确 6×, 交互减少 3× |
| **Track Anything (TAM)** | SAM + 跟踪，交互式视频分割 |

---

## 8. 商业产品与系统

| 产品 | 公司 | 最新版本 | 关键特点 |
|-----|------|---------|---------|
| **Runway** | Runway | Gen-4 (2025) | 4K 输出, AI Magic Tools (修复/外扩/重光照); 专业级 |
| **Sora** | OpenAI | Sora 2.0 (2026) | 11B 参数; 世界模拟器; 最强物理理解 |
| **Kling** | 快手 | Kling 2.6 / O1 (2025) | 2 分钟 1080p; 音视频联合生成; 统一多模态模型 |
| **Veo** | Google | Veo 3 (2025) | 首个原生音视频联合生成; Gemini 集成 |
| **Pika** | Pika Labs | Pika 2.5 (2025) | 消费者友好; Pikaffects 特效; Scene Ingredients |
| **Luma** | Luma AI | Dream Machine / Ray | 快速生成, 强运动控制, 相机控制 |
| **Jimeng** | 字节跳动 | - | 文本转视频, TikTok 生态集成 |
| **Minimax** | MiniMax | Hailuo AI | 中文 T2V 强模型 |

---

## 9. 数据集与评估基准

### 9.1 常用数据集

**源视频数据集:**
| 数据集 | 描述 | 规模 |
|-------|------|------|
| **DAVIS** | 密集标注视频分割 | 50-150 视频, 480p |
| **YouTube-VOS** | 大规模视频目标分割 | 4,453 视频 |
| **WebVid-10M** | 网络视频-文本对 | 1000 万 |
| **HD-VILA-100M** | 视频-文本对 | 1 亿 |

**编辑专用数据集:**
| 数据集 | 规模 | 用途 |
|-------|------|------|
| **InsV2V 合成数据集** | 40 万+ 编辑对 | 训练指令式编辑模型 |
| **InsViE-1M** | 100 万三元组 | GPT-4o 过滤的高质量指令编辑训练 |
| **VPData** (VideoPainter) | 39 万+ 片段 | 最大视频修复数据集 |

### 9.2 评估指标

**文本-视频对齐:**
| 指标 | 描述 |
|-----|------|
| **CLIP-T** | 编辑帧 CLIP 图像嵌入与目标文本嵌入的余弦相似度 |
| **CLIP Direction Score** | 图像空间和文本空间编辑方向的余弦相似度 |
| **ViCLIP Score** | 时序感知视频级 CLIP 评分 |
| **FiVE-Acc** | VLM 视觉问答式精细编辑评估 |

**时序一致性:**
| 指标 | 描述 |
|-----|------|
| **CLIP Frame Consistency** | 连续帧对的 CLIP 余弦相似度均值 |
| **Warping Error** | 用光流 warp 编辑帧到下一帧，衡量像素差异 |

**视频质量:**
| 指标 | 描述 |
|-----|------|
| **FVD** | Frechet Video Distance，衡量生成视频与真实视频分布距离 |
| **PSNR / SSIM / LPIPS** | 像素级/结构/感知距离指标 |

**多维评估工具 (2025):**
| 工具 | 描述 |
|-----|------|
| **TDVE-Assessor** | LMM 基 (Qwen2.5-VL-7B) 三维评估: 编辑质量、对齐度、结构一致性 |
| **FiVE-Acc** | VLM 视觉问答式编辑成功率 |

### 9.3 专用评测基准

| 基准 | 年份/会议 | 规模 | 特点 |
|-----|----------|------|------|
| **LOVEU-TGVE** | CVPR 2023 Workshop | 76 视频, 4 编辑类别 | 首个大规模文本视频编辑基准; ViCLIP 指标 |
| **FiVE-Bench** | ICCV 2025 | 100 视频, 420 提示对, 14 指标 | 首次基准测试 Rectified Flow 编辑; RF 方法显著优于扩散方法 |
| **TDVE-DB** | 2025 | 3,857 编辑视频, 17.3 万+ 人类评分 | 12 个模型, 8 个编辑类别, 三维人类标注 |
| **VBench** | CVPR 2024 | 16 维度评估 | 综合视频生成质量评估 |

---

## 10. 开源模型与框架

### 视频生成模型

| 模型 | 参数量 | 许可 | 亮点 |
|-----|-------|------|------|
| **HunyuanVideo 1.5** | 8.3B | 开源 | 消费级 GPU (14GB); SSTA 高效推理 |
| **Wan 2.1** | 1.3B / 14B | 开源 | 最易获取 (8.19GB VRAM); LoRA 友好 |
| **CogVideoX** | 2B / 5B | Apache-2.0 | 强 I2V; 完整 3D 注意力 |
| **Open-Sora 2.0** | 11B | 开源 | 完整训练配方; $20 万成本 |
| **SVD** | 1.52B | 研究 | 成熟生态; LoRA 支持 |
| **AnimateDiff** | ~400M | Apache-2.0 | 即插即用运动模块; SD 生态兼容 |

### 视频编辑框架

| 框架 | 训练需求 | 特点 |
|-----|---------|------|
| **TokenFlow** | 无 (训练免费) | 特征传播; 兼容任意 T2I 编辑方法 |
| **FateZero** | 无 (零样本) | 注意力融合; 支持风格/属性/形状编辑 |
| **Rerender-A-Video** | 无 (训练免费) | 关键帧编辑 + 光流传播 |
| **AnyV2V** | 无 (调优免费) | 解耦编辑和一致性问题 |

### 基础设施

| 平台 | 用途 |
|-----|------|
| **ComfyUI** | 节点式可视化工作流; 支持所有主流视频模型 |
| **HuggingFace Diffusers** | Python 库; 内置视频扩散管线 |
| **xDiT** | 大型视频 DiT 分布式推理 |

---

## 11. 关键挑战与未来方向

### 当前核心挑战

| 挑战 | 描述 |
|-----|------|
| **时序一致性** | 逐帧编辑产生闪烁和不连续；复杂运动场景下尤为困难 |
| **运动保持** | 编辑必须保留原始运动动态，避免引入非自然运动 |
| **身份保持** | 编辑属性时保持主体身份（面部、体型等）一致 |
| **计算成本** | DDIM 反演、自注意力的二次复杂度、长视频窗口化处理 |
| **编辑精度** | 确保编辑仅限于目标区域，不影响无关区域 |
| **T2I→T2V 扩展鸿沟** | 直接从 T2I 扩展到 T2V 产生严重伪影 |
| **评估标准化** | 缺乏统一公认的评测协议 |

### 未来趋势

1. **从 T2I 扩展到原生 T2V 编辑**: VideoDirector (CVPR 2025) 开启新范式
2. **从扩散到 Flow Matching**: Rectified Flow 消除反演，减少超参敏感性
3. **VLM 融合编辑**: VIVA 代表 VLM + 扩散模型协同方向
4. **消费级部署**: HunyuanVideo 1.5 (14GB)、Wan 2.1 (8.19GB) 让研究更可及
5. **MoE 架构**: Wan 2.2 首创视频扩散 MoE，容量扩展不成比例增加计算
6. **音视频联合生成**: Veo 3、Kling 2.6 原生音视频联合扩散
7. **多维 LMM 评估**: 从 CLIP 分数到 TDVE-Assessor 多维自动评估
8. **实时编辑**: PAB (21.6 FPS)、SSTA (1.87× 加速) 缩小与实时的差距

---

## 12. 参考文献

### 文本引导视频编辑
- Wu et al. "Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation." ICCV 2023. [arXiv:2212.11565](https://arxiv.org/abs/2212.11565)
- Qi et al. "FateZero: Fusing Attentions for Zero-shot Text-based Video Editing." ICCV 2023 (Oral). [arXiv:2303.09535](https://arxiv.org/abs/2303.09535)
- Geyer et al. "TokenFlow: Consistent Diffusion Features for Consistent Video Editing." ICLR 2024. [arXiv:2307.10373](https://arxiv.org/abs/2307.10373)
- Khachatryan et al. "Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators." ICCV 2023 (Oral). [arXiv:2303.13439](https://arxiv.org/abs/2303.13439)
- Li et al. "VidToMe: Video Token Merging for Zero-Shot Video Editing." CVPR 2024. [arXiv:2312.10656](https://arxiv.org/abs/2312.10656)
- Yang et al. "Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation." SIGGRAPH Asia 2023. [arXiv:2306.07954](https://arxiv.org/abs/2306.07954)
- "FLATTEN: Optical Flow-guided Attention for Consistent Text-to-Video Editing." ICLR 2024. [arXiv:2310.05922](https://arxiv.org/abs/2310.05922)
- "FRESCO: Spatial-Temporal Correspondence for Zero-Shot Video Translation." CVPR 2024. [arXiv:2403.12962](https://arxiv.org/abs/2403.12962)
- "AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks." TMLR 2024. [arXiv:2403.14468](https://arxiv.org/abs/2403.14468)
- Wang et al. "VideoDirector: Precise Video Editing via Text-to-Video Models." CVPR 2025. [arXiv:2411.17592](https://arxiv.org/abs/2411.17592)
- "VIVA: VLM-Guided Instruction-Based Video Editing with Reward Optimization." 2025. [arXiv:2512.16906](https://arxiv.org/abs/2512.16906)
- "InsViE-1M: Effective Instruction-based Video Editing with Elaborate Dataset Construction." ICCV 2025. [arXiv:2503.20287](https://arxiv.org/abs/2503.20287)
- Wu et al. "InstructVid2Vid: Controllable Video Editing with Natural Language Instructions." ICME 2024. [arXiv:2305.12328](https://arxiv.org/abs/2305.12328)
- "InsV2V: Consistent Video-to-Video Transfer Using Synthetic Dataset." ICLR 2024.
- Feng et al. "CCEdit: Creative and Controllable Video Editing via Diffusion Models." [arXiv:2309.16496](https://arxiv.org/abs/2309.16496)
- "LOVECon: Text-driven Training-Free Long Video Editing with ControlNet." [arXiv:2310.09711](https://arxiv.org/abs/2310.09711)

### 视频修复
- Wexler et al. "Space-Time Completion of Video." TPAMI 2007.
- Newson et al. "Video Inpainting of Complex Scenes." SIAM J. Imaging Sci. 2014.
- Xu et al. "Deep Flow-Guided Video Inpainting." CVPR 2019.
- Lee et al. "Copy-and-Paste Networks for Deep Video Inpainting." ICCV 2019.
- Zeng et al. "STTN: Learning Joint Spatial-Temporal Transformations for Video Inpainting." ECCV 2020.
- Liu et al. "FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting." ICCV 2021.
- Li et al. "E2FGVI: Towards An End-to-End Framework for Flow-Guided Video Inpainting." CVPR 2022.
- Gao et al. "FGVC: Flow-edge Guided Video Completion." ECCV 2020.
- Zhou et al. "ProPainter: Improving Propagation and Transformer for Video Inpainting." ICCV 2023.
- Zhang et al. "AVID: Any-Length Video Inpainting with Diffusion Model." CVPR 2024.
- Li et al. "DiffuEraser: A Diffusion Model for Video Inpainting." 2025. [arXiv:2501.10018](https://arxiv.org/abs/2501.10018)
- "VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control." SIGGRAPH 2025.

### 视频风格迁移
- Ruder et al. "Artistic Style Transfer for Videos." 2016. [arXiv:1604.08610](https://arxiv.org/abs/1604.08610)
- Deng et al. "StyTr2: Image Style Transfer with Transformers." CVPR 2022.
- "StyleCrafter: Taming Artistic Video Diffusion with Reference-Augmented Adapter Learning." ACM TOG 2024.
- "BIVDiff: Bridging Image and Video Diffusion Models." CVPR 2024.
- "HiCAST: Highly Customized Arbitrary Style Transfer with Adapter Enhanced Diffusion Models." CVPR 2024.
- "StyleID: Style Injection in Diffusion." CVPR 2024 (Highlight).
- "UniVST: A Unified Framework for Training-free Localized Video Style Transfer." TPAMI 2025.
- "StyleMaster: Stylize Your Video with Artistic Generation and Translation." CVPR 2025.

### 视频生成基础模型
- Ho et al. "Video Diffusion Models." NeurIPS 2022. [arXiv:2204.03458](https://arxiv.org/abs/2204.03458)
- Singer et al. "Make-A-Video: Text-Free Text-to-Video Generation." 2022. [arXiv:2209.14792](https://arxiv.org/abs/2209.14792)
- Ho et al. "Imagen Video: High Definition Video Generation with Diffusion Models." 2022. [arXiv:2210.02303](https://arxiv.org/abs/2210.02303)
- Blattmann et al. "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets." 2023. [arXiv:2311.15127](https://arxiv.org/abs/2311.15127)
- Yang et al. "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer." ICLR 2025. [arXiv:2408.06072](https://arxiv.org/abs/2408.06072)
- "Sora: Video Generation Models as World Simulators." OpenAI, 2024.
- Kong et al. "HunyuanVideo: A Systematic Framework for Large Video Generative Models." 2024. [arXiv:2412.03603](https://arxiv.org/abs/2412.03603)
- "HunyuanVideo 1.5 Technical Report." 2025. [arXiv:2511.18870](https://arxiv.org/abs/2511.18870)
- "Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k." 2025. [arXiv:2503.09642](https://arxiv.org/abs/2503.09642)
- Kondratyuk et al. "VideoPoet: A Large Language Model for Zero-Shot Video Generation." ICML 2024 (Best Paper). [arXiv:2312.14125](https://arxiv.org/abs/2312.14125)

### 视频到视频翻译
- Wang et al. "Video-to-Video Synthesis." NeurIPS 2018. [arXiv:1808.06601](https://arxiv.org/abs/1808.06601)
- Mallya et al. "World-Consistent Video-to-Video Synthesis." ECCV 2020.
- Liang et al. "FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis." CVPR 2024. [arXiv:2312.17681](https://arxiv.org/abs/2312.17681)

### 视频超分辨率
- Chan et al. "BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond." CVPR 2021.
- Chan et al. "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment." CVPR 2022.
- Liang et al. "VRT: A Video Restoration Transformer." 2022.
- Liang et al. "RVRT: Recurrent Video Restoration Transformer with Guided Deformable Attention." NeurIPS 2022.

### 视频帧插值
- Huang et al. "RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation." ECCV 2022.
- Li et al. "AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation." CVPR 2023.
- Zhang et al. "EMA-VFI: Extracting Motion and Appearance via Inter-Frame Attention." CVPR 2023.

### 视频抠图与分割
- Ke et al. "MODNet: Real-Time Trimap-Free Portrait Matting." AAAI 2022.
- Ravi et al. "SAM 2: Segment Anything in Images and Videos." ICLR 2025. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)

### 综述
- "Diffusion Model-Based Video Editing: A Survey." 2024. [arXiv:2407.07111](https://arxiv.org/abs/2407.07111)
- "A Survey on Video Diffusion Models." [arXiv:2310.10647](https://arxiv.org/abs/2310.10647)

### 评测基准
- "FiVE-Bench." ICCV 2025. [arXiv:2503.13684](https://arxiv.org/abs/2503.13684)
- "TDVE-Assessor." 2025. [arXiv:2505.19535](https://arxiv.org/abs/2505.19535)
- "VBench." CVPR 2024. [GitHub](https://github.com/Vchitect/VBench)

---

> **文档维护:** 本文档将持续更新，欢迎团队成员补充最新进展。
> **最后更新:** 2026-02-11
