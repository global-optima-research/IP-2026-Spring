# PVTT数据集构建技术综述

## 目录

1. [项目背景与数据管线概述](#1-项目背景与数据管线概述)
2. [视频预处理与场景分割](#2-视频预处理与场景分割)
3. [视频目标分割](#3-视频目标分割)
4. [视频修复与背景恢复](#4-视频修复与背景恢复)
5. [视频目标合成与组合](#5-视频目标合成与组合)
6. [数据质量评估与过滤](#6-数据质量评估与过滤)
7. [电商视频数据集与合成训练数据](#7-电商视频数据集与合成训练数据)
8. [管线配置建议与总结](#8-管线配置建议与总结)
9. [参考文献](#9-参考文献)

---

## 1. 项目背景与数据管线概述

### 1.1 PVTT项目简介

PVTT (Product Video Template Transfer) 是一个面向电商场景的产品视频智能编辑系统。其核心任务为：给定源视频、参考产品图像和掩码，输出将源视频中产品替换为参考产品的编辑视频。

### 1.2 数据集构建管线

数据集构建分为四个核心阶段：

```
阶段1: 预处理
  ├── 镜头检测与场景分割 (TransNetV2, PySceneDetect)
  ├── 产品分割 (SAM2 / Grounded-SAM2)
  └── 背景恢复 (VideoPainter)

阶段2: 交叉配对
  └── 模板视频 × 电商产品图像 → 训练对

阶段3: 视频合成
  └── VideoAnyDoor / InsertAnywhere → 合成Ground Truth

阶段4: 质量过滤
  ├── CLIP-I / DINO-I 身份保持评分
  ├── MUSIQ / DOVER 质量评估
  ├── 时序一致性检测
  └── VLM (Qwen2-VL) 语义评估
```

**目标规模**: 500K-1M候选对 → 经70-75%过滤 → 100K-300K高质量训练三元组

---

## 2. 视频预处理与场景分割

### 2.1 PySceneDetect

| 字段 | 信息 |
|------|------|
| **官网** | [scenedetect.com](https://www.scenedetect.com/) |
| **代码** | [github.com/Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) |
| **版本** | 0.6.7.1 |

#### 核心检测器

| 检测器 | 机制 | 关键参数 | 适用场景 |
|--------|------|----------|----------|
| **ContentDetector** | HSV色彩空间加权像素变化 | threshold=27.0, weights, min_scene_len=15 | 硬切 |
| **AdaptiveDetector** | 两遍处理：ContentDetector分数 + 滑动窗口平均 | adaptive_threshold=3.0, window_width=2 | 镜头运动 |
| **ThresholdDetector** | RGB平均像素强度变化 | threshold, min_scene_len | 淡入/淡出 |
| **HistogramDetector** | HSV直方图变化 | — | 快速切换 |
| **HashDetector** | 感知图像哈希 | — | 快速切换 |

**优势**: 易部署、速度快、无需GPU、Python/OpenCV实现
**局限**: SHOT数据集上F1 < 0.6，对渐变过渡处理较弱

**与PVTT的关联**: 作为初始粗粒度镜头分割工具，VideoPainter的VPData管线直接使用了PySceneDetect。

### 2.2 TransNetV2

| 字段 | 信息 |
|------|------|
| **论文** | TransNet V2: An effective deep network architecture for fast shot transition detection |
| **arXiv** | [2008.04838](https://arxiv.org/abs/2008.04838) |
| **代码** | [github.com/soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) |

#### 技术架构

- 带有批归一化和跳跃连接的深层3D卷积网络
- 双分类头：单帧头（正类权重5×）和全帧头（贡献折扣0.1）
- 在ClipShots、BBC、RAI基准上达到SOTA
- 提供预训练模型，可直接部署

**与PVTT的关联**: 比PySceneDetect更准确，尤其在渐变过渡方面。推荐策略：PySceneDetect做快速粗检测，TransNetV2做精确验证。

### 2.3 AutoShot

| 字段 | 信息 |
|------|------|
| **论文** | AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection |
| **arXiv** | [2304.06116](https://arxiv.org/abs/2304.06116) |
| **会议** | CVPR NAS Workshop 2023 |
| **代码** | [github.com/wentaozhu/AutoShot](https://github.com/wentaozhu/AutoShot) |

- 基于NAS的3D卷积/Transformer混合架构
- 提出SHOT数据集：853个短视频，11,606个镜头标注
- F1提升：SHOT +4.2%, ClipShots +1.1%, BBC +0.9%, RAI +1.2%

**与PVTT的关联**: 当前短视频镜头检测SOTA，特别适合电商短视频场景。

---

## 3. 视频目标分割

### 3.1 SAM 2 (Segment Anything Model 2)

| 字段 | 信息 |
|------|------|
| **论文** | SAM 2: Segment Anything in Images and Videos |
| **arXiv** | [2408.00714](https://arxiv.org/abs/2408.00714) |
| **机构** | Meta AI (FAIR) |
| **代码** | [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2) |

#### 完整架构

**图像编码器**:
- MAE预训练的Hiera分层架构
- 特征金字塔网络融合Stage 3和Stage 4的stride-16/stride-32特征
- 窗口化绝对位置嵌入
- 变体: Tiny (T), Small (S), Base+ (B+), Large (L)
- 空间分辨率: 512, 768, 1024像素

**记忆注意力 (默认L=4个Transformer块)**:
- 每块模式: 自注意力 → 对记忆+对象指针的交叉注意力 → MLP
- 2D空间旋转位置编码 (RoPE)

**提示编码器与掩码解码器**:
- 支持点击、边界框、掩码作为提示
- "双向"Transformer块更新提示和帧嵌入
- **遮挡预测头**: 额外token + MLP产生可见性概率 — 处理目标临时消失

**记忆编码器**:
- 复用Hiera图像嵌入，无需独立编码器
- 通过卷积下采样输出掩码
- 投影至64维记忆特征

**记忆库**:
- FIFO队列: 最近N=6帧的记忆
- 独立FIFO队列: 最多M个提示帧
- 存储空间特征图 + 轻量对象指针向量 (256维，分为4×64维token)

#### 视频传播流程

1. 图像编码器处理当前帧（整个交互会话只需一次）
2. 提供无条件特征嵌入作为token
3. 记忆注意力将帧嵌入条件化于记忆库
4. 解码器接受条件化帧嵌入 + 可选提示 → 输出分割掩码
5. 记忆编码器转换预测 + 图像嵌入用于后续帧
6. 掩码下采样、融合、存入FIFO队列
7. 对象指针token存储用于后续帧的交叉注意力

#### SA-V数据集

- **规模**: 50.9K视频, 642.6K masklets, 35.5M掩码 (比任何先前VOS数据集大53倍)
- **三阶段数据引擎**:
  - Phase 1 (SAM逐帧): 37.8秒/帧, 16K masklets
  - Phase 2 (SAM + SAM2掩码): 7.4秒/帧 (5.1×加速)
  - Phase 3 (完整SAM2): 4.5秒/帧 (8.4×加速)

#### 基准性能

| 基准 | J&F |
|------|-----|
| DAVIS 2017 val | 90.9-91.6 |
| YouTube-VOS 2019 | 88.4-89.1 |
| MOSE val | 75.8-77.2 |
| 推理速度 | Hiera-B+ @ 1024: 43.8 FPS; Hiera-L: 30.2 FPS |

**与PVTT的关联**: 主要的视频掩码传播工具。Grounded-SAM2检测产品后，SAM2的流式记忆架构将分割掩码传播到所有后续帧。遮挡预测头对电商视频中产品被部分遮挡的场景至关重要。

### 3.2 Grounded-SAM2

| 字段 | 信息 |
|------|------|
| **论文** | Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks |
| **arXiv** | [2401.14159](https://arxiv.org/abs/2401.14159) |
| **代码** | [github.com/IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |

#### 工作流程

1. **文本提示** (如"handbag", "shoe", "product") → **Grounding DINO** 检测边界框
2. SAM 2 从检测框生成分割掩码
3. SAM 2 视频传播跨所有帧跟踪掩码

#### Grounding DINO 1.5 ([2405.10300](https://arxiv.org/abs/2405.10300))

- **Pro模型**: COCO 54.3 AP, LVIS-minival 零样本55.7 AP
- **Edge模型**: TensorRT下75.2 FPS, LVIS-minival零样本36.2 AP
- 深度早期融合架构，20M+带标注图像训练

**与PVTT的关联**: 管线第一步。文本提示 → Grounding DINO检测产品边界框 → SAM2生成并传播分割掩码。55.7 AP LVIS零样本意味着可处理多种电商产品类别。

### 3.3 其他视频目标分割方法

**Cutie** (CVPR 2024 Highlight) - [GitHub](https://github.com/hkchengrex/Cutie):
- 对象记忆 + 对象Transformer实现双向信息交互
- 在严重遮挡场景下优于XMem，可作为后备方案

**XMem** ([2207.07042](https://arxiv.org/abs/2207.07042), ECCV 2022):
- Atkinson-Shiffrin记忆模型：感觉记忆、工作记忆和长期记忆
- 三个独立记忆库，擅长长视频处理

**DEVA** - "Tracking Anything with Decoupled Video Segmentation":
- 解耦为图像级分割 + 时序传播
- 允许插入任意通用图像分割模型

---

## 4. 视频修复与背景恢复

### 4.1 VideoPainter

| 字段 | 信息 |
|------|------|
| **论文** | VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control |
| **arXiv** | [2503.05639](https://arxiv.org/abs/2503.05639) |
| **会议** | SIGGRAPH 2025 |
| **机构** | TencentARC |
| **代码** | [github.com/TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter) |

#### 双分支架构

- **上下文编码器**: 仅使用预训练DiT的前2层（骨干参数的6%）。处理拼接输入：噪声潜变量 + 掩码视频潜变量 + 下采样掩码
- **Token选择性集成**: 仅纯背景token被添加回去；前景token被排除。基于分割掩码的预过滤防止前景-背景歧义
- **冻结DiT骨干** + 可训练上下文编码器

#### 目标区域ID重采样（任意长度视频）

- 训练阶段: 冻结DiT + 上下文编码器; 可训练ID-Resample Adapters (LoRA)
- 推理阶段: 前一片段的修复区域token拼接到当前KV对，维持长视频ID一致性

#### VPData数据集 (390K+片段, >866.7小时)

- **收集**: ~450K视频来自Videvo和Pexels
- **标注管线 (5步)**:
  1. 收集: 获取原始视频
  2. 标注: 级联式 — Recognize Anything Model → Grounding DINO → SAM2
  3. 过滤: 帧间掩码面积变化 Δ < 20%，帧覆盖率30-70%
  4. 分割: PySceneDetect场景转换，10秒间隔，丢弃<6秒片段
  5. 选择: LAION美学评分，RAFT光流（运动强度），SD安全检查

#### 性能

| 基准 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| VPBench-S | 23.32 | 0.89 | 6.85e-2 |
| VPBench-L | 22.19 | 0.85 | — |
| DAVIS | 25.27 | 0.94 | — |

- 用户研究 (30人): 背景保持74.2%, 文本对齐82.5%, 视频质量87.4%
- **即插即用**: 可与任何预训练DiT（T2V和I2V）配合使用

**与PVTT的关联**: 主要的背景恢复工具。给定Grounded-SAM2的产品掩码，VideoPainter修复被遮挡区域生成时序一致的背景。其即插即用设计可与Wan2.1/2.2骨干配合。VPData构建管线（RAM + Grounding DINO + SAM2 + PySceneDetect + 美学评分）与PVTT计划的工具链几乎完全一致。

### 4.2 ProPainter

| 字段 | 信息 |
|------|------|
| **论文** | ProPainter: Improving Propagation and Transformer for Video Inpainting |
| **arXiv** | [2309.03897](https://arxiv.org/abs/2309.03897) |
| **会议** | ICCV 2023 |
| **代码** | [github.com/sczhou/ProPainter](https://github.com/sczhou/ProPainter) |

#### 三组件架构

1. **递归光流补全**: 高效递归网络补全损坏的光流场
2. **双域传播**: 结合图像扭曲和特征扭曲利用全局对应关系
3. **掩码引导稀疏视频Transformer**: 基于掩码引导丢弃不必要窗口实现高效

**性能**: 比先前方法提升 **1.46 dB PSNR**

**与PVTT的关联**: 基于光流的替代/补充方案。适合短片段或需要确定性结果时使用。非生成式方法，对干净背景恢复可能更优。

### 4.3 其他视频修复方法

**DiffuEraser** ([2501.10018](https://arxiv.org/abs/2501.10018), 2025年1月):
- 基于Stable Diffusion + 辅助BrushNet分支
- 使用ProPainter输出作为先验初始化

**EraserDiT** ([2506.12853](https://arxiv.org/abs/2506.12853), 2025年6月):
- 基于DiT的视频修复 + **环形位置偏移**策略
- 自动检测目标、交互式移除

---

## 5. 视频目标合成与组合

### 5.1 VideoAnyDoor

| 字段 | 信息 |
|------|------|
| **论文** | VideoAnydoor: High-fidelity Video Object Insertion with Precise Motion Control |
| **arXiv** | [2501.01427](https://arxiv.org/abs/2501.01427) |
| **会议** | SIGGRAPH 2025 |
| **机构** | HKU, DAMO Academy / Alibaba Group |
| **代码** | [github.com/yuanpengtu/VideoAnydoor](https://github.com/yuanpengtu/VideoAnydoor) |

#### 技术架构

- **基础模型**: SDXL + 时序运动模块 → 3D U-Net
- **输入**: 9通道张量 (噪声潜变量 + 掩码视频潜变量 + 掩码)
- **ID提取器**: DINOv2视觉编码器 → 全局token (1×1536) + patch tokens (256×1536) → 257×1024 → 通过交叉注意力注入每层U-Net
- **像素扭曲器 (Pixel Warper)**: 内容编码器 + 运动编码器 → 交叉注意力融合到U-Net → 精细细节保持 + 用户可控运动
- **ControlNet分支**: 从Pixel Warper输出提取多尺度中间特征

#### 关键点轨迹管线

1. X-Pose检测首帧关键点
2. NMS过滤密集点
3. 运动跟踪计算路径长度
4. 保留前N=8个最大运动点
5. 不同颜色分配到轨迹图

#### 训练

- **数据集**: YouTubeVOS, YouTubeVIS, UVO, MOSE, VIPSeg, SAM2, Pexels, MVImgNet等 (总计346K+视频)
- **分辨率**: 512×512
- **迭代**: 120K次, 16×A100, batch 32
- **采样**: DDIM 50步, CFG=10.0

#### 性能

| 方法 | PSNR | CLIP-Score | DINO-Score |
|------|------|-----------|-----------|
| ReVideo | 33.5 | 74.2 | 51.7 |
| **VideoAnyDoor** | **38.0** | **81.4** | **59.1** |

用户研究: 质量3.75/4.0, 保真度3.80/4.0, 平滑度3.65/4.0

**与PVTT的关联**: **主要的Ground Truth生成器**。提供产品图像作为参考，指定边界框序列和关键点轨迹，VideoAnyDoor生成身份一致且运动合理的合成视频。DINOv2 ID提取范式直接适用于产品身份保持。

### 5.2 InsertAnywhere

| 字段 | 信息 |
|------|------|
| **论文** | InsertAnywhere: Bridging 4D Scene Geometry and Diffusion Models for Realistic Video Object Insertion |
| **arXiv** | [2512.17504](https://arxiv.org/abs/2512.17504) |
| **日期** | 2025年12月 |
| **机构** | DAVIAN Lab, KAIST |
| **代码** | [github.com/myyzzzoooo/InsertAnywhere](https://github.com/myyzzzoooo/InsertAnywhere) |

#### 两阶段管线

**阶段1: 4D感知掩码生成**
- 基于Uni4D进行4D场景重建
- 单视图→3D点云，刚体变换 → 用户可调缩放、旋转、平移
- 场景流传播 + SAM2生成时序对齐的二值掩码序列

**阶段2: 扩散合成**
- **骨干**: Wan2.1-VACE-14B（与PVTT目标架构完全一致！）
- **微调**: LoRA rank 128
- **训练**: 5,000次迭代, lr=1e-4, 单卡H200, ~40小时
- **分辨率**: 832×480, 81帧/片段
- **推理**: 50去噪步

#### ROSE++数据集

- 从ROSE目标移除数据集扩展为插入三元组
- 每个样本: (目标移除视频[源], 目标存在视频[目标], 二值掩码, 参考目标图像)
- 参考图像通过VLM + 白底生成
- DINO相似度排序防止复制粘贴伪影

#### 性能

| 指标 | InsertAnywhere | Kling | Pika-Pro |
|------|---------------|-------|----------|
| CLIP-I | 0.8122 | 0.6349 | 0.6023 |
| DINO-I | 0.5678 | 0.5028 | 0.4012 |
| 背景一致性 | 0.9429 | 0.9335 | 0.9253 |
| 主体一致性 | 0.9520 | 0.9494 | 0.9449 |

**与PVTT的关联**: **关键参考** — 直接使用Wan2.1-VACE-14B作为骨干，与PVTT目标架构完全一致。其4D感知方法自然处理遮挡和光照。ROSE++数据构建方法论直接可用于PVTT管线设计。

### 5.3 GenCompositor

| 字段 | 信息 |
|------|------|
| **论文** | GenCompositor: Generative Video Compositing with Diffusion Transformer |
| **arXiv** | [2509.02460](https://arxiv.org/abs/2509.02460) |
| **会议** | ICLR 2026 |
| **机构** | TencentARC |
| **代码** | [github.com/TencentARC/GenCompositor](https://github.com/TencentARC/GenCompositor) |

#### 技术架构

- **骨干**: 6B参数DiT, 42个DiT融合块
- **DiT融合块**: 在token级别拼接噪声潜变量和前景条件（非通道级别），然后自注意力
- **扩展旋转位置嵌入 (ERoPE)**: 为两个未对齐视频在高度维度分配唯一位置标签
- **背景保持分支 (BPBranch)**: 2个标准DiT块处理拼接的掩码视频
- **前景增强**: Gamma校正 (γ ∈ [0.4, 1.9]) 实现自适应亮度协调

#### VideoComp数据集 (61K视频集)

- 来源: 409K源视频 (240K电影HD + 169K Tiger200K)
- 标注: CogVLM + QWen + Grounded SAM2
- 过滤: 排除无显著目标视频，前景居中处理

#### 性能

| 基准 | PSNR | SSIM | CLIP |
|------|------|------|------|
| HYouTube | 42.00 | 0.9487 | 0.9713 |

推理: ~65s, 480×720, 49帧, 34GB VRAM

**与PVTT的关联**: DiT原生的合成方法，与Wan2.1/2.2架构直接兼容。ERoPE处理前景/背景未对齐问题对产品插入至关重要。

### 5.4 Insert Anything (图像级)

| 字段 | 信息 |
|------|------|
| **论文** | Insert Anything: Image Insertion via In-Context Editing in DiT |
| **arXiv** | [2504.15009](https://arxiv.org/abs/2504.15009) |
| **代码** | [github.com/song-wensong/insert-anything](https://github.com/song-wensong/insert-anything) |

- **骨干**: FLUX.1 Fill [dev] (DiT修复模型)
- **编码器**: T5 + SigLIP, LoRA rank 256微调
- **上下文提示**: 掩码提示二联画 + 文本提示三联画
- **AnyInsertion数据集**: 159,908样本, 来源包括电商网站

**与PVTT的关联**: 图像级方法，但DiT上下文编辑范式与Wan2.1/2.2 DiT骨干对齐。AnyInsertion的电商数据采集方法可直接借鉴。

### 5.5 其他合成方法

**AnyDoor** (ECCV 2024) - 零样本目标级图像定制:
- DINOv2 ID提取器 + 频率感知细节提取器
- VideoAnyDoor的图像级前身

**ObjectMate** ([2412.08645](https://arxiv.org/abs/2412.08645)):
- 利用"重复先验" — 量产产品在大规模数据集中反复出现
- 收集**4.5M个对象**，多视角、多场景、多光照
- **与PVTT高度相关**: 电商产品天然是量产品，此方法的数据采集策略可补充交叉配对

**ObjectDrop** (ECCV 2024):
- 通过物理拍摄创建反事实训练数据
- 正确处理阴影、反射等物理效果

**Paint by Inpaint** ([2404.18212](https://arxiv.org/abs/2404.18212)):
- "反向修复"数据构建范式 — 先修复移除目标，创建(原始, 移除后)对

**Point2Insert** ([2602.04167](https://arxiv.org/abs/2602.04167), 2026年2月):
- 稀疏点控制的视频目标插入（无需密集掩码）

---

## 6. 数据质量评估与过滤

### 6.1 CLIP-I与DINO-I相似度

#### CLIP-I (基于CLIP的图像相似度)

- 计算参考产品图像与合成帧中产品区域的CLIP嵌入余弦相似度
- 侧重**语义**相似性，512/768维特征空间
- 在困难相似度基准上准确率28.45%

#### DINO-I (基于DINOv2的图像相似度)

- 使用DINOv2自监督特征计算余弦相似度
- 捕获更**细粒度**视觉结构（纹理、形状、标记）
- 准确率**64%** — 比CLIP高2.25倍

#### DiffSim ([2412.14580](https://arxiv.org/abs/2412.14580))

- 使用扩散模型特征进行相似度评估
- 解决CLIP和DINO在外观细节评估上的局限

**推荐PVTT策略**:
- **CLIP-I ≥ 阈值1**: 语义过滤 — "这是同一产品类别吗？"
- **DINO-I ≥ 阈值2**: 身份过滤 — "细粒度视觉细节是否保持？"
- 两个指标逐帧在合成产品区域（掩码内）计算

### 6.2 视频质量评估

#### MUSIQ (ICCV 2021)

| 字段 | 信息 |
|------|------|
| **论文** | MUSIQ: Multi-scale Image Quality Transformer |
| **arXiv** | [2108.05997](https://arxiv.org/abs/2108.05997) |

- 多尺度图像表示：原始分辨率 + 保持宽高比的缩放变体
- 哈希2D空间嵌入 + 尺度嵌入
- **关键优势**: 处理原始分辨率图像，无需resize
- 在PaQ-2-PiQ, SPAQ, KonIQ-10k上达到SOTA

**用途**: 逐帧质量评分，过滤模糊、失真或伪影严重的合成帧

#### DOVER (ICCV 2023)

| 字段 | 信息 |
|------|------|
| **论文** | Exploring Video Quality Assessment on User Generated Contents |
| **arXiv** | [2211.04894](https://arxiv.org/abs/2211.04894) |
| **代码** | [github.com/VQAssessment/DOVER](https://github.com/VQAssessment/DOVER) |

- **双分支**: 美学质量评估器 (AQE) + 技术质量评估器 (TQE)
- 0.91 SRCC on KoNViD-1k, 0.89 SRCC on LSVQ

**用途**: 视频级质量评分，同时提供美学和技术分数

#### FAST-VQA (ECCV 2022)

| 字段 | 信息 |
|------|------|
| **论文** | FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling |
| **arXiv** | [2207.02595](https://arxiv.org/abs/2207.02595) |
| **代码** | [github.com/timothyhtimothy/FAST-VQA](https://github.com/timothyhtimothy/FAST-VQA) |

- 网格小块采样 (GMS) + 片段注意力网络 (FANet)
- 减少99.5% FLOPs同时提升约10%准确率

**用途**: 大规模候选池的高效逐视频质量评分

### 6.3 时序一致性指标

#### 时序扭曲误差 (TWE)

- 使用RAFT计算连续帧光流
- 将帧t扭曲到帧t+1，测量像素差异
- **用途**: 在合成产品区域检测闪烁、身份漂移或不自然运动

#### 世界一致性分数 (WCS) ([2508.00144](https://arxiv.org/abs/2508.00144))

- 统一指标整合四个子组件: 对象持久性、关系稳定性、因果合规性、闪烁惩罚
- 比单独TWE更全面，捕获产品交互的物理合理性

#### VBench运动平滑度

- 评估对象运动轨迹是否遵循物理合理路径

### 6.4 VLM语义评估 (Qwen2-VL)

| 字段 | 信息 |
|------|------|
| **论文** | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution |
| **arXiv** | [2409.12191](https://arxiv.org/abs/2409.12191) |
| **规模** | 2B, 8B, 72B参数 |

- **原始动态分辨率**: 不同分辨率图像产生不同数量的视觉token
- **M-RoPE**: 将旋转嵌入分解为时间、高度和宽度分量
- 72B模型性能可与GPT-4o和Claude 3.5 Sonnet媲美

**推荐PVTT评分维度**:
1. **产品身份保持**: "插入的产品与参考图像是否一致？"
2. **场景自然度**: "产品放置是否自然？"
3. **物理合理性**: "阴影、反射和光照是否一致？"
4. **合成质量**: "是否有可见伪影、混合问题或不自然边缘？"
5. **总体质量**: "1-5分评估视频编辑质量"

### 6.5 综合评估框架

**VBench** (CVPR 2024 Highlight) - [GitHub](https://github.com/Vchitect/VBench):
- 16个解耦维度: 主体身份、运动平滑度、时序闪烁、空间关系等
- VBench-2.0: 18个维度，包含常识推理和物理

**EvalCrafter** (CVPR 2024, [2310.11440](https://arxiv.org/abs/2310.11440)):
- 700个多样提示, 17个客观指标
- 评估: 视觉质量、内容质量、运动质量、文本-视频对齐

---

## 7. 电商视频数据集与合成训练数据

### 7.1 VACE (Wan2.1/2.2原生编辑框架)

| 字段 | 信息 |
|------|------|
| **论文** | VACE: All-in-One Video Creation and Editing |
| **arXiv** | [2503.07598](https://arxiv.org/abs/2503.07598) |
| **会议** | ICCV 2025 |
| **代码** | [github.com/ali-vilab/VACE](https://github.com/ali-vilab/VACE) |

- **视频条件单元 (VCU)**: V = [T; F; M] (文本 + 上下文帧序列 + 时空掩码)
- **上下文适配器**: Res-Tuning方法 + 分布式Transformer块
- **概念解耦**: 活跃帧 F_c = F × M, 非活跃帧 F_k = F × (1-M)

**与PVTT的关联**: 这**就是**目标模型的编辑框架。理解VACE的VCU设计直接影响训练数据的结构化方式。

### 7.2 大规模合成视频编辑数据集

#### InsViE-1M (ICCV 2025)

| 字段 | 信息 |
|------|------|
| **论文** | InsViE-1M: Effective Instruction-based Video Editing with Elaborate Dataset Construction |
| **arXiv** | [2503.20287](https://arxiv.org/abs/2503.20287) |

- **两阶段编辑-过滤管线**:
  - 阶段1: 首帧编辑 + GPT-4o筛选 (6个候选, CFG ∈ [3,8])
  - 阶段2: SVD视频传播 + GPT-4o评估 + 光流EPE过滤
- **规模**: ~1.02M三元组, 平均GPT评分3.74
- 多阶段训练: 全量 → 高质量子集 → 混合

#### OpenVE-3M (2025年12月)

- **3M样本**, 8个子类别 (空间对齐6 + 非空间对齐2)
- 分辨率720P, 65-129帧/视频
- "局部添加"和"局部修改"类别与PVTT直接相关

#### VIVID-10M (2024年11月)

- **9.7M样本**, 视频局部编辑 (添加、删除、修改)
- 管线: RAM → Grounding DINO → SAM2 → VLM标注
- **掩码增强**: 扩展、凸包、边界框操作

### 7.3 视频目标插入/移除数据集

#### ROSE / ROSE++

| 字段 | 信息 |
|------|------|
| **论文** | ROSE: Remove Objects with Side Effects in Videos |
| **arXiv** | [2508.18633](https://arxiv.org/abs/2508.18633) |
| **数据集** | [HuggingFace: Kunbyte/ROSE-Dataset](https://huggingface.co/datasets/Kunbyte/ROSE-Dataset) |

- **16,678个合成视频对**, Unreal Engine渲染
- 28个高质量环境, 450个独特场景
- 5种物理效果: 阴影、反射、光照、半透明、镜面
- **ROSE++ (插入扩展)**: 将移除数据转换为插入三元组

#### VideoComp (GenCompositor)

- 61K视频集, 标注通过CogVLM + QWen + Grounded SAM2

#### VPData (VideoPainter)

- 390K+片段 (>866.7小时), 最大视频修复数据集

### 7.4 参考数据管线：Stable Video Diffusion

| 字段 | 信息 |
|------|------|
| **论文** | Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets |
| **arXiv** | [2311.15127](https://arxiv.org/abs/2311.15127) |

**黄金标准视频数据策展管线**:

1. **切割检测**: 场景分割为单镜头片段
2. **描述**: CoCa (中间帧) + V-BLIP (视频描述) + LLM融合
3. **标注**: 密集光流、OCR、CLIP嵌入
4. **过滤**: 光流分数 + OCR + CLIP美学分数 + CLIP文本-图像相似度
5. **规模**: 580M原始片段 → **152M策展片段** (74%拒绝率)

**与PVTT的关联**: 过滤阶段应包含：美学评分、运动分析、文本检测、多帧CLIP评分。74%拒绝率是合理基线预期。

---

## 8. 管线配置建议与总结

### 8.1 推荐PVTT管线

```
[预处理]
PySceneDetect (AdaptiveDetector)
  → TransNetV2验证
  → 6-10秒片段

[产品分割]
Grounding DINO 1.5 Pro (文本提示: 产品类别)
  → SAM2 Hiera-L (掩码生成+传播)

[背景恢复]
主力: VideoPainter (即插即用Wan2.1 DiT)
备选: ProPainter (短片段/确定性结果)

[视频合成]
主力: VideoAnyDoor (运动控制, PSNR 38.0)
辅助: InsertAnywhere (遮挡场景, Wan2.1-VACE)

[质量过滤 - 三级级联]
L1: DINO-I > 0.5, MUSIQ, OCR水印检测
L2: TWE时序一致性, DOVER, CLIP-I
L3: Qwen2-VL语义评分 (5个维度)

[目标规模]
500K-1M候选 → 70-75%拒绝率 → 100K-300K高质量训练三元组
```

### 8.2 方法-管线映射

#### 预处理步骤

| 子步骤 | 主要方法 | 备选 | 关键参考 |
|--------|---------|------|----------|
| 镜头检测 | PySceneDetect | TransNetV2, AutoShot | VPData使用PySceneDetect |
| 产品检测 | Grounding DINO 1.5 Pro | Florence-2, RAM | 54.3 AP COCO |
| 掩码生成 | SAM2 (Hiera-B+/L) | Cutie, XMem | 43.8 FPS |
| 背景修复 | VideoPainter | ProPainter, DiffuEraser | 23.32 PSNR |

#### 视频合成

| 方法 | 架构 | 骨干 | 适用场景 |
|------|------|------|----------|
| VideoAnyDoor | 3D U-Net + DINOv2 + Pixel Warper | SDXL | 运动控制, PSNR 38.0 |
| InsertAnywhere | 4D几何 + LoRA | Wan2.1-VACE-14B | 遮挡场景, CLIP-I 0.812 |
| GenCompositor | DiT融合 + ERoPE | CogVideoX-5B | DiT原生, PSNR 42.0 |

#### 质量过滤

| 指标 | 工具 | 度量 |
|------|------|------|
| CLIP-I | OpenCLIP | 语义身份 |
| DINO-I | DINOv2 | 细粒度身份 |
| MUSIQ | MUSIQ Transformer | 逐帧质量 |
| DOVER | DOVER双分支 | 美学+技术 |
| TWE | RAFT光流 | 时序一致性 |
| Qwen2-VL | Qwen2-VL-72B | 语义评估 |

### 8.3 关键发现

1. **InsertAnywhere直接使用Wan2.1-VACE-14B** — 与PVTT目标架构完全一致
2. **DINO-I优于CLIP-I** — 细粒度身份保持评估DINO-I准确率(64%)是CLIP(28%)的2.25倍
3. **VideoPainter的VPData管线** — RAM + Grounding DINO + SAM2 + PySceneDetect几乎就是PVTT所需的完整工具链
4. **质量过滤基线** — 参考SVD的74%拒绝率

---

## 9. 参考文献

### 视频目标插入/合成
1. VideoAnyDoor — [arXiv:2501.01427](https://arxiv.org/abs/2501.01427) — SIGGRAPH 2025
2. InsertAnywhere — [arXiv:2512.17504](https://arxiv.org/abs/2512.17504) — KAIST 2025
3. Insert Anything — [arXiv:2504.15009](https://arxiv.org/abs/2504.15009) — 2025
4. AnyDoor — [arXiv:2307.09481](https://arxiv.org/abs/2307.09481) — ECCV 2024
5. ObjectDrop — [arXiv:2403.18818](https://arxiv.org/abs/2403.18818) — ECCV 2024
6. ObjectMate — [arXiv:2412.08645](https://arxiv.org/abs/2412.08645) — 2024
7. Paint by Inpaint — [arXiv:2404.18212](https://arxiv.org/abs/2404.18212) — 2024
8. GenCompositor — [arXiv:2509.02460](https://arxiv.org/abs/2509.02460) — ICLR 2026
9. Over++ — [arXiv:2512.19661](https://arxiv.org/abs/2512.19661) — 2025
10. Point2Insert — [arXiv:2602.04167](https://arxiv.org/abs/2602.04167) — 2026

### 视频分割
11. SAM 2 — [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) — Meta 2024
12. Grounded SAM — [arXiv:2401.14159](https://arxiv.org/abs/2401.14159) — IDEA Research
13. Grounding DINO 1.5 — [arXiv:2405.10300](https://arxiv.org/abs/2405.10300) — IDEA Research

### 视频修复
14. VideoPainter — [arXiv:2503.05639](https://arxiv.org/abs/2503.05639) — SIGGRAPH 2025
15. ProPainter — [arXiv:2309.03897](https://arxiv.org/abs/2309.03897) — ICCV 2023
16. DiffuEraser — [arXiv:2501.10018](https://arxiv.org/abs/2501.10018) — 2025
17. EraserDiT — [arXiv:2506.12853](https://arxiv.org/abs/2506.12853) — 2025

### 镜头检测
18. TransNetV2 — [arXiv:2008.04838](https://arxiv.org/abs/2008.04838) — 2020
19. AutoShot — [arXiv:2304.06116](https://arxiv.org/abs/2304.06116) — CVPR NAS 2023

### 质量评估
20. MUSIQ — [arXiv:2108.05997](https://arxiv.org/abs/2108.05997) — ICCV 2021
21. DOVER — [arXiv:2211.04894](https://arxiv.org/abs/2211.04894) — ICCV 2023
22. FAST-VQA — [arXiv:2207.02595](https://arxiv.org/abs/2207.02595) — ECCV 2022
23. VBench — [GitHub](https://github.com/Vchitect/VBench) — CVPR 2024
24. EvalCrafter — [arXiv:2310.11440](https://arxiv.org/abs/2310.11440) — CVPR 2024
25. World Consistency Score — [arXiv:2508.00144](https://arxiv.org/abs/2508.00144) — 2025
26. Qwen2-VL — [arXiv:2409.12191](https://arxiv.org/abs/2409.12191) — 2024

### 数据集与数据策展
27. InsViE-1M — [arXiv:2503.20287](https://arxiv.org/abs/2503.20287) — ICCV 2025
28. OpenVE-3M — [arXiv:2512.07826](https://arxiv.org/abs/2512.07826) — 2025
29. VIVID-10M — [arXiv:2411.15260](https://arxiv.org/abs/2411.15260) — 2024
30. ROSE — [arXiv:2508.18633](https://arxiv.org/abs/2508.18633) — 2025
31. Stable Video Diffusion — [arXiv:2311.15127](https://arxiv.org/abs/2311.15127) — 2023

### 目标模型架构
32. Wan2.1 — [arXiv:2503.20314](https://arxiv.org/abs/2503.20314) — 2025
33. VACE — [arXiv:2503.07598](https://arxiv.org/abs/2503.07598) — ICCV 2025

---

**文档版本**: v1.0
**最后更新**: 2026-02-11
**编写**: Claude Code (基于2024-2026最新研究文献)
