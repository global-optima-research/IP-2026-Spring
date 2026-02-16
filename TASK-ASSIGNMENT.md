# PVTT 项目详细任务分工

> **Product Video Template Transfer** | IP-2026-Spring
> 10 人团队，3-3-4 模块化分配 | 预计周期 16 周

---

## 总览

| 模块 | 任务 | 成员 | 人数 |
|------|------|------|------|
| **Task 1** | 数据集构建 | 王洁怡、刘麓琰、王志铭 | 3 |
| **Task 2** | Teacher Model 训练 | 钟蕊、肖泓锐、方子维 | 3 |
| **Task 3** | DMD 蒸馏加速 | 李一佳、陈庆展、Sze Long、邱张西子 | 4 |

---

## Task 1：数据集构建（3 人）

### 目标
从电商平台采集原始视频素材，经过四阶段处理，产出 **20K+ 高质量训练三元组**（source_video, reference_image, edited_video, mask, caption）。

---

### 王洁怡 — 数据采集 & 预处理 Pipeline

**核心职责：** 负责原始数据采集和视频预处理的全流程搭建。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| 电商视频采集 | 从 Etsy/Amazon/淘宝等平台爬取产品展示视频，覆盖多品类（手表、珠宝、箱包、化妆品等） | Scrapy / Selenium | 500+ 原始产品视频 |
| 镜头检测与切割 | 将长视频切分为 1.5-5s 的单镜头片段，去除转场和无效帧 | PySceneDetect (ContentDetector) | 每视频 3-8 个有效片段 |
| 视频标准化 | 统一分辨率（720p）、帧率（24fps）、时长（2-4s），格式标准化 | FFmpeg / OpenCV | 标准化视频库 |
| 产品图片采集 | 采集对应产品的多角度白底图/RGBA 透明底图 | rembg / BiRefNet | 53+ 产品 × 多视角图片 |
| 数据管理 | 建立数据索引和元信息管理系统（产品ID、品类、视角、分辨率等） | JSON/SQLite | 完整数据索引 |

**关键里程碑：**
- Week 1-2：完成 50+ 产品视频采集和镜头切割
- Week 3-4：完成视频标准化和产品图片采集
- Week 5：完成数据索引系统

---

### 刘麓琰 — 商品分割 & 背景修复

**核心职责：** 负责视频中商品区域的精确分割和背景视频的修复重建。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| 商品检测 | 基于 Grounded-SAM2 实现视频首帧中商品的自动定位检测 | Grounded-SAM2 | 首帧 BBox + 初始 Mask |
| 视频级商品分割 | 利用 SAM2 视频传播能力，将首帧 Mask 传播至整段视频所有帧 | SAM2 (video predictor) | 每帧精确 Mask 序列 |
| 分割质量检查 | 检查 Mask 时序一致性（IoU > 0.9）、边缘质量、漏检/误检 | 自定义脚本 | 过滤后的高质量 Mask |
| 背景修复 | 使用 VideoPainter 对 Mask 区域进行视频级 inpainting，获取纯背景视频 | VideoPainter | 纯背景视频 |
| 交叉配对 | 53 模板视频 × 52 产品图片 × 多视角进行交叉配对，合理性过滤 | 自定义脚本 | ~35K 候选训练对 |

**关键里程碑：**
- Week 2-3：SAM2 分割 pipeline 搭建完成
- Week 4-5：VideoPainter 背景修复完成
- Week 5-6：交叉配对完成，产出 35K 候选对

---

### 王志铭 — 视频合成 & 质量过滤

**核心职责：** 负责 Ground Truth 视频的合成生成和多级质量过滤系统。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| 路线 A 合成（主力） | 使用 VideoAnyDoor / InsertAnywhere，基于背景视频 + 商品 RGBA + 运动轨迹合成 GT | VideoAnyDoor / InsertAnywhere | ~25K 合成视频 |
| 路线 B 合成（补充） | FLUX 编辑首帧 + Wan2.1 I2V 视频传播，作为补充合成方案 | FLUX + Wan2.1 | 补充合成数据 |
| L1 硬性过滤 | 完整性检查、商品存在性验证、BG SSIM > 0.85 | OpenCV / scikit-image | 去除明显缺陷样本 |
| L2 软性评分 | CLIP-I（身份保持）、DINO-I、时序一致性、MUSIQ（美学质量）综合评分 | CLIP / DINO / MUSIQ | 多维度质量分数 |
| L3 VLM + 人工抽检 | Qwen2-VL 多维语义评分 + 10% 随机人工抽检 | Qwen2-VL | 最终 20K+ 高质量训练对 |
| 数据格式封装 | 统一封装为训练格式（source_video, target_image, edited_video, mask, caption） | 自定义脚本 | 训练就绪数据集 |

**关键里程碑：**
- Week 4-6：VideoAnyDoor 合成 pipeline 完成
- Week 7-8：三级质量过滤系统完成
- Week 8-9：产出 20K+ 高质量训练对

---

## Task 2：Teacher Model 训练（3 人）

### 目标
改造 Wan2.1 / 2.2 架构使其接收 source video + reference image 双条件输入，通过 LoRA 微调训练 PVTT 专用 Teacher Model，达到 CLIP-I > 0.85、FVD < 100。

---

### 钟蕊 — 架构改造 & 条件注入

**核心职责：** 负责 Wan2.1/2.2 模型架构改造，实现 Reference Image 条件注入通道。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| Wan2.1 源码分析 | 深入分析 Wan2.1/2.2 DiT 架构，理清 Spatial/Temporal Self-Attention、Cross-Attention、FFN 各层结构 | PyTorch | 架构分析文档 |
| Ref Image 编码器 | 设计 Reference Image 编码模块（CLIP Image Encoder / DINOv2），将商品图编码为条件向量 | CLIP / DINOv2 | Image Encoder 模块 |
| Cross-Attention 注入 | 修改 Cross-Attention 层，使其同时接收 text embedding 和 reference image embedding | PyTorch | 改造后的 DiT 模块 |
| Source Video 条件化 | 实现 source video + mask 作为额外条件输入（channel concatenation 或 AdaIN） | PyTorch | 多条件输入接口 |
| 架构验证 | 在小规模数据上验证改造后模型能否正常前向传播和反向传播 | PyTorch | 验证报告 |

**关键里程碑：**
- Week 1-3：Wan2.1 源码分析完成
- Week 3-5：架构改造实现
- Week 5-6：架构验证通过

---

### 肖泓锐 — LoRA 微调 & 渐进训练

**核心职责：** 负责 LoRA 微调策略设计和渐进式训练全流程。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| LoRA 配置设计 | 确定 LoRA 注入位置（Temporal Self-Attn + Cross-Attn）、rank（64/256）、alpha 等超参 | PEFT | LoRA 配置方案 |
| Phase 1 训练 | 1.3B 模型 + LoRA (rank 64)，480p / 16帧 / 5K 数据，快速验证 pipeline 可行性 | DiffSynth-Studio | Phase 1 checkpoint |
| Phase 2 训练 | 14B 模型 + LoRA (rank 256)，720p / 24帧 / 20K 数据，规模化训练 | DeepSpeed ZeRO-2/3 | Phase 2 checkpoint |
| Phase 3 训练 | 14B 模型，解冻更多层进行 Partial / Full Fine-tuning，精细调优 | DeepSpeed ZeRO-3 | 最终 Teacher Model |
| 训练日志监控 | 使用 Wandb 监控 loss 曲线、梯度、学习率、样本质量等 | Wandb | 训练监控 Dashboard |
| 超参搜索 | 学习率（1e-4 ~ 5e-5）、batch size、gradient accumulation、warmup 等搜索 | Optuna / 手动 | 最优超参配置 |

**关键里程碑：**
- Week 5-7：Phase 1 训练完成，验证 pipeline
- Week 7-10：Phase 2 规模化训练完成
- Week 10-12：Phase 3 最终模型训练完成

---

### 方子维 — 评测体系 & 训练框架

**核心职责：** 负责搭建自动化评测体系和训练工程基础设施。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| 评测指标实现 | 实现 CLIP-I、FVD、时序一致性、BG PSNR、LPIPS 等评测指标 | torchmetrics / 自定义 | 评测工具库 |
| 自动化评测 Pipeline | 每隔 N 步自动生成样本并计算指标，推送到 Wandb | Python + Wandb | 自动评测系统 |
| 训练框架搭建 | 基于 DiffSynth-Studio 搭建训练框架，集成 DeepSpeed、梯度累积、混合精度 | DiffSynth-Studio | 训练框架 |
| 数据加载优化 | 视频数据的高效加载（多进程 DataLoader、预解码缓存、数据增强） | PyTorch DataLoader | 高效数据管线 |
| Benchmark 构建 | 构建固定测试集（50-100 对），用于不同版本模型的横向对比 | 自定义 | Benchmark 测试集 |

**关键里程碑：**
- Week 3-5：评测指标和自动化 pipeline 完成
- Week 4-6：训练框架搭建完成
- Week 6-8：Benchmark 构建完成

---

## Task 3：DMD 蒸馏加速（4 人）

### 目标
通过 Distribution Matching Distillation 将 50 步 Teacher Model 渐进压缩至 **4 步 Student Model**，实现 4-8× 加速，质量损失 < 5%。

> **策略：先复现，再改进。** Task 3 全组在前期需先基于已有开源 codebase 跑通蒸馏训练全流程，积累实操经验后再进入各自分工的深入研发。

---

### Phase 0：现有方案复现（全组 4 人共同参与，Week 1-6）

在进入各自分工前，全组 4 人需要 **先跑通现有蒸馏方案的训练和推理**，建立对蒸馏 pipeline 的完整理解。

#### 核心 Codebase

| # | 项目 | GitHub | 特点 | 优先级 |
|---|------|--------|------|--------|
| 1 | **FastVideo** | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | DMD + Sparse Attention 联合训练；**原生支持 Wan2.1 (1.3B/14B) + Wan2.2 (5B)**；已发布 FastWan2.1-1.3B、FastWan2.2-5B 蒸馏模型 | ⭐ 最高 |
| 2 | **NVIDIA FastGen** | [NVlabs/FastGen](https://github.com/NVlabs/FastGen) | 统一蒸馏框架，支持 DMD2/ECT/MeanFlow/CausVid 等 10+ 种方法；**原生支持 Wan2.1 14B**；64×H100 上 16h 收敛 | ⭐ 最高 |
| 3 | **distill_wan2.1** | [azuresky03/distill_wan2.1](https://github.com/azuresky03/distill_wan2.1) | 专门针对 Wan2.1 的蒸馏，支持 Consistency Distillation + DMD2 + RL；代码清晰，适合学习 | ⭐ 高 |
| 4 | **LightX2V + Wan2.2-Lightning** | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) / [ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning) | 轻量推理框架 + 步数蒸馏；4 步无 CFG 推理，~20× 加速 | ⭐ 高 |
| 5 | **CausVid** | [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | CVPR 2025，DMD 扩展到视频的自回归流式生成；VBench 第一名 (84.27) | 参考 |
| 6 | **TurboDiffusion** | [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) | rCM + SLA + W8A8 组合加速，100-200× 加速 | 参考 |

#### 复现分工

| 成员 | 复现任务 | 具体目标 |
|------|---------|---------|
| **李一佳** | FastVideo — DMD 蒸馏 on Wan2.1 | 使用 FastVideo 框架对 Wan2.1-1.3B 进行 DMD 蒸馏训练，跑通 T2V 任务的完整蒸馏流程（50→4 步），记录训练配置、Loss 曲线、显存占用 |
| **陈庆展** | NVIDIA FastGen — DMD2 on Wan2.1 | 使用 FastGen 框架复现 DMD2 方法在 Wan2.1 上的蒸馏，对比不同蒸馏方法（DMD2 vs ECT vs Consistency），记录收敛速度和生成质量 |
| **Sze Long** | distill_wan2.1 + LightX2V | 复现 distill_wan2.1 的 DMD2 + RL 训练流程；同时搭建 LightX2V 推理环境，测试 Wan2.2-Lightning 蒸馏模型的推理性能 |
| **邱张西子** | FastGen — MeanFlow / rCM on Wan2.1 | 使用 FastGen 框架复现 MeanFlow 和 rCM 方法在 Wan2.1 上的训练，作为 Mean Flow 探索的起点 |

#### 复现产出要求

每人需提交一份复现报告，包含：

1. **环境配置**：硬件（GPU 型号/数量）、软件依赖、安装步骤
2. **训练配置**：超参数、batch size、学习率、训练步数、数据集
3. **训练日志**：Loss 曲线、训练时长、显存峰值
4. **生成质量**：蒸馏前后的生成样本对比（FVD / FID / CLIP-Score）
5. **踩坑记录**：遇到的问题和解决方案
6. **框架评估**：该 codebase 的代码质量、可扩展性、与我们 PVTT 项目的适配难度评估

**关键里程碑：**
- Week 1-2：环境搭建，阅读代码，跑通推理
- Week 3-4：跑通蒸馏训练流程（可用小数据集 / 小模型验证）
- Week 5-6：完成复现报告，全组讨论选定最优 codebase 作为后续开发基础

---

### Phase 1 及后续：基于复现经验的深入研发（Week 7-16）

> 在 Phase 0 跑通复现后，全组根据评估结果选定 1-2 个最优 codebase 作为开发基础，然后进入各自分工。

---

### 李一佳 — DMD 蒸馏核心 & Loss 设计

**核心职责：** 基于复现选定的 codebase（预计 FastVideo / FastGen），负责针对 PVTT 视频编辑任务的 DMD 蒸馏适配和 Loss 函数扩展。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| Codebase 适配 | 将选定蒸馏框架适配到 PVTT Teacher Model（修改条件输入：source video + ref image + mask） | FastVideo / FastGen | PVTT 蒸馏训练代码 |
| Distribution Loss | 基于现有 DMD 实现，调优对抗训练策略（GAN loss / f-divergence），使 Student 匹配 Teacher 分布 | PyTorch | Distribution Loss |
| Regression Loss | 在已有 MSE 基础上增加 LPIPS 感知损失，确保 Student 在像素和感知层面接近 Teacher | LPIPS / PyTorch | Regression Loss |
| Temporal Loss | 实现帧间一致性损失（光流 warp loss / 相邻帧 feature 距离），解决视频蒸馏特有的时序闪烁问题 | RAFT / PyTorch | Temporal Loss |
| BG + Identity Loss | 实现 Mask 外区域的背景保持损失 + CLIP-based 商品外观保真度损失 | CLIP / PyTorch | 视频编辑专用 Loss |
| Loss 权重调优 | 各 Loss 权重的搜索和平衡（λ_dist, λ_reg, λ_temp, λ_bg, λ_id） | Wandb | 最优 Loss 配置 |

**关键里程碑：**
- Week 7-9：Codebase 适配 PVTT 完成
- Week 9-11：全部 Loss 实现和调优
- Week 11-12：首轮蒸馏（50→16步）完成

---

### 陈庆展 — 渐进式蒸馏 & Discriminator

**核心职责：** 基于复现经验，负责渐进式蒸馏策略实施和 3D Video Discriminator 训练。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| 3D Video Discriminator | 基于 FastVideo/FastGen 已有 Discriminator 设计，适配视频编辑场景（需关注商品区域 vs 背景区域） | PyTorch | Discriminator 模型 |
| 蒸馏阶段 1 | 50 步 → 16 步蒸馏，~20K 训练步 | FastVideo + DeepSpeed | 16 步 Student |
| 蒸馏阶段 2 | 16 步 → 8 步蒸馏，~20K 训练步 | FastVideo + DeepSpeed | 8 步 Student |
| 蒸馏阶段 3 | 8 步 → 4 步蒸馏，~20K 训练步 | FastVideo + DeepSpeed | 4 步 Student |
| 质量门槛控制 | 每阶段设定质量门槛（CLIP-I / FVD），不达标则延长训练或调整策略 | 评测体系 | 质量达标报告 |
| EMA 策略 | Student 模型的 EMA (Exponential Moving Average) 更新策略 | PyTorch | EMA 配置 |

**关键里程碑：**
- Week 8-10：Discriminator 训练完成
- Week 10-12：50→16→8 步蒸馏完成
- Week 12-14：8→4 步蒸馏完成

---

### Sze Long — 工程加速 & 量化部署

**核心职责：** 基于 LightX2V/FastVideo 复现经验，负责推理加速的工程优化和最终模型的量化部署。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| Flash Attention 2 | 集成 Flash Attention 2，优化注意力计算（2-4× 加速） | flash-attn | FA2 集成 |
| SageAttention | 集成 SageAttention（FastVideo 采用的低精度注意力方案），额外加速 30-50% | SageAttention | SA 集成 |
| INT8 / FP8 量化 | 模型权重和激活值的量化（显存降低 ~40%，速度提升 ~30%） | bitsandbytes / quanto | 量化模型 |
| torch.compile | 使用 torch.compile 进行图编译优化（额外加速 10-20%） | PyTorch 2.x | 编译优化模型 |
| TensorRT 部署 | 导出 TensorRT 引擎，用于生产环境的高效推理 | TensorRT | TRT 推理引擎 |
| 推理 Pipeline | 封装完整的推理 pipeline（输入预处理 → 模型推理 → 后处理） | Python | 推理 API |
| 性能 Benchmark | 不同优化组合下的延迟/吞吐/显存对比（A100/H100/4090 等） | 自定义 | 性能报告 |

**关键里程碑：**
- Week 7-9：基于 LightX2V 完成基础推理优化框架
- Week 10-12：Flash Attention 2 + SageAttention + 量化集成
- Week 12-14：torch.compile + TensorRT 部署
- Week 14-16：完整推理 pipeline 和性能报告

---

### 邱张西子 — Mean Flow 方法在视频生成任务上的探索

**核心职责：** 基于 FastGen 中 MeanFlow/rCM 的复现，深入探索 Flow Matching 类方法在视频生成/编辑中的应用。

| 子任务 | 详细描述 | 工具/框架 | 产出 |
|--------|---------|----------|------|
| 文献调研 | 调研 Flow Matching / Rectified Flow / Mean Flow / rCM 等方法的最新进展 | 论文阅读 | 调研报告 |
| FastGen 复现深化 | 在 Phase 0 基础上，深入分析 FastGen 中 MeanFlow 和 rCM 的实现细节 | [NVlabs/FastGen](https://github.com/NVlabs/FastGen) / [NVlabs/rcm](https://github.com/NVlabs/rcm) | 代码分析报告 |
| 视频编辑适配 | 将 MeanFlow/rCM 适配到 PVTT 视频编辑任务（source video + ref image 条件输入） | PyTorch | 适配代码 |
| 对比实验 | MeanFlow vs DMD2 vs Consistency Distillation 在 PVTT 任务上的对比实验 | 评测体系 | 实验报告 |
| 混合方案探索 | 探索 MeanFlow + DMD 的混合加速方案（先 Flow 直化再 DMD 蒸馏） | PyTorch | 混合方案 |
| CausVid 探索（选做） | 探索 CausVid 的自回归蒸馏方案是否可用于 PVTT 的流式推理 | [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | 可行性报告 |

**关键里程碑：**
- Week 1-6：Phase 0 复现 + 文献调研
- Week 7-10：视频编辑适配和对比实验
- Week 10-14：混合方案探索
- Week 14-16：最终实验报告和方案总结

---

### 参考 Codebase 汇总

| 项目 | GitHub | 方法 | Wan 支持 | 备注 |
|------|--------|------|---------|------|
| FastVideo | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | DMD + Sparse-Distill | Wan2.1 ✅ Wan2.2 ✅ | **首选**，5s 480p 视频在 H200 上 5s 生成 |
| NVIDIA FastGen | [NVlabs/FastGen](https://github.com/NVlabs/FastGen) | DMD2 / ECT / MeanFlow / CausVid 等 | Wan2.1 ✅ Wan2.2 ✅ | **方法最全**，插件式架构 |
| distill_wan2.1 | [azuresky03/distill_wan2.1](https://github.com/azuresky03/distill_wan2.1) | DMD2 + Consistency + RL | Wan2.1 ✅ | 代码清晰，适合学习 |
| LightX2V | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) | Step Distill + CFG Distill | Wan2.1 ✅ Wan2.2 ✅ | 4 步无 CFG，~20× 加速 |
| Wan2.2-Lightning | [ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning) | Step Distillation | Wan2.2 ✅ | 蒸馏模型已发布 |
| CausVid | [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | DMD + 自回归 | 通用 DiT | CVPR 2025，VBench #1 |
| TurboDiffusion | [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) | rCM + SLA + W8A8 | Wan2.1 ✅ | 100-200× 加速 |
| NVIDIA rCM | [NVlabs/rcm](https://github.com/NVlabs/rcm) | Rectified Consistency Model | 通用 | FastGen 子组件 |
| Sparse VideoGen | [svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen) | Sparse Attention（免训练） | Wan2.1 ✅ | 与蒸馏方法互补 |

---

## 时间线总览

```
Week 1-4    ████████░░░░░░░░░░░░░░░░░░░░░░░░  数据采集 & 预处理
Week 3-6    ░░░░████████░░░░░░░░░░░░░░░░░░░░  分割 & 修复 & 配对
Week 4-9    ░░░░░░████████████░░░░░░░░░░░░░░  合成 & 过滤 → 20K+ 训练对
Week 1-6    ████████████░░░░░░░░░░░░░░░░░░░░  架构改造 & 训练框架
Week 5-12   ░░░░░░░░████████████████░░░░░░░░  LoRA 微调 & 渐进训练
Week 8-14   ░░░░░░░░░░░░░░████████████░░░░░░  DMD 蒸馏 (50→16→8→4)
Week 8-16   ░░░░░░░░░░░░░░████████████████░░  Mean Flow 探索
Week 10-16  ░░░░░░░░░░░░░░░░░░████████████░░  工程加速 & 部署
```

---

## 协作依赖关系

```
Task 1 (数据集) ──产出 20K+ 训练对──→ Task 2 (模型训练)
                                        │
                                 产出 Teacher Model
                                        │
                                        ▼
                               Task 3 (DMD 蒸馏加速)
                                        │
                                 产出 4步 Student Model
                                        │
                                        ▼
                               最终 PVTT 系统交付
```

**关键交接节点：**
- **Week 8-9**：Task 1 → Task 2（20K+ 训练对交付）
- **Week 12**：Task 2 → Task 3（Teacher Model 交付）
- **Week 16**：Task 3 产出最终 4 步推理模型
