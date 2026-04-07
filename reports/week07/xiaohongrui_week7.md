# Week 7 周报

---

## 基本信息

- **姓名：** XIAO Hongrui
- **日期：** 2026-04-07

---

## 1. 研究领域

LoRA 微调 —— 基于视频生成大模型的产品宣传视频定制化生成。

## 2. 领域核心问题

用户提供产品图片和一段宣传模板视频（视频中展示的不是该产品），模型需要生成一段类似风格的宣传视频，其中宣传的产品替换为用户提供的产品。核心挑战在于：
1. **产品主体一致性**：生成视频中的产品外观需与用户提供的产品图高度一致；
2. **视频风格保持**：生成视频的背景、动作、镜头运动等需与模板视频风格相似。

## 3. 技术方案

以 **Wan2.2 TI2V 5B** 为底座模型，采用 **FFGO（First Frame Is the Place to Go for Video Content Customization）** 论文的微调方法，通过 LoRA 微调教会模型"首帧概念记忆 + 场景过渡"的能力：
- 训练时，将产品图与背景拼合为"拼贴式首帧"（I_mix），配合触发词，让模型学会从拼贴首帧自然过渡到连贯的宣传视频；
- 推理时，用户提供产品图和背景图，模型自动完成目标替换和视频生成。

## 4. 本周工作

- [x] 调研并制定 Wan2.2 TI2V + FFGO LoRA 微调的技术选型和超参数设定（`ffgo_lora_finetuning_config_plan.md`）
- [x] 搭建 Wan2.2 TI2V + FFGO LoRA 微调的训练环境和训练代码（`ffgo_lora_training_guide.md`）

## 5. 结论与发现

### 5.1 技术选型及理由

| 决策项 | 选型 | 理由 |
|--------|------|------|
| **底座模型** | Wan2.2 TI2V 5B（5B Dense） | 单 GPU 可训练；相比 I2V-A14B（27B MoE 双 Transformer）显存需求大幅降低，训练时间从 ~10h 缩短至 ~1.5-3.5h |
| **训练框架** | DiffSynth-Studio | 已有现成的 TI2V-5B LoRA 训练脚本，适配成本最低；数据格式（CSV metadata + 视频文件）简单易用 |
| **微调方法** | FFGO LoRA | 少量数据（50-150 样本）即可教会模型首帧概念记忆 + 场景过渡能力 |
| **首帧条件注入** | Noise Blending（区别于 FFGO 原版的 Channel Concat） | TI2V-5B 的 I2V 机制为 noise blending，首帧 latent 直接嵌入初始噪声，模型隐式保持首帧信息，可能更有利于 FFGO 的过渡学习 |

### 5.2 训练数据集规模与属性

| 属性 | 规格 |
|------|------|
| 样本数量 | 50-150 个视频样本（FFGO 原版 50 个，可按需扩展至 150） |
| 视频帧数 | 81 帧 |
| 训练分辨率 | 832 × 480 |
| FPS | 16 |
| 每样本组成 | 拼贴首帧 I_mix（PNG）+ 触发词 caption（TXT）+ 原始视频 V_mix（MP4） |
| 首帧布局 | 左侧：前景产品 RGBA 竖排；右侧：干净背景居中 |
| 触发词格式 | `"ad23r2 the camera view suddenly changes"` + 描述性 caption |

### 5.3 LoRA 微调超参数设定

| 参数 | 值 | 备注 |
|------|------|------|
| LoRA Rank | 128 | 与 FFGO 原版一致，保持足够表达能力 |
| LoRA Alpha | 128 | scale = alpha/rank = 1.0 |
| LoRA Target Modules | q, k, v, o, ffn.0, ffn.2 | 覆盖 Self-Attn、Cross-Attn、FFN |
| 总 LoRA 模块数 | 301（30 layers × 10 modules + 1 time_proj） | |
| 总可训练参数 | ~327M（占模型 5B 参数的 ~6.5%） | |
| 优化器 | AdamW | betas=(0.9, 0.999), eps=1e-10 |
| 学习率 | 1e-4（constant scheduler） | 与 FFGO 原版一致 |
| 权重衰减 | 3e-2 | |
| 最大梯度范数 | 0.05 | |
| 有效 Batch Size | 4 | 单 GPU × batch 1-2 × 梯度累积 2-4 |
| 训练步数 | ~600 步（通过 epochs 控制，150 样本 × 50 epochs） | |
| Checkpoint 保存间隔 | 每 100 步 | |
| 混合精度 | bf16 | 搭配 gradient checkpointing |

### 5.4 输入输出配置

| 项目 | 配置 |
|------|------|
| **训练输入** | I_mix 拼贴首帧（832×480）+ 触发词 caption + 81 帧目标视频 |
| **首帧条件注入** | Noise Blending：`initial_latent = (1-mask) * image_latent + mask * noise` |
| **推理输入** | 产品 RGBA 图 + 背景图 + prompt（自动添加触发词前缀） |
| **推理输出** | 生成 81 帧 → 丢弃前 4 帧（Fc=4，VAE temporal 压缩比）→ 保留 77 帧有效视频 |
| **推理步数** | 50 步（TI2V-5B 默认） |
| **Guidance Scale** | 5.0 |

### 5.5 Loss 函数配置

| 项目 | 配置 |
|------|------|
| 训练范式 | Flow Matching（rectified flows） |
| Scheduler | FlowMatchEulerDiscreteScheduler |
| Loss 函数 | MSE（velocity prediction，即模型预测的 velocity field 与真实 velocity field 的均方误差） |
| Timestep 采样 | Uniform，全范围 [0, 1000]（非 MoE，无需分段） |
| Flow Shift | 5.0 |

### 5.6 资源预估

| GPU | 显存 | 832×480 预估训练时间 | 预估显存占用 |
|-----|------|---------------------|-------------|
| H100/H800 (80GB) | 80GB | ~1.5-2 小时 | ~35-45 GB |
| RTX 5090 (32GB) | 32GB | ~2.5-3.5 小时 | ~24-30 GB |
| RTX 4090 (24GB) | 24GB | ~3-5 小时（需优化） | ~24 GB（紧张） |

## 6. 下周计划

- [ ] 执行 Wan2.2 TI2V + FFGO LoRA 微调（如果有训练数据集的话）
- [ ] 搭建在 PVTT benchmark 上的评估环境和评估代码（包括 VACE、Wan 2.2 TI2V 5B），作为 Wan2.2 TI2V + FFGO LoRA 微调效果的对比 baseline
- [ ] 调研 ViFeEdit 论文工作（arXiv:2603.15478）—— 只靠图像就能训练视频编辑模型，训练成本低

---

## 附录

- `ffgo_lora_finetuning_config_plan.md`：Wan2.2 TI2V 5B + FFGO LoRA 微调的完整技术选型与超参数设定文档
- `ffgo_lora_training_guide.md`：Wan2.2 TI2V 5B + FFGO LoRA 微调的训练环境搭建与代码说明文档
- FFGO 论文：arXiv:2511.15700v1
- ViFeEdit 论文：arXiv:2603.15478
