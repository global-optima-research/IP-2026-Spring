# DMD GitHub 实操代码汇总

> **目标**: 整理 GitHub 上可用的 DMD 训练代码，帮助解决 DMD 训练问题
> **日期**: 2025年2月

---

## 目录

1. [代码仓库总览](#1-代码仓库总览)
2. [DMD2 官方实现（推荐）](#2-dmd2-官方实现推荐)
3. [OpenDMD 开源实现](#3-opendmd-开源实现)
4. [CIFAR-10 DMD 实现](#4-cifar-10-dmd-实现)
5. [DMDR 强化学习版本](#5-dmdr-强化学习版本)
6. [SenseFlow 大规模蒸馏](#6-senseflow-大规模蒸馏)
7. [Phased DMD 实现](#7-phased-dmd-实现)
8. [AMD Nitro-1 实现](#8-amd-nitro-1-实现)
9. [各仓库对比与选择建议](#9-各仓库对比与选择建议)

---

## 1. 代码仓库总览

| 仓库 | 类型 | 支持模型 | 硬件需求 | 训练代码 | 推荐度 |
|------|------|---------|---------|---------|--------|
| [tianweiy/DMD2](https://github.com/tianweiy/DMD2) | 官方 | SDXL, SD1.5, ImageNet | 7-64 GPU | ✅ 完整 | ⭐⭐⭐⭐⭐ |
| [Zeqiang-Lai/OpenDMD](https://github.com/Zeqiang-Lai/OpenDMD) | 社区 | DreamShaper | 单卡可训 | ✅ 有 | ⭐⭐⭐ |
| [devrimcavusoglu/dmd](https://github.com/devrimcavusoglu/dmd) | 社区 | CIFAR-10 | 单卡 | ✅ 完整 | ⭐⭐⭐⭐ |
| [vvvvvjdy/dmdr](https://github.com/vvvvvjdy/dmdr) | 官方 | ImageNet | 多卡 | ✅ Demo | ⭐⭐⭐ |
| [XingtongGe/SenseFlow](https://github.com/XingtongGe/SenseFlow) | 官方 | SDXL, SD3.5, FLUX | 多卡 | ✅ 完整 | ⭐⭐⭐⭐ |
| [ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning) | 官方 | Wan2.2 视频 | - | ❌ 仅推理 | ⭐⭐ |
| [AMD-AIG-AIMA/AMD-Diffusion-Distillation](https://github.com/AMD-AIG-AIMA/AMD-Diffusion-Distillation) | 官方 | SD2.1, PixArt | AMD GPU | ✅ 完整 | ⭐⭐⭐ |

---

## 2. DMD2 官方实现（推荐）

### 仓库信息

- **地址**: [https://github.com/tianweiy/DMD2](https://github.com/tianweiy/DMD2)
- **论文**: NeurIPS 2024 Oral
- **作者**: MIT, Adobe Research
- **Stars**: 1k+

### 支持的模型与硬件需求

| 模型 | 分辨率 | GPU 数量 | 训练时间 | FID |
|------|--------|---------|---------|-----|
| ImageNet | 64×64 | 7 GPU | 38-70 小时 | 1.28 |
| SD v1.5 | 512×512 | 64 GPU (8节点) | ~25 小时 | 8.35 |
| SDXL | 1024×1024 | 64 GPU (8节点) | ~57 小时 | - |

### 安装步骤

```bash
# 创建环境
conda create -n dmd2 python=3.8 -y
conda activate dmd2

# 安装依赖
pip install --upgrade anyio
pip install -r requirements.txt
python setup.py develop
```

### ImageNet 训练（入门推荐）

**硬件要求**: 最少 7 张 GPU，支持 BF16

```bash
# 1. 设置环境变量
export CHECKPOINT_PATH="/path/to/checkpoint"
export WANDB_ENTITY="your_wandb_entity"
export WANDB_PROJECT="your_wandb_project"

# 2. 下载预训练模型和数据
bash scripts/download_imagenet.sh $CHECKPOINT_PATH

# 3. 开始训练（7 GPU）
bash experiments/imagenet/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch.sh \
    $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT

# 4. 同时运行评估（在另一个终端）
python main/edm/test_folder_edm.py \
    --folder $CHECKPOINT_PATH/[training_folder]/[timestamp] \
    --wandb_name test_imagenet \
    --resolution 64 --label_dim 1000 \
    --ref_path $CHECKPOINT_PATH/imagenet_fid_refs_edm.npz \
    --detector_url $CHECKPOINT_PATH/inception-2015-12-05.pkl
```

### SD v1.5 训练

**硬件要求**: 64 GPU（8 节点 × 8 GPU）

```bash
# 1. 设置环境变量
export CHECKPOINT_PATH="/path/to/checkpoint"
export WANDB_ENTITY="your_entity"
export WANDB_PROJECT="your_project"
export MASTER_IP="your_master_node_ip"

# 2. 下载模型和数据
bash scripts/download_sdv15.sh $CHECKPOINT_PATH

# 3. 在所有 8 个节点上运行（NODE_RANK 从 0 到 7）
bash experiments/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch.sh \
    $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT $MASTER_IP NODE_RANK
```

### SDXL 训练

**硬件要求**: 64 GPU（8 节点 × 8 GPU）

```bash
# 1. 创建 FSDP 配置
python main/sdxl/create_sdxl_fsdp_configs.py \
    --folder fsdp_configs/EXP_NAME \
    --master_ip $MASTER_IP \
    --num_machines 8 \
    --sharding_strategy 4

# 2. 下载 SDXL 模型
bash scripts/download_sdxl.sh $CHECKPOINT_PATH

# 3. 4步模型训练
bash experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh \
    $CHECKPOINT_PATH $WANDB_ENTITY $WANDB_PROJECT fsdp_configs/EXP_NAME NODE_RANK
```

### 预训练模型下载

```python
# 使用 HuggingFace 加载
from diffusers import DiffusionPipeline
import torch

# SDXL 4步模型
pipe = DiffusionPipeline.from_pretrained(
    "tianweiy/DMD2",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# 生成图像
image = pipe(
    "a photo of a cat",
    num_inference_steps=4,
    guidance_scale=0
).images[0]
```

### 解决的 DMD 训练问题

| 问题 | DMD2 解决方案 |
|------|--------------|
| 需要预计算数据集 | ✅ 移除回归损失，无需预计算 |
| Fake critic 不稳定 | ✅ 两时间尺度更新规则 |
| 模式崩溃 | ⚠️ 部分缓解（通过 GAN 损失） |
| 仅支持单步 | ✅ 支持多步训练 |

---

## 3. OpenDMD 开源实现

### 仓库信息

- **地址**: [https://github.com/Zeqiang-Lai/OpenDMD](https://github.com/Zeqiang-Lai/OpenDMD)
- **特点**: 社区实现，更易上手
- **支持**: DreamShaper 等自定义模型

### 核心文件

```
OpenDMD/
├── train_dmd.py          # 主训练脚本
├── train.sh              # 训练启动脚本
├── build_regression_data.py  # 数据准备
├── gradio_dmd.py         # Gradio 演示
└── download_caption.py   # 下载 caption 数据
```

### 训练步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Zeqiang-Lai/OpenDMD.git
cd OpenDMD

# 2. 准备回归数据
python build_regression_data.py

# 3. 开始训练
bash train.sh
```

### 预训练模型

| 模型 | 训练步数 | 说明 |
|------|---------|------|
| dreamshaper-8-dmd-1kstep | 1,000 | 实验性，效果一般 |
| dreamshaper-8-dmd-kl-only-6kstep | 6,000 | 仅 KL 损失，效果有限 |

### Gradio 演示

```bash
python gradio_dmd.py
```

### 注意事项

⚠️ 作者声明这是"非常实验性的发布"，训练不充分，效果可能不理想。适合学习和实验，不推荐生产使用。

---

## 4. CIFAR-10 DMD 实现

### 仓库信息

- **地址**: [https://github.com/devrimcavusoglu/dmd](https://github.com/devrimcavusoglu/dmd)
- **特点**: 完整的非官方实现，适合学习
- **数据集**: CIFAR-10 (32×32)

### 安装

```bash
# 使用 conda
conda env create -f environment.yml
conda activate dmd
```

### 训练命令

```bash
# 下载蒸馏数据集
bash scripts/download_distillation_dataset.sh

# 开始训练
python -m dmd train \
    --model-path https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --data-path /path/to/distillation_data \
    --output-dir /path/to/output \
    --epochs 2 \
    --batch-size 32
```

### 生成图像

```python
from dmd.generate import DMDGenerator

# 加载模型
generator = DMDGenerator(network_path="path/to/checkpoint.pt")

# 生成 25 张图像（类别 0）
samples = generator.generate_batch(
    seeds=list(range(25)),
    class_ids=0
)
```

### 特点

- ✅ 完整的训练和推理代码
- ✅ 支持 Neptune 实验追踪
- ✅ 详细的文档和示例
- ⚠️ 仅支持 CIFAR-10
- ⚠️ 非商业许可（CC-BY-NC-SA 4.0）

---

## 5. DMDR 强化学习版本

### 仓库信息

- **地址**: [https://github.com/vvvvvjdy/dmdr](https://github.com/vvvvvjdy/dmdr)
- **论文**: Distribution Matching Distillation Meets Reinforcement Learning
- **特点**: DMD + RL 联合训练，可超越教师

### 核心创新

```
传统 DMD: 学生 ≤ 教师
DMDR:     学生 > 教师（通过 RL 优化）
```

### 训练代码

```bash
# ImageNet SiT 训练
cd train_cc/sit/
# 参考 README 进行训练
```

### 相关模型

- **Z-Image-Turbo**: 8步蒸馏模型，使用 Decoupled-DMD + DMDR
- **SD3.5 Large**: 4 NFE 微调版本

### Decoupled DMD 发现

```
DMD = CFG Augmentation (主要) + Distribution Matching (正则化)

CFG Augmentation: 负责少步转换的核心能力
Distribution Matching: 作为正则化器，防止偏离
```

---

## 6. SenseFlow 大规模蒸馏

### 仓库信息

- **地址**: [https://github.com/XingtongGe/SenseFlow](https://github.com/XingtongGe/SenseFlow)
- **论文**: ICLR 2026
- **特点**: 解决大规模 Flow 模型蒸馏的收敛问题

### 支持的模型

| 模型 | 步数 | HPS v2 | ImageReward |
|------|------|--------|-------------|
| SDXL | 4-8 步 | 0.3010 | 0.9951 |
| SD 3.5 Medium | 4-8 步 | 0.3016 | - |
| SD 3.5 Large | 4-8 步 | - | - |
| FLUX.1 dev | 4-8 步 | - | - |

### 核心技术

**1. Implicit Distribution Alignment (IDA)**
- 正则化生成器和 fake 分布的距离
- 防止模式崩溃

**2. Intra-segment Guidance (ISG)**
- 重新分配时间步的重要性
- 从教师模型迁移知识

### 训练命令

```bash
# SDXL 训练
bash train_sdxl.sh NUM_NODES GPUS_PER_NODE CONFIG_PATH SAVE_DIR

# SD3.5 训练
bash train_sd35.sh NUM_NODES GPUS_PER_NODE CONFIG_PATH SAVE_DIR

# FLUX 训练
bash train_flux.sh NUM_NODES GPUS_PER_NODE CONFIG_PATH SAVE_DIR
```

### 预训练模型

所有模型权重在 HuggingFace 上开源。

---

## 7. Phased DMD 实现

### 仓库信息

- **地址**: [https://github.com/ModelTC/Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning)
- **项目主页**: [https://x-niper.github.io/projects/Phased-DMD/](https://x-niper.github.io/projects/Phased-DMD/)
- **特点**: SNR 分段蒸馏，保持多样性

### 核心思想

```
传统 DMD: 全局优化 → 多样性退化
Phased DMD: 分阶段优化 → 保持多样性

Phase 1 [低 SNR]: 决定结构 → 冻结
Phase 2 [中 SNR]: 中间细节 → 冻结
Phase 3 [高 SNR]: 精细纹理 → 完成
```

### 可用模型

| 模型 | 日期 | 步数 |
|------|------|------|
| Wan2.2-T2V-A14B-NFE4-V2.0 | 2025-11-08 | 4步 |
| Wan2.2-T2V-A14B-NFE4-0928 | 2025-09-28 | 4步 |
| Wan2.2-I2V-A14B-NFE4-V1 | 2025-08-07 | 4步 |

### 性能对比

| 方法 | DINOv3 相似度 ↓ | LPIPS ↑ | 光流 ↑ |
|------|----------------|---------|--------|
| DMD+SGTS | 0.826 | 0.521 | 3.23 |
| **Phased DMD** | **0.782** | **0.544** | **7.57** |

### 注意事项

⚠️ 目前仅发布推理代码和模型权重，训练代码暂未开源。

---

## 8. AMD Nitro-1 实现

### 仓库信息

- **地址**: [https://github.com/AMD-AIG-AIMA/AMD-Diffusion-Distillation](https://github.com/AMD-AIG-AIMA/AMD-Diffusion-Distillation)
- **特点**: LADD 方法实现，针对 AMD GPU 优化

### 支持的模型

| 模型 | 基座 | 分辨率 | FLOPs 降低 |
|------|------|--------|-----------|
| Nitro-1-SD | SD 2.1 | 512px | 95.9% |
| Nitro-1-PixArt | PixArt-Sigma | 1024px | - |

### 训练流程

```bash
# 1. 生成合成训练数据
bash run_gen_data.sh        # SD 2.1
bash run_gen_data_pixart.sh # PixArt

# 2. 训练模型
bash run_train.sh

# 3. 评估
bash run_eval.sh
```

### 数据生成

使用 DiffusionDB prompts 从基座模型生成训练数据：

```python
# 伪代码
for prompt in diffusion_db_prompts:
    image = base_model.generate(prompt, steps=50)
    save_pair(prompt, image)
```

---

## 9. 各仓库对比与选择建议

### 按使用场景选择

```
你的情况是什么？
│
├── 想学习 DMD 原理
│   └── 推荐: devrimcavusoglu/dmd (CIFAR-10)
│       - 代码简洁清晰
│       - 单卡可训练
│       - 完整的文档
│
├── 想训练生产级模型
│   ├── 有 64+ GPU
│   │   └── 推荐: tianweiy/DMD2
│   │       - 官方实现
│   │       - 最佳性能
│   │       - 完整的 SDXL/SD1.5 支持
│   │
│   └── GPU 资源有限
│       └── 推荐: Zeqiang-Lai/OpenDMD
│           - 单卡可训练
│           - 社区支持
│
├── 想训练 Flow 模型 (SD3, FLUX)
│   └── 推荐: XingtongGe/SenseFlow
│       - 专为 Flow 模型设计
│       - 解决收敛问题
│
├── 想要超越教师性能
│   └── 推荐: vvvvvjdy/dmdr
│       - RL + DMD 联合训练
│       - 可超越教师
│
└── 使用 AMD GPU
    └── 推荐: AMD-AIG-AIMA/AMD-Diffusion-Distillation
        - AMD 优化
        - 完整训练代码
```

### 按解决的问题选择

| 问题 | 推荐仓库 | 原因 |
|------|---------|------|
| 模式崩溃 | SenseFlow | IDA 正则化 |
| 训练不稳定 | DMD2 | 两时间尺度更新 |
| 资源有限 | OpenDMD / CIFAR-10 DMD | 单卡可训 |
| 想超越教师 | DMDR | RL 优化 |
| Flow 模型收敛困难 | SenseFlow | 专门设计 |
| 视频生成 | Wan2.2-Lightning | Phased DMD |

### 硬件需求汇总

| 仓库 | 最低 GPU | 推荐 GPU | 训练时间 |
|------|---------|---------|---------|
| DMD2 (ImageNet) | 7× A100 | 8× A100 | 38-70h |
| DMD2 (SD1.5) | 64× A100 | 64× A100 | ~25h |
| DMD2 (SDXL) | 64× A100 | 64× A100 | ~57h |
| OpenDMD | 1× RTX 3090 | 4× A100 | 可变 |
| CIFAR-10 DMD | 1× RTX 3090 | 1× A100 | 数小时 |
| SenseFlow | 8× A100 | 32× A100 | 可变 |

---

## 附录 A: 快速开始脚本

### 最简单的开始方式（CIFAR-10）

```bash
# 1. 克隆仓库
git clone https://github.com/devrimcavusoglu/dmd.git
cd dmd

# 2. 创建环境
conda env create -f environment.yml
conda activate dmd

# 3. 下载数据
bash scripts/download_distillation_dataset.sh

# 4. 开始训练
python -m dmd train \
    --model-path https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --data-path ./distillation_data \
    --output-dir ./output \
    --epochs 2 \
    --batch-size 32

# 5. 生成图像
python -m dmd generate \
    --network-path ./output/checkpoint.pt \
    --output-dir ./generated \
    --seeds 0-24 \
    --class-ids 0
```

### 使用预训练 DMD2 模型

```python
# pip install diffusers torch

from diffusers import DiffusionPipeline
import torch

# 加载 1 步模型
pipe = DiffusionPipeline.from_pretrained(
    "tianweiy/DMD2",
    torch_dtype=torch.float16
).to("cuda")

# 生成图像（单步！）
image = pipe(
    "a beautiful sunset over mountains",
    num_inference_steps=1,
    guidance_scale=0
).images[0]

image.save("output.png")
```

---

## 附录 B: 常见问题

### Q1: 训练时出现 NaN 怎么办？

**解决方案**:
1. 使用 BF16 而非 FP16（DMD2 要求）
2. 降低学习率
3. 检查 Fake Diffusion 更新频率

### Q2: 模式崩溃怎么办？

**解决方案**:
1. 使用 SenseFlow 的 IDA 正则化
2. 尝试 DP-DMD 的梯度阻断
3. 增加 GAN 损失权重

### Q3: 资源不足怎么办？

**解决方案**:
1. 从 CIFAR-10 实现开始学习
2. 使用 OpenDMD 的单卡训练
3. 使用 LoRA 微调而非全量训练

### Q4: 如何评估模型质量？

**指标**:
- FID (越低越好): 生成质量
- LPIPS (作为多样性指标时越高越好): 多样性
- CLIP Score: 文本对齐度
- Human Preference: 人类偏好

---

## 参考链接

1. [DMD2 官方仓库](https://github.com/tianweiy/DMD2)
2. [OpenDMD](https://github.com/Zeqiang-Lai/OpenDMD)
3. [CIFAR-10 DMD](https://github.com/devrimcavusoglu/dmd)
4. [DMDR](https://github.com/vvvvvjdy/dmdr)
5. [SenseFlow](https://github.com/XingtongGe/SenseFlow)
6. [Phased DMD 项目主页](https://x-niper.github.io/projects/Phased-DMD/)
7. [Wan2.2-Lightning](https://github.com/ModelTC/Wan2.2-Lightning)
8. [AMD Nitro-1](https://github.com/AMD-AIG-AIMA/AMD-Diffusion-Distillation)

---

*文档最后更新: 2025年2月*
