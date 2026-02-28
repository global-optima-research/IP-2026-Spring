# SAM 2 复现指南 (Running Guide)

本文档记录了在服务器上复现 SAM 2 (Segment Anything Model 2) 图像分割与视频追踪的完整流程。

---

## 目录

1. [环境信息](#1-环境信息)
2. [项目结构](#2-项目结构)
3. [环境安装](#3-环境安装)
4. [模型权重下载](#4-模型权重下载)
5. [验证安装](#5-验证安装)
6. [图像分割推理](#6-图像分割推理)
7. [视频追踪推理](#7-视频追踪推理)
8. [结果说明](#8-结果说明)
9. [常见问题](#9-常见问题)

---

## 1. 环境信息

| 项目 | 信息 |
|------|------|
| 服务器 | liuluyan@111.17.197.107 |
| 项目路径 | `/data/liuluyan/SAM2` |
| Conda 环境 | lly311 |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| GPU | NVIDIA RTX 5090 |
| CUDA | 12.8 |
| 网络 | 服务器无外网，需本地上传模型和依赖 |

---

## 2. 项目结构

```
/data/liuluyan/SAM2/
├── run_image_inference.py          # 图像分割推理脚本（自动掩码 + 点提示）
├── run_video_inference.py          # 视频追踪推理脚本
├── results/
│   ├── images/                     # 图像分割结果
│   │   ├── *_auto_masks.png        # 自动掩码生成结果
│   │   └── *_best_mask.png         # 点提示最佳掩码结果
│   └── video/                      # 视频追踪结果
│       ├── bunny_segmented.mp4     # 追踪输出视频
│       └── frame_000_prompt.png    # 第一帧提示可视化
├── sam2/                           # SAM 2 源码（克隆自 facebook/sam2）
│   ├── sam2/                       # 核心模型代码
│   │   ├── build_sam.py
│   │   ├── automatic_mask_generator.py
│   │   ├── sam2_image_predictor.py
│   │   ├── sam2_video_predictor.py
│   │   ├── configs/                # 模型配置文件（yaml）
│   │   ├── modeling/               # 模型结构（backbone, SAM head 等）
│   │   └── utils/                  # 工具函数
│   ├── notebooks/
│   │   ├── images/                 # 输入图像
│   │   │   ├── bai.jpg
│   │   │   ├── bunny.png
│   │   │   ├── cars.jpg
│   │   │   ├── groceries.jpg
│   │   │   └── truck.jpg
│   │   ├── videos/                 # 输入视频
│   │   │   ├── bunny.mp4           # 输入视频
│   │   │   └── bunny_frames/       # 从视频提取的帧（运行时自动生成）
│   │   ├── automatic_mask_generator_example.ipynb
│   │   ├── image_predictor_example.ipynb
│   │   └── video_predictor_lly311.ipynb   # 自定义 notebook
│   ├── verify.py                   # 安装验证脚本
│   ├── setup.py
│   └── pyproject.toml
```

---

## 3. 环境安装

### 3.1 激活 Conda 环境

```bash
source /data/liuluyan/miniconda3/etc/profile.d/conda.sh
conda activate lly311
```

### 3.2 安装 SAM 2

```bash
cd /data/liuluyan/SAM2/sam2
pip install -e ".[notebooks]"
```

> **注意：** 服务器无外网访问。如果需要安装依赖包，必须从本地机器下载后上传到服务器。

> **RTX 5090 用户注意：** 需要 PyTorch >= 2.7.0 和 CUDA >= 12.8 才能支持 RTX 5090（sm_120 架构）。

### 3.3 关键依赖

```
torch >= 2.7.0
torchvision
numpy
matplotlib
Pillow
opencv-python (cv2)
hydra-core
iopath
ffmpeg          # 系统工具，用于视频帧提取
```

---

## 4. 模型权重下载

本项目使用 **SAM 2.1 Hiera Large** 模型。权重文件位于：

```
/data/liuluyan/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
```

> 该权重文件与 Grounded-SAM-2 项目共享，无需重复下载。

如需重新下载（需在有网络的机器上操作）：

```bash
# 在本地机器下载
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 上传到服务器
scp sam2.1_hiera_large.pt liuluyan@111.17.197.107:/data/liuluyan/Grounded-SAM-2/checkpoints/
```

### 可用模型列表

| 模型 | 配置文件 | 权重文件 |
|------|----------|----------|
| Hiera Tiny | `configs/sam2.1/sam2.1_hiera_t.yaml` | `sam2.1_hiera_tiny.pt` |
| Hiera Small | `configs/sam2.1/sam2.1_hiera_s.yaml` | `sam2.1_hiera_small.pt` |
| Hiera B+ | `configs/sam2.1/sam2.1_hiera_b+.yaml` | `sam2.1_hiera_base_plus.pt` |
| **Hiera Large** | **`configs/sam2.1/sam2.1_hiera_l.yaml`** | **`sam2.1_hiera_large.pt`** |

---

## 5. 验证安装

```bash
cd /data/liuluyan/SAM2/sam2
python verify.py
```

验证脚本会：
1. 检查 PyTorch 和 CUDA 是否可用
2. 尝试加载 SAM 2 模型
3. 输出 "模型加载成功！安装完成。" 表示环境就绪

---

## 6. 图像分割推理

### 6.1 运行

```bash
source /data/liuluyan/miniconda3/etc/profile.d/conda.sh
conda activate lly311
cd /data/liuluyan/SAM2
python run_image_inference.py
```

### 6.2 功能说明

脚本包含两部分推理：

**Part 1 — 自动掩码生成 (Automatic Mask Generation)**

- 使用 `SAM2AutomaticMaskGenerator` 对每张图像自动检测所有可分割区域
- 不需要任何人工提示（prompt-free）
- 对每个检测到的区域用不同颜色的半透明掩码覆盖，蓝色边界线标注轮廓
- 输出文件命名：`{图片名}_auto_masks.png`

**Part 2 — 点提示分割 (Point Prompt Prediction)**

- 使用 `SAM2ImagePredictor`，以图像中心点作为正样本提示（positive click）
- 开启 `multimask_output=True`，生成多个候选掩码并按置信度排序
- 保存得分最高的掩码
- 输出文件命名：`{图片名}_best_mask.png`

### 6.3 输入图像

| 文件 | 说明 |
|------|------|
| `bai.jpg` | 高分辨率图像（脚本会自动缩放至 2048px 以内防止 OOM） |
| `bunny.png` | 兔子图像 |
| `cars.jpg` | 停车场车辆 |
| `groceries.jpg` | 超市货架 |
| `truck.jpg` | 卡车 |

输入目录：`/data/liuluyan/SAM2/sam2/notebooks/images/`

### 6.4 输出结果

输出目录：`/data/liuluyan/SAM2/results/images/`

```
bai_auto_masks.png           # bai.jpg 的自动掩码
bai_best_mask.png            # bai.jpg 的点提示最佳掩码
bunny_auto_masks.png         # bunny.png 的自动掩码
bunny_best_mask.png          # bunny.png 的点提示最佳掩码
cars_auto_masks.png          # cars.jpg 的自动掩码
cars_best_mask.png           # cars.jpg 的点提示最佳掩码
groceries_auto_masks.png     # groceries.jpg 的自动掩码
groceries_best_mask.png      # groceries.jpg 的点提示最佳掩码
truck_auto_masks.png         # truck.jpg 的自动掩码
truck_best_mask.png          # truck.jpg 的点提示最佳掩码
```

---

## 7. 视频追踪推理

### 7.1 运行

```bash
source /data/liuluyan/miniconda3/etc/profile.d/conda.sh
conda activate lly311
cd /data/liuluyan/SAM2
python run_video_inference.py
```

### 7.2 流程说明

脚本分为 5 个步骤：

| 步骤 | 说明 |
|------|------|
| **Step 1** | **提取视频帧** — 使用 ffmpeg 将 `bunny.mp4` 拆分为 JPEG 帧序列，存放在 `bunny_frames/`。如果帧已存在则跳过。 |
| **Step 2** | **加载模型** — 使用 `build_sam2_video_predictor` 加载 SAM 2.1 Large 模型并初始化推理状态（inference_state）。 |
| **Step 3** | **添加提示** — 在第 0 帧的图像中心点添加一个正样本点击（positive click），标记追踪目标。保存第一帧的提示可视化图。 |
| **Step 4** | **传播掩码** — 调用 `propagate_in_video` 将掩码从第 0 帧自动传播到整个视频的所有帧。 |
| **Step 5** | **保存结果** — 将追踪掩码（半透明彩色区域 + 白色轮廓线）叠加到原始帧上，合成输出视频。每隔 5 帧另存一张可视化 PNG。 |

### 7.3 输入

- 视频文件：`/data/liuluyan/SAM2/sam2/notebooks/videos/bunny.mp4`

### 7.4 输出结果

输出目录：`/data/liuluyan/SAM2/results/video/`

```
frame_000_prompt.png         # 第一帧的提示点（绿色星号）和初始掩码可视化
bunny_segmented.mp4          # 完整的追踪结果视频（15 FPS，带掩码叠加和轮廓线）
```

---

## 8. 结果说明

### 图像分割结果

- **自动掩码 (`*_auto_masks.png`)**：SAM 2 自动检测图像中所有可分割区域，每个区域用不同半透明颜色表示，蓝色边界线标注轮廓。区域按面积从大到小排序绘制。
- **点提示掩码 (`*_best_mask.png`)**：以图像中心为提示点（绿色星号标记），分割出置信度最高的目标区域（蓝色半透明覆盖）。

### 视频追踪结果

- **`bunny_segmented.mp4`**：在视频每一帧上叠加了追踪掩码（半透明彩色区域 + 白色轮廓），可以看到目标从第一帧被持续追踪到最后一帧。
- 输出视频帧率：15 FPS
- 掩码混合比例：原始像素 40% + 掩码颜色 60%

---

## 9. 常见问题

### Q: 运行时报 `MissingConfigException: Cannot find primary config`

确保已在 SAM 2 源码目录执行过安装：

```bash
cd /data/liuluyan/SAM2/sam2
pip install -e ".[notebooks]"
```

### Q: 运行图像推理时 OOM（显存不足）

脚本已内置自动缩放功能：超过 2048px 的图像会自动等比缩放。如果仍然 OOM，可以在 `run_image_inference.py` 中将 `MAX_IMAGE_SIZE` 改小：

```python
MAX_IMAGE_SIZE = 1024  # 从 2048 改为 1024
```

### Q: 找不到权重文件

脚本中权重路径硬编码为：

```
/data/liuluyan/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
```

确保该文件存在。如需修改路径，编辑脚本中的 `CHECKPOINT` 变量。

### Q: ffmpeg 不可用导致视频帧提取失败

确认 ffmpeg 已安装：

```bash
ffmpeg -version
```

如未安装，可通过 conda 安装：

```bash
conda install -c conda-forge ffmpeg
```

### Q: `ImportError: cannot import name '_C' from 'sam2'`

需要编译 CUDA 扩展：

```bash
cd /data/liuluyan/SAM2/sam2
python setup.py build_ext --inplace
```

如果编译失败也不影响核心功能，只是会跳过后处理步骤（移除掩码中的小孔洞和碎片）。

### Q: 如何更换追踪目标？

编辑 `run_video_inference.py` 中 Step 3 的提示点坐标：

```python
# 修改这两行来指定目标位置（像素坐标）
center_x, center_y = 300, 200
points = np.array([[center_x, center_y]], dtype=np.float32)
```

### Q: 如何同时追踪多个目标？

在 Step 3 中为每个目标分别添加提示，使用不同的 `obj_id`：

```python
# 目标 1
predictor.add_new_points_or_box(
    inference_state=inference_state, frame_idx=0, obj_id=1,
    points=np.array([[x1, y1]], dtype=np.float32),
    labels=np.array([1], np.int32))

# 目标 2
predictor.add_new_points_or_box(
    inference_state=inference_state, frame_idx=0, obj_id=2,
    points=np.array([[x2, y2]], dtype=np.float32),
    labels=np.array([1], np.int32))
```

每个目标会在输出视频中以不同颜色的掩码显示。
