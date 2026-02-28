# VideoPainter 复现指南

本文档记录在 **8x RTX 5090 (32GB)** 服务器上从零复现 VideoPainter 推理流程的完整步骤。

---

## 1. 服务器环境

| 项目 | 配置 |
|------|------|
| GPU | 8x NVIDIA GeForce RTX 5090 (32GB) |
| Driver | 580.65.06 |
| CUDA | 13.0 |
| OS | Ubuntu 22.04 |
| 内存 | ~1TB |

---

## 2. 创建 Conda 环境

```bash
conda create -n videopainter python=3.10 -y
conda activate videopainter
```

### 2.1 安装 PyTorch

RTX 5090 (Blackwell / sm_120) 需要 PyTorch 2.10+ 和 CUDA 12.8：

```bash
pip install torch==2.10.0 torchvision==0.22.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

> 如果是 A100/H100 等旧卡，可使用 `cu124` 或 `cu118` 版本。

### 2.2 安装项目依赖

```bash
pip install -r requirements.txt

# 安装自定义 diffusers
cd diffusers
pip install -e .
cd ..

# 安装 SAM2
cd app
pip install -e .
cd ..

# 安装 ffmpeg
conda install -c conda-forge ffmpeg -y
```

### 2.3 SAM2 C 扩展兼容性修复

SAM2 预编译的 `_C.so` 可能与当前 PyTorch 版本不兼容，报错：
```
ImportError: sam2/_C.so: undefined symbol: _ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_RKSs
```

运行修复脚本，用纯 Python (scipy) 实现替代 C 扩展：

```bash
python fix_sam2_c.py
```

---

## 3. 下载模型权重

所有模型存放在 `ckpt/` 目录下。中国大陆服务器建议使用 [hf-mirror.com](https://hf-mirror.com) 加速下载。

### 3.1 CogVideoX-5b-I2V (基础视频生成模型, ~21GB)

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="THUDM/CogVideoX-5b-I2V", local_dir="ckpt/CogVideoX-5b-I2V")
```

### 3.2 VideoPainter (Inpainting Branch, ~680MB)

```python
snapshot_download(repo_id="TencentARC/VideoPainter", local_dir="ckpt/VideoPainter")
```

### 3.3 VideoPainterID (LoRA 权重, ~505MB)

```python
snapshot_download(repo_id="TencentARC/VideoPainterID", local_dir="ckpt/VideoPainterID")
```

### 3.4 SAM2 (视频分割模型, ~857MB)

```bash
wget https://hf-mirror.com/facebook/sam2-hiera-large/resolve/main/sam2_hiera_large.pt -O ckpt/sam2_hiera_large.pt
```

### 3.5 FLUX.1-Fill-dev (首帧修复, ~55GB, 可选)

此模型为 Gated Repo，需要先在 [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) 申请访问权限并获取 token：

```python
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-Fill-dev",
    local_dir="ckpt/flux_inp",
    token="hf_YOUR_TOKEN_HERE"
)
```

> FLUX 模型为可选项，不影响核心 inpainting 功能。跳过后首帧将不做额外修复。

### 最终目录结构

```
ckpt/
├── CogVideoX-5b-I2V/          # ~21GB, 基础视频模型
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/
│   ├── transformer/
│   └── vae/
├── VideoPainter/               # ~680MB, Inpainting Branch
│   └── checkpoints/branch/
│       ├── config.json
│       └── diffusion_pytorch_model.safetensors
├── VideoPainterID/             # ~505MB, ID LoRA
│   └── checkpoints/
│       └── pytorch_lora_weights.safetensors
├── flux_inp/                   # ~55GB, FLUX (可选)
└── sam2_hiera_large.pt         # ~857MB, SAM2
```

---

## 4. 准备测试视频

将测试视频放入 `test_videos/` 目录：

```bash
mkdir -p test_videos
# 将你的视频复制到此目录，例如：
cp /path/to/your/video.mp4 test_videos/
```

---

## 5. 运行推理 (命令行模式)

使用 `run_demo.py` 一键完成 **SAM2 分割 → 目标追踪 → VideoPainter 背景修复** 全流程。

### 5.1 基本用法

```bash
cd infer
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_demo.py \
    --video ../test_videos/bunny.mp4 \
    --click_x 360 \
    --click_y 240 \
    --prompt "A clean surface with soft lighting" \
    --output_dir ../output
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 输入视频路径 | (必填) |
| `--click_x` | 目标对象在首帧的 X 坐标 (0~719) | (必填) |
| `--click_y` | 目标对象在首帧的 Y 坐标 (0~479) | (必填) |
| `--prompt` | 视频描述 (描述期望的背景) | (必填) |
| `--output_dir` | 输出目录 | `./output` |
| `--num_inference_steps` | 推理步数，越大质量越好 | 50 |
| `--guidance_scale` | CFG 引导强度 | 6.0 |
| `--seed` | 随机种子 | 42 |
| `--dilate_size` | Mask 膨胀大小 (像素) | 32 |
| `--target_fps` | 目标帧率 | 8 |
| `--max_seconds` | 最大处理时长 (秒) | 6.0 |
| `--mask_only` | 仅生成 mask，不跑 inpainting | - |

> **坐标说明**：视频会被缩放至 480x720 (高x宽)。click_x 范围 0~719，click_y 范围 0~479。可先用 `--mask_only` 查看分割效果，确认坐标正确后再跑完整流程。

### 5.2 分步运行（推荐）

**第一步：仅生成 mask，验证分割效果**

```bash
python run_demo.py \
    --video ../test_videos/bunny.mp4 \
    --click_x 360 --click_y 240 \
    --prompt "placeholder" \
    --output_dir ../output \
    --mask_only
```

检查 `output/frame0_with_mask.png` 确认 SAM2 是否正确选中目标。如果不准确，调整 `click_x` 和 `click_y`。

**第二步：运行完整 inpainting**

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_demo.py \
    --video ../test_videos/bunny.mp4 \
    --click_x 360 --click_y 240 \
    --prompt "A clean surface with soft lighting, no objects" \
    --output_dir ../output
```

### 5.3 输出文件

| 文件 | 说明 |
|------|------|
| `result.mp4` | 去除目标后的纯背景视频 |
| `result_vis.mp4` | 四格对比 (原始 \| 遮罩 \| mask \| 生成) |
| `frame0_with_mask.png` | 首帧 SAM2 分割可视化 |
| `frame0_original.png` | 原始首帧 |
| `mask0.png` | 目标对象的分割 mask |

---

## 6. 运行推理 (Shell 脚本模式)

如果已有预计算的 mask (`.npz` 格式) 和元数据 CSV，可使用原始推理脚本：

```bash
cd infer
bash inpaint.sh
```

需要准备：
- 预计算 mask：`data/video_inpainting/{videovo|pexels}/{video_id}/all_masks.npz`
- 元数据 CSV：`data/pexels_videovo_test_dataset.csv`
- 原始视频：`data/videovo/raw_video/` 或 `data/pexels/pexels/raw_video/`

---

## 7. 显存说明

| 模式 | 显存占用 | 说明 |
|------|----------|------|
| 完整加载 (pipe.to("cuda")) | ~31GB | 需要 >32GB 显存 (A100 80GB) |
| CPU Offload (enable_model_cpu_offload) | ~27GB | 32GB 显卡可用 (RTX 5090/4090) |
| + VAE Slicing/Tiling | ~27GB | 进一步优化长视频 |

`run_demo.py` 已默认启用 CPU Offload + VAE Slicing/Tiling。

单卡 RTX 5090 (32GB) 处理 33 帧 (480x720)，推理约 2~3 分钟。

---

## 8. 常见问题

### Q: SAM2 报错 `undefined symbol`
运行 `python fix_sam2_c.py` 修复 C 扩展兼容问题。

### Q: CUDA OOM
- 确保设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 脚本已使用 `enable_model_cpu_offload()`，如仍 OOM 可减少 `--max_seconds`
- 确认没有其他进程占用 GPU：`nvidia-smi`

### Q: 视频输出比例被压缩
模型固定在 480x720 分辨率上推理，输出视频会是 480x720。如需恢复原始比例，可用 ffmpeg 后处理：
```bash
ffmpeg -i output/result.mp4 -vf "scale=1080:1080" -c:v libx264 output/result_1080.mp4
```

### Q: cv2.VideoWriter 输出的视频无法播放
用 ffmpeg 转为 H.264 编码：
```bash
ffmpeg -y -i result.mp4 -c:v libx264 -pix_fmt yuv420p result_h264.mp4
```

---

## 9. 引用

```bibtex
@article{bian2025videopainter,
  title={VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control},
  author={Bian, Yuxuan and Zhang, Zhaoyang and Ju, Xuan and Cao, Mingdeng and Xie, Liangbin and Shan, Ying and Xu, Qiang},
  journal={arXiv preprint arXiv:2503.05639},
  year={2025}
}
```
