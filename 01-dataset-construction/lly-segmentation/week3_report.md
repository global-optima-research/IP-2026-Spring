# Week 3 Progress Report — Dataset100 Segmentation & Inpainting Pipeline

**Author:** Luluyan Liu
**Date:** 2026-03-08
**Server:** liuluyan@111.17.197.107
**Base Path:** `/data/liuluyan/`

---

## 1. Overview

本周在服务器上使用三个模型对 PVTT evaluation dataset（dataset100）中的样本数据进行了端到端的 segmentation 与 inpainting 实验，验证了完整的 product video try-on 数据构建 pipeline：

| 模型 | 功能 | 结果路径 |
|------|------|----------|
| **Grounded-SAM-2** | 文本驱动检测 + 分割 + 视频追踪 | `Grounded-SAM-2-clean/dataset100_results/` |
| **SAM2** | 自动掩码生成 + 点提示分割 + 视频追踪 | `SAM2/dataset100_results/` |
| **VideoPainter** | 视频 inpainting（目标移除/替换） | `VideoPainter/dataset100results/` |

**测试样本：**
- 图像：`handbag_1.jpg`, `sunglasses_1.jpg`（来自 `/data/datasets/pvtt_evaluation_datasets/product_images/`）
- 视频：`0001-handfan1.mp4`, `0006-handbag1_scene01.mp4`（来自 `/data/datasets/pvtt_evaluation_datasets/videos/`）

---

## 2. Grounded-SAM-2 Results

**路径：** `/data/liuluyan/Grounded-SAM-2-clean/dataset100_results/`

### 2.1 图像分割（Image Segmentation）

使用 Grounding DINO + SAM2 pipeline，通过文本提示（如 "handbag", "sunglasses"）自动检测并分割产品：

| 样本 | 输出文件 |
|------|----------|
| handbag_1 | `handbag_1_boxes.jpg`（检测框）, `handbag_1_masks.jpg`（分割掩码）, `handbag_1_results.json`（检测结果） |
| sunglasses_1 | `sunglasses_1_boxes.jpg`, `sunglasses_1_masks.jpg`, `sunglasses_1_results.json` |

**优势：** 文本驱动，无需手动标注提示点，可自动识别目标产品类别。

### 2.2 视频追踪（Video Tracking）

使用 Grounding DINO 在首帧检测产品 → SAM2 生成掩码 → 逐帧传播追踪：

| 样本 | 帧数 | 输出 |
|------|------|------|
| 0001-handfan1 | 434帧 | `0001-handfan1_tracking.mp4` + 逐帧追踪结果 |
| 0006-handbag1_scene01 | 77帧 | `0006-handbag1_scene01_tracking.mp4` |

---

## 3. SAM2 Results

**路径：** `/data/liuluyan/SAM2/dataset100_results/`

### 3.1 图像分割

#### Automatic Mask Generation（自动掩码生成）
对整张图像进行无提示分割，检测所有可分割区域：

| 样本 | 检测到的掩码数量 | 输出 |
|------|-----------------|------|
| handbag_1 | 21 个掩码 | `handbag_1_auto_masks.png`, `handbag_1_masks_info.json` |
| sunglasses_1 | 15 个掩码 | `sunglasses_1_auto_masks.png`, `sunglasses_1_masks_info.json` |

#### Point Prompt Prediction（点提示预测）
使用图像中心点作为正样本提示，分割中心区域的目标：

| 样本 | 输出 |
|------|------|
| handbag_1 | `handbag_1_best_mask.png`（得分最高的掩码） |
| sunglasses_1 | `sunglasses_1_best_mask.png` |

### 3.2 视频追踪（Video Tracking）

使用 SAM2 Video Predictor，在首帧中心点添加正样本提示，逐帧传播掩码：

| 样本 | 帧数 | 输出 |
|------|------|------|
| 0001-handfan1 | 434帧 | `0001-handfan1_tracking.mp4`, `0001-handfan1_frame0_prompt.png` |
| 0006-handbag1_scene01 | 77帧 | `0006-handbag1_scene01_tracking.mp4`, `0006-handbag1_scene01_frame0_prompt.png` |

**环境配置：**
- Conda 环境: `lly311` (Python 3.11)
- 模型: SAM 2.1 Hiera Large (`sam2.1_hiera_large.pt`)
- GPU: NVIDIA RTX 5090, CUDA 12.8, PyTorch 2.7.0

---

## 4. VideoPainter Results

**路径：** `/data/liuluyan/VideoPainter/dataset100results/`

VideoPainter 接收分割掩码，对视频中的产品区域进行 inpainting（移除/替换），用于构建训练数据对。

### 4.1 图像 Inpainting

| 样本 | 输出文件 |
|------|----------|
| image_handbag1 | `frame0_original.png`, `frame0_with_mask.png`, `mask0.png`, `result.mp4`, `result_vis.mp4` |
| image_sunglasses1 | 同上结构 |

### 4.2 视频 Inpainting

| 样本 | 输出文件 |
|------|----------|
| video_handbag1 | `frame0_original.png`, `frame0_with_mask.png`, `mask0.png`, `result.mp4`, `result_vis.mp4` |
| video_handfan1 | 同上结构 |

### 4.3 额外实验

- `image_handbag1_new` / `video_handbag1_new` / `video_handfan1_new`：使用更新的参数或掩码重新跑的结果
- `image_sunglasses1_dilate4`：对 sunglasses 掩码进行 4 像素膨胀后的 inpainting 效果
- `pseudo_videos`：由静态图像生成的伪视频数据
- `dataset100results_resized`：调整分辨率后的结果

---

## 5. Pipeline 总结

完整的 dataset100 数据构建 pipeline：

```
Product Image/Video
        │
        ▼
┌─────────────────────┐
│  Grounded-SAM-2     │  文本提示 → 检测 + 分割
│  (Detection + Seg)  │  输出: bounding boxes + masks
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  SAM2               │  点/框提示 → 精细分割 + 视频追踪
│  (Segmentation +    │  输出: auto masks + tracking masks
│   Video Tracking)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  VideoPainter       │  掩码 → 视频 inpainting
│  (Video Inpainting) │  输出: 移除产品后的视频
└─────────────────────┘
```

---

## 6. 文件结构

```
/data/liuluyan/
├── Grounded-SAM-2-clean/
│   └── dataset100_results/
│       ├── image_results/
│       │   ├── handbag_1/    (boxes.jpg, masks.jpg, results.json)
│       │   └── sunglasses_1/ (boxes.jpg, masks.jpg, results.json)
│       └── video_results/
│           ├── 0001-handfan1/    (tracking.mp4, frames/, tracking_results/)
│           └── 0006-handbag1_scene01/ (tracking.mp4)
├── SAM2/
│   ├── dataset100_results/
│   │   ├── image_results/
│   │   │   ├── handbag_1/    (auto_masks.png, best_mask.png, masks_info.json)
│   │   │   └── sunglasses_1/ (auto_masks.png, best_mask.png, masks_info.json)
│   │   └── video_results/
│   │       ├── 0001-handfan1/        (tracking.mp4, frame0_prompt.png)
│   │       └── 0006-handbag1_scene01/ (tracking.mp4, frame0_prompt.png)
│   ├── results/          (demo results: bunny, cars, etc.)
│   ├── run_image_inference.py
│   ├── run_video_inference.py
│   └── run_dataset100_inference.py
├── VideoPainter/
│   ├── dataset100results/
│   │   ├── image_handbag1/      (original, mask, result, result_vis)
│   │   ├── image_sunglasses1/
│   │   ├── video_handbag1/
│   │   ├── video_handfan1/
│   │   └── ...
│   └── dataset100results_resized/
└── repo/
    └── week3_report.md
```

---

## 7. 关键发现与 Next Steps

### 本周发现
1. **Grounded-SAM-2** 能通过文本提示自动检测产品，适合大规模自动化标注
2. **SAM2** 提供更精细的分割（21/15 个掩码），适合需要多粒度分割的场景
3. **VideoPainter** 可生成高质量的 inpainting 结果，成功移除产品区域
4. 三个工具形成完整 pipeline：检测 → 分割/追踪 → inpainting

### 待改进
- 当前使用图像中心点作为 SAM2 提示，实际应用中需结合 Grounded-SAM-2 的检测框作为提示
- 仅在 2 张图 + 2 个视频上测试，需扩展到完整 dataset100（35 张产品图 + 53 个视频）
- VideoPainter 的掩码膨胀参数（dilate）需要调优以获得更自然的 inpainting 边界
