# Week 2 工作汇报：基于 SAM2 与 Grounded-SAM2 的目标分割复现

**负责人**: LLY (Liu Luyan)
**时间**: 2026 Spring - Week 2 
**所属模块**: 01-dataset-construction (数据集构建)

---

## 1. 工作概述
本周主要完成了 PVTT 数据集构建管线中 **“视频目标分割”** 模块的调研与复现。
重点对比了 **SAM2 (Segment Anything Model 2)** 原生模型与 **Grounded-SAM2** (结合文本提示的分割) 在电商场景下的表现，并跑通了从图片到视频的完整推理流程。

## 2. 目录结构说明 (`lly-segmentation`)

本项目文件组织如下，分为数据源、Grounded-SAM2 实验与 SAM2 实验三部分：

### 数据准备 (`demo_dataset/`)
存放用于测试的原始素材，包含不同类别的电商场景：
- **`images/`**: 单帧测试图片（如车辆、杂货等）。
- **`videos/`**: 视频测试片段（如 bunny 视频）。

### 核心实验一：Grounded-SAM2 (`gsam2/`)
*结合 Grounding DINO 与 SAM2，实现通过文本提示（Text Prompt）自动分割目标，是本项目自动化的关键。*

- **代码文件**:
  - `gsam2img.ipynb`: **[图片推理]** 输入文本（如 "truck"），输出分割 Mask。
  - `gsam2video.ipynb`: **[视频推理]** 输入首帧文本提示，利用 SAM2 在视频中跟踪分割对象。
- **实验结果**:
  - `image_results/`: 图片分割结果展示。
    - `fixed_bai.jpg`, `fixed_bunny.png`: 针对不同物体的分割测试。
    - `fixed_cars.jpg`, `fixed_truck.jpg`: 复杂背景下的车辆分割。
  - `videos_results/`: 视频分割结果展示。
    - `bunny_result.mp4`: 原始 Mask 叠加视频。
    - `bunny_result_bbox.mp4`: 包含检测框 (Bounding Box) 的可视化视频。

### 对照实验二：SAM2 原生复现 (`sam2/`)
*使用官方原生代码进行复现，主要测试点提示（Point Prompt）的交互效果。*

- `image_predictor_example.ipynb`: 单张图片的交互式分割复现。
- `video_predictor_colab.ipynb`: 视频流中的 Mask 传播机制复现。

---

## 3. 关键结论与观察
1. **自动化优势**: `Grounded-SAM2` 更适合我们的任务。因为它不需要人工在图上点点（Point Prompt），直接输入类别名称（如 "product"）即可生成 Mask，利于后续大规模批量处理。
2. **视频稳定性**: 在 `bunny_result.mp4` 的测试中，即使目标发生移动和形变，SAM2 的视频跟踪能力依然能保持 Mask 的边缘贴合。

## 4. 下周计划 (Week 3)
**目标**: 视频修复 (Video Inpainting)
1. 基于本周生成的 Mask（如 `videos_results` 中的结果），调研并部署 **Video Inpainting** 模型（如 ProPainter 或 E2FGVI）。
2. 尝试将视频中的目标（如 bunny）完全移除，恢复出干净的背景视频，为后续合成做准备。

---
