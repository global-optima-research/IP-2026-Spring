# Grounded SAM 2 - Complete Running Guide

This guide walks you through setting up and running the **Grounded SAM 2** pipeline from scratch, covering environment setup, model downloads, and running the demo scripts for both image segmentation and video object tracking.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Hardware & Software Requirements](#2-hardware--software-requirements)
- [3. Environment Setup](#3-environment-setup)
- [4. Install Dependencies](#4-install-dependencies)
  - [4.1 Install SAM 2](#41-install-sam-2)
  - [4.2 Install Grounding DINO (Local)](#42-install-grounding-dino-local)
  - [4.3 Install Additional Python Packages](#43-install-additional-python-packages)
- [5. Download Model Checkpoints](#5-download-model-checkpoints)
  - [5.1 SAM 2.1 Checkpoints](#51-sam-21-checkpoints)
  - [5.2 Grounding DINO Checkpoints](#52-grounding-dino-checkpoints)
  - [5.3 BERT Base Uncased (Text Encoder)](#53-bert-base-uncased-text-encoder)
- [6. Prepare Input Data](#6-prepare-input-data)
- [7. Running the Demos](#7-running-the-demos)
  - [7.1 Image Segmentation (Local Grounding DINO)](#71-image-segmentation-local-grounding-dino)
  - [7.2 Video Object Tracking (HuggingFace Grounding DINO)](#72-video-object-tracking-huggingface-grounding-dino)
  - [7.3 Video Object Tracking (Local Grounding DINO)](#73-video-object-tracking-local-grounding-dino)
- [8. Project Structure](#8-project-structure)
- [9. Troubleshooting](#9-troubleshooting)

---

## 1. Project Overview

Grounded SAM 2 combines two powerful models:

- **Grounding DINO**: An open-set object detector that detects objects based on text prompts (e.g., "car", "stuffed toy").
- **SAM 2 (Segment Anything Model 2)**: A segmentation model that can segment any object in images and propagate masks across video frames.

**Pipeline (Video Tracking)**:
```
Text Prompt + First Frame
        |
        v
  Grounding DINO  -->  Bounding Boxes
        |
        v
  SAM 2 Image Predictor  -->  Segmentation Masks (first frame)
        |
        v
  SAM 2 Video Predictor  -->  Propagated Masks (all frames)
        |
        v
  Visualization  -->  Annotated Video
```

---

## 2. Hardware & Software Requirements

| Requirement | Specification |
|---|---|
| **GPU** | NVIDIA GPU with CUDA support (8GB+ VRAM recommended) |
| **CUDA Toolkit** | >= 11.8 (match your PyTorch version) |
| **Python** | >= 3.10 |
| **PyTorch** | >= 2.3.1 |
| **OS** | Linux recommended; Windows via WSL also works |

> **Note on CUDA versions**:
> - For RTX 30xx/40xx: CUDA 11.8 or 12.1 works fine
> - For RTX 5090 / Blackwell GPUs: You need **CUDA >= 12.8**
> - Check your CUDA version with `nvcc --version`

---

## 3. Environment Setup

We recommend using **conda** to manage the environment.

```bash
# Create a new conda environment
conda create -n gsam2 python=3.11 -y
conda activate gsam2

# Install PyTorch (choose the command matching your CUDA version)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4+:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

---

## 4. Install Dependencies

### 4.1 Install SAM 2

From the **project root directory**:

```bash
cd Grounded-SAM-2

# Install SAM 2 in editable mode (this also builds the CUDA extension)
pip install -e ".[notebooks]"
```

If CUDA extension build fails, you can skip it (won't affect most results):

```bash
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

**Verify SAM 2 installation**:
```bash
python -c "from sam2.build_sam import build_sam2; print('SAM 2 OK')"
```

### 4.2 Install Grounding DINO (Local)

Grounding DINO has a CUDA extension for the deformable attention operator. You need to compile it:

```bash
cd grounding_dino

# Install Grounding DINO with CUDA ops
pip install --no-build-isolation -e .

cd ..
```

> **For newer CUDA versions (>= 13.0)** that don't support older architectures (e.g., `compute_70`), set the `TORCH_CUDA_ARCH_LIST` environment variable before building:
> ```bash
> # Example for RTX 5090 (sm_120):
> TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation -e .
>
> # Example for RTX 4090 (sm_89):
> TORCH_CUDA_ARCH_LIST="8.9" pip install --no-build-isolation -e .
>
> # Example for RTX 3090 (sm_86):
> TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -e .
> ```
> You can find your GPU's compute capability at: https://developer.nvidia.com/cuda-gpus

**Verify Grounding DINO installation**:
```bash
python -c "from groundingdino.models import build_model; print('Grounding DINO OK')"
```

### 4.3 Install Additional Python Packages

```bash
pip install "transformers<=4.44.2"   # IMPORTANT: version > 4.44 breaks BertModel API
pip install supervision>=0.22.0
pip install addict yapf timm
pip install opencv-python pycocotools
```

> **Why `transformers<=4.44.2`?**
> Grounding DINO's `BertModelWarper` calls `self.get_head_mask()`, which was removed in `transformers >= 5.0`. Using version 4.44.2 ensures compatibility.

---

## 5. Download Model Checkpoints

### 5.1 SAM 2.1 Checkpoints

Download the SAM 2.1 checkpoint into `checkpoints/`:

```bash
cd checkpoints

# Download SAM 2.1 Large (recommended, 898MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Or download all available sizes:
# bash download_ckpts.sh

cd ..
```

Available SAM 2.1 checkpoints:

| Model | Size | URL |
|---|---|---|
| sam2.1_hiera_tiny | 156MB | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt` |
| sam2.1_hiera_small | 184MB | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt` |
| sam2.1_hiera_base_plus | 324MB | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt` |
| **sam2.1_hiera_large** | **898MB** | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` |

### 5.2 Grounding DINO Checkpoints

Download the Grounding DINO checkpoint into `gdino_checkpoints/`:

```bash
cd gdino_checkpoints

# Download Grounding DINO SwinT (694MB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Or download all checkpoints:
# bash download_ckpts.sh

cd ..
```

### 5.3 BERT Base Uncased (Text Encoder)

Grounding DINO uses **BERT** as its text encoder. You need to download the `bert-base-uncased` model locally.

**Option A: Using Python (recommended)**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    local_dir="./bert-base-uncased",
    token="YOUR_HF_TOKEN"  # optional, only needed if rate-limited
)
```

**Option B: Using `git clone`**

```bash
# Install git-lfs first
git lfs install
git clone https://huggingface.co/google-bert/bert-base-uncased ./bert-base-uncased
```

**Option C: Auto-download (requires internet)**

If your machine has internet access, Grounding DINO can auto-download `bert-base-uncased` from HuggingFace at runtime. No manual download is needed. However, if you're on an **offline machine** (e.g., a cluster without internet), you must download it beforehand and place it in the project root as `./bert-base-uncased/`.

After download, verify the directory structure:
```
bert-base-uncased/
  config.json
  model.safetensors (or pytorch_model.bin)
  tokenizer.json
  tokenizer_config.json
  vocab.txt
```

---

## 6. Prepare Input Data

### For Image Segmentation

Place your input images anywhere and update the `IMG_PATH` variable in the script. Demo images are already provided:

```
notebooks/images/
  cars.jpg
  groceries.jpg
  truck.jpg
```

### For Video Object Tracking

SAM 2 video predictor requires **a directory of JPEG frames** (not a video file directly).

**Convert a video to frames**:
```bash
# Using ffmpeg
ffmpeg -i your_video.mp4 -q:v 2 -start_number 0 "frames/%05d.jpg"

# Or using Python
python -c "
import cv2, os
os.makedirs('frames', exist_ok=True)
cap = cv2.VideoCapture('your_video.mp4')
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f'frames/{idx:05d}.jpg', frame)
    idx += 1
cap.release()
print(f'Extracted {idx} frames')
"
```

A demo video is already provided at `notebooks/videos/bunny.mp4`.

---

## 7. Running the Demos

### 7.1 Image Segmentation (Local Grounding DINO)

This demo detects objects in a **single image** using local Grounding DINO + SAM 2 and outputs annotated images with bounding boxes and masks.

```bash
python grounded_sam2_local_demo.py
```

**Default settings** (edit in script):
- Input: `notebooks/images/truck.jpg`
- Text prompt: `"car. tire."`
- Output: `outputs/grounded_sam2_local_demo/`

**To customize**, modify these variables at the top of the script:
```python
TEXT_PROMPT = "your object. another object."   # MUST be lowercase + end with dot
IMG_PATH = "path/to/your/image.jpg"
BOX_THRESHOLD = 0.35     # confidence threshold for detection
TEXT_THRESHOLD = 0.25     # text matching threshold
```

### 7.2 Video Object Tracking (HuggingFace Grounding DINO)

This demo uses the **HuggingFace API** version of Grounding DINO (auto-downloaded) + SAM 2 video predictor to track objects across video frames.

> **Requires internet access** to download the HuggingFace model on first run.

```bash
python grounded_sam2_tracking_demo.py
```

**Default settings**:
- Input: `notebooks/videos/car/` (JPEG frames)
- Text prompt: `"car."`
- Output: `./tracking_results/` + `./children_tracking_demo_video.mp4`

### 7.3 Video Object Tracking (Local Grounding DINO)

This demo uses **local Grounding DINO** (no internet needed) + SAM 2 video predictor. This is ideal for **offline/cluster environments**.

**Before running**, ensure you have:
1. Extracted video frames (see [Section 6](#6-prepare-input-data))
2. Downloaded all three model checkpoints (SAM 2.1, Grounding DINO, bert-base-uncased)

```bash
python run_bunny_tracking.py
```

**Default settings**:
- Input: `notebooks/videos/bunny_frames/` (JPEG frames extracted from bunny.mp4)
- Text prompt: `"stuffed toy. plush rabbit."`
- SAM 2 checkpoint: `./checkpoints/sam2.1_hiera_large.pt`
- Grounding DINO checkpoint: `gdino_checkpoints/groundingdino_swint_ogc.pth`
- BERT model: `./bert-base-uncased`
- Output annotated frames: `./notebooks/videos/bunny_tracking_results/`
- Output video: `./notebooks/videos/bunny_tracking_result.mp4`

**To customize for your own video**, edit these variables in `run_bunny_tracking.py`:

```python
text = "your object. another object."          # text prompt (lowercase + dot)
video_dir = "path/to/your/video_frames"        # directory of JPEG frames
BOX_THRESHOLD = 0.30                           # detection confidence threshold
TEXT_THRESHOLD = 0.25                           # text matching threshold
```

**Script workflow**:

1. Load SAM 2 (video predictor + image predictor)
2. Load local Grounding DINO with local bert-base-uncased
3. Detect objects in the **first frame** using text prompts
4. Segment detected objects using SAM 2 image predictor
5. Register masks to SAM 2 video predictor
6. Propagate masks across **all frames** automatically
7. Render annotated frames (bounding boxes + labels + masks)
8. Create output video from annotated frames

---

## 8. Project Structure

```
Grounded-SAM-2/
|
|-- sam2/                          # SAM 2 model source code
|   |-- build_sam.py               # Build SAM 2 predictor
|   |-- sam2_image_predictor.py    # Image segmentation predictor
|   |-- sam2_video_predictor.py    # Video tracking predictor
|   |-- configs/                   # SAM 2 model config YAML files
|   |   |-- sam2.1/                # SAM 2.1 configs
|   |   |-- sam2/                  # SAM 2.0 configs
|   |-- modeling/                  # Model architecture
|   |-- csrc/                      # CUDA source code
|   `-- utils/                     # Utility functions
|
|-- grounding_dino/                # Grounding DINO model source code
|   |-- groundingdino/
|   |   |-- config/                # Model config files (SwinT, SwinB)
|   |   |-- models/GroundingDINO/  # Model architecture + CUDA ops
|   |   |-- datasets/              # Data transforms
|   |   `-- util/                  # Inference, tokenizer, misc utilities
|   |-- setup.py                   # Build Grounding DINO CUDA extension
|   `-- requirements.txt           # Grounding DINO dependencies
|
|-- utils/                         # Project utility scripts
|   |-- video_utils.py             # Create video from image frames
|   |-- track_utils.py             # Point sampling from masks
|   |-- mask_dictionary_model.py   # Mask data model
|   |-- common_utils.py            # Common helpers
|   `-- supervision_utils.py       # Visualization helpers
|
|-- notebooks/                     # Jupyter notebook examples
|   |-- image_predictor_example.ipynb
|   |-- video_predictor_example.ipynb
|   |-- automatic_mask_generator_example.ipynb
|   |-- images/                    # Demo images
|   `-- videos/                    # Demo videos (bunny.mp4)
|
|-- checkpoints/                   # SAM 2 model weights (download required)
|   `-- download_ckpts.sh
|
|-- gdino_checkpoints/             # Grounding DINO weights (download required)
|   `-- download_ckpts.sh
|
|-- bert-base-uncased/             # BERT text encoder (download required)
|
|-- run_bunny_tracking.py          # Video tracking demo (local, offline)
|-- grounded_sam2_local_demo.py    # Image segmentation demo (local)
|-- grounded_sam2_tracking_demo.py # Video tracking demo (HuggingFace)
|-- grounded_sam2_hf_model_demo.py # Image demo (HuggingFace)
|-- setup.py                       # SAM 2 package setup
|-- pyproject.toml                 # Build system config
`-- tools/                         # Additional tools (VOS inference)
```

---

## 9. Troubleshooting

### `NameError: name '_C' is not defined` (Grounding DINO)

The Grounding DINO CUDA extension is not compiled. Rebuild it:

```bash
cd grounding_dino
pip install --no-build-isolation -e .
cd ..
```

### `nvcc fatal: Unsupported gpu architecture 'compute_70'`

Your CUDA version is too new and doesn't support older architectures. Set `TORCH_CUDA_ARCH_LIST`:

```bash
# Replace with your GPU's compute capability
TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -e .
```

Common compute capabilities:
| GPU | Compute Capability |
|---|---|
| RTX 3060/3070/3080/3090 | 8.6 |
| RTX 4060/4070/4080/4090 | 8.9 |
| RTX 5090 (Blackwell) | 12.0 |
| A100 | 8.0 |
| H100 | 9.0 |

### `'BertModel' object has no attribute 'get_head_mask'`

You have `transformers >= 5.0` installed, which removed this method. Downgrade:

```bash
pip install "transformers<=4.44.2"
```

### `OSError: Can't load tokenizer for 'bert-base-uncased'`

The BERT model files are not found. Either:
1. Download `bert-base-uncased` locally (see [Section 5.3](#53-bert-base-uncased-text-encoder))
2. Or ensure your machine has internet access for auto-download

For local usage, make sure the script's `text_encoder_type` points to the correct path:
```python
args.text_encoder_type = "./bert-base-uncased"
```

### `MissingConfigException: Cannot find primary config 'configs/sam2.1/...'`

SAM 2 is not properly installed. Reinstall from the project root:

```bash
pip install -e .
```

### `CUDA error: no kernel image is available for execution on the device`

The CUDA kernel was compiled for a different GPU architecture. Recompile with your GPU's architecture:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"  # your GPU's compute capability
pip install -e .                    # rebuild SAM 2
cd grounding_dino && pip install --no-build-isolation -e . && cd ..  # rebuild Grounding DINO
```

### No objects detected

- Ensure your text prompt is **lowercase** and **ends with a dot** (e.g., `"car."` not `"Car"`)
- Try lowering `BOX_THRESHOLD` (e.g., from 0.35 to 0.25)
- Use more descriptive prompts separated by dots (e.g., `"stuffed toy. plush rabbit."`)

### Out of memory (OOM)

- Use a smaller SAM 2 checkpoint (e.g., `sam2.1_hiera_tiny.pt` instead of `sam2.1_hiera_large.pt`)
- Reduce input image/video resolution
- Use `torch.cuda.empty_cache()` between processing steps

---

## Quick Start (TL;DR)

```bash
# 1. Setup environment
conda create -n gsam2 python=3.11 -y && conda activate gsam2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install packages
cd Grounded-SAM-2
pip install -e ".[notebooks]"
cd grounding_dino && pip install --no-build-isolation -e . && cd ..
pip install "transformers<=4.44.2" supervision>=0.22.0 addict yapf timm opencv-python pycocotools

# 3. Download checkpoints
cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt && cd ..
cd gdino_checkpoints && wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && cd ..

# 4. Download BERT (for local Grounding DINO)
python -c "from huggingface_hub import snapshot_download; snapshot_download('google-bert/bert-base-uncased', local_dir='./bert-base-uncased')"

# 5. Prepare video frames
ffmpeg -i notebooks/videos/bunny.mp4 -q:v 2 -start_number 0 "notebooks/videos/bunny_frames/%05d.jpg"

# 6. Run!
python run_bunny_tracking.py
```
