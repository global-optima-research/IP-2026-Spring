#!/bin/bash
# setup_server.sh — One-time server environment setup for FastGen
# Run on server: bash setup_server.sh

set -euo pipefail

CONDA_DIR="/data/chenqingzhan/miniconda3"
ENV_NAME="fastgen"
FASTGEN_DIR="/data/chenqingzhan/FastGen"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"

echo "=== Step 1: Configure pip mirror (China) ==="
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'PIPEOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
PIPEOF
echo "pip mirror set to Aliyun"

echo "=== Step 2: Install PyTorch with CUDA 12.8 ==="
# RTX 5090 (Blackwell SM 10.0) needs cu128 for native support
# Try official PyTorch source first, fallback to cu126 if cu128 unavailable
echo "Attempting cu128 install..."
if ! $PIP install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --timeout 600 2>&1 | tee -a /tmp/pytorch_install.log | tail -10; then
    echo "cu128 failed, trying cu126..."
    $PIP install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126 \
        --timeout 600 2>&1 | tail -10
fi
echo "PyTorch install done"

echo "=== Step 3: Install FastGen ==="
cd $FASTGEN_DIR
$PIP install -e . 2>&1 | tail -10
echo "FastGen install done"

echo "=== Step 4: Verify installation ==="
$PYTHON -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

echo "=== Step 5: Set environment variables ==="
cat >> ~/.bashrc << 'BASHEOF'

# FastGen environment
export FASTGEN_OUTPUT_ROOT="/data/chenqingzhan/fastgen_output"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/data/chenqingzhan/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"
BASHEOF
mkdir -p /data/chenqingzhan/fastgen_output
mkdir -p /data/chenqingzhan/.cache/huggingface
echo "Environment variables added to ~/.bashrc"

echo "=== SETUP COMPLETE ==="
