#!/usr/bin/env bash
# ============================================
# RAVE-TFG Installer (Linux / macOS)
# ============================================
# Installs all dependencies, working around
# acids-rave's incompatible scipy==1.10.0 pin,
# and installs PyTorch with CUDA when an NVIDIA
# GPU is available.
# ============================================
set -e

echo "[1/6] Detecting GPU..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    # cu124 matches the RunPod PyTorch 2.4.0 base image and any recent driver.
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    echo "  NVIDIA GPU detected - installing CUDA 12.8 PyTorch wheels."
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "  No NVIDIA GPU detected - installing CPU-only PyTorch wheels."
fi

echo "[2/6] Checking for ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "  ffmpeg not found - installing..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get install -y ffmpeg
    elif command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    else
        echo "ERROR: Cannot install ffmpeg automatically. Install it manually and ensure it is on your PATH."
        exit 1
    fi
    echo "  ffmpeg installed."
else
    echo "  ffmpeg already installed."
fi

echo "[3/6] Installing PyTorch (torch, torchaudio)..."
# Skip if torch 2.8+ already installed (e.g. RunPod base image).
# Use --no-deps on requirements.txt later to avoid pip downgrading torch.
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [[ "$TORCH_VERSION" == 2.8* ]]; then
    echo "  torch $TORCH_VERSION already installed, skipping."
else
    pip install --index-url "$TORCH_INDEX" "torch==2.8.0" "torchaudio==2.8.0"
fi

echo "[4/6] Installing core dependencies..."
# --no-deps prevents pip from resolving torch again and downgrading it.
pip install --no-deps -r requirements.txt
# Install deps that are safe to resolve normally (no torch conflict).
pip install scipy pytorch-lightning absl-py cached-conv einops Flask gin-config \
    GPUtil nn-tilde PyYAML scikit-learn tensorboard tqdm udls lmdb pathos \
    librosa soundfile sounddevice customtkinter ipython numpy requests

echo "[5/6] Installing acids-rave and acids-msprior (without pinned deps)..."
pip install --no-deps "acids-rave>=2.3.0"
pip install --no-deps "acids-msprior>=0.1.0"

echo "[6/6] Patching acids-rave for scipy compatibility..."
python install/patch_rave.py

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Run: python main.py"
echo "============================================"
