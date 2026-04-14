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

echo "[1/5] Detecting GPU..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    # cu124 matches the RunPod PyTorch 2.4.0 base image and any recent driver.
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "  NVIDIA GPU detected - installing CUDA 12.4 PyTorch wheels."
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "  No NVIDIA GPU detected - installing CPU-only PyTorch wheels."
fi

echo "[2/5] Installing PyTorch (torch, torchaudio)..."
# --force-reinstall so a preinstalled torch (e.g. the one shipped by the
# RunPod PyTorch base image) cannot stay behind with a different CUDA build
# than torchaudio and trigger a version-mismatch error at import time.
pip install --force-reinstall --index-url "$TORCH_INDEX" "torch>=2.0.0" "torchaudio>=2.0.0"

echo "[3/5] Installing core dependencies..."
pip install -r requirements.txt

echo "[4/5] Installing acids-rave and acids-msprior (without pinned deps)..."
pip install --no-deps "acids-rave>=2.3.0"
pip install --no-deps "acids-msprior>=0.1.0"

echo "[5/5] Patching acids-rave for scipy compatibility..."
python install/patch_rave.py

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Run: python main.py"
echo "============================================"
