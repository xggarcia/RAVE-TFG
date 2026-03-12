#!/usr/bin/env bash
# ============================================
# RAVE-TFG Installer (Linux / macOS)
# ============================================
# Installs all dependencies, working around
# acids-rave's incompatible scipy==1.10.0 pin.
# ============================================
set -e

echo "[1/4] Installing core dependencies..."
pip install -r requirements.txt

echo "[2/4] Installing acids-rave (without pinned deps)..."
pip install --no-deps "acids-rave>=2.3.0"

echo "[3/4] Installing acids-msprior (without pinned deps)..."
pip install --no-deps "acids-msprior>=0.1.0"

echo "[4/4] Patching acids-rave for scipy compatibility..."
python install/patch_rave.py

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Run: python main.py"
echo "============================================"
