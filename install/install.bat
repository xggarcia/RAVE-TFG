@echo off
REM ============================================
REM RAVE-TFG Installer (Windows)
REM ============================================
REM Installs all dependencies, working around
REM acids-rave's incompatible scipy==1.10.0 pin,
REM and installs PyTorch with CUDA when an NVIDIA
REM GPU is available.
REM ============================================

echo [1/5] Detecting GPU...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    nvidia-smi >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set TORCH_INDEX=https://download.pytorch.org/whl/cu121
        echo   NVIDIA GPU detected - installing CUDA 12.1 PyTorch wheels.
    ) else (
        set TORCH_INDEX=https://download.pytorch.org/whl/cpu
        echo   nvidia-smi present but no GPU responded - falling back to CPU wheels.
    )
) else (
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    echo   No NVIDIA GPU detected - installing CPU-only PyTorch wheels.
)

echo [2/5] Installing PyTorch (torch, torchaudio)...
pip install --index-url %TORCH_INDEX% "torch>=2.0.0" "torchaudio>=2.0.0"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install PyTorch from %TORCH_INDEX%.
    exit /b 1
)

echo [3/5] Installing core dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install core dependencies.
    exit /b 1
)

echo [4/5] Installing acids-rave and acids-msprior (without pinned deps)...
pip install --no-deps "acids-rave>=2.3.0"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install acids-rave.
    exit /b 1
)
pip install --no-deps "acids-msprior>=0.1.0"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install acids-msprior.
    exit /b 1
)

echo [5/5] Patching acids-rave for scipy compatibility...
python install\patch_rave.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Patch failed, but installation may still work.
)

echo.
echo ============================================
echo   Installation complete!
echo   Run: python main.py
echo ============================================
