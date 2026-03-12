@echo off
REM ============================================
REM RAVE-TFG Installer (Windows)
REM ============================================
REM Installs all dependencies, working around
REM acids-rave's incompatible scipy==1.10.0 pin.
REM ============================================

echo [1/4] Installing core dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install core dependencies.
    exit /b 1
)

echo [2/4] Installing acids-rave (without pinned deps)...
pip install --no-deps "acids-rave>=2.3.0"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install acids-rave.
    exit /b 1
)

echo [3/4] Installing acids-msprior (without pinned deps)...
pip install --no-deps "acids-msprior>=0.1.0"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install acids-msprior.
    exit /b 1
)

echo [4/4] Patching acids-rave for scipy compatibility...
python install\patch_rave.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Patch failed, but installation may still work.
)

echo.
echo ============================================
echo   Installation complete!
echo   Run: python main.py
echo ============================================
