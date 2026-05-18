@echo off
REM RAVE-TFG installer (Windows). Pre-requisites: Python 3.10+ and ffmpeg on PATH.
pip install --no-deps "acids-rave>=2.3.0" "acids-msprior>=0.1.0"
pip install -e .
python install\patch_rave.py
echo.
echo Installation complete. Run: python main.py
