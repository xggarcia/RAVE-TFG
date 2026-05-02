import os
import sys
from pathlib import Path

_REPO_ROOT: Path | None = None


def get_repo_root() -> Path:
    """Return the repo root path, correctly handling PyInstaller onefile/onedir."""
    global _REPO_ROOT
    if _REPO_ROOT is not None:
        return _REPO_ROOT

    # PyInstaller: sys._MEIPASS points to the extracted temp dir (onefile)
    # or the bundled dir. For onedir, we look relative to the executable.
    if getattr(sys, "frozen", False):
        # Running as compiled executable
        exe_dir = Path(sys.executable).parent
        # For onedir: exe_dir/RAVE-TFG.exe, repo root is exe_dir
        # Check if it contains the expected structure
        if (exe_dir / "models").exists() or (exe_dir / "app").exists():
            _REPO_ROOT = exe_dir
        else:
            # Fall back to exe_dir parent
            _REPO_ROOT = exe_dir
    else:
        # Running from source: app/__init__.py is one level below repo root
        _REPO_ROOT = Path(__file__).parent.parent

    return _REPO_ROOT


def get_app_resources() -> Path:
    """Return path to bundled app resources (UI files, etc.)."""
    if getattr(sys, "frozen", False):
        # PyInstaller: check _MEIPASS first (onefile extraction dir)
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS) / "app"
        # For onedir, resources are relative to exe
        return get_repo_root() / "app"
    return Path(__file__).parent
