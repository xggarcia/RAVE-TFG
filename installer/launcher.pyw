# Entry point for the installed app.
# .pyw extension → pythonw.exe runs it with no console window.
# Located at {install_dir}\launcher.pyw, so the repo root is its parent dir.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.__main__ import main

main()
