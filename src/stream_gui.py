# -*- coding: utf-8 -*-
"""Compatibility launcher for the modular streaming GUI package."""

import sys
import tkinter as tk


def launch_gui():
    """Launch the GUI application."""
    import torch
    from src.streaming_gui import RAVEStreamGUI

    torch.set_grad_enabled(False)

    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")

    root = tk.Tk()
    app = RAVEStreamGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
