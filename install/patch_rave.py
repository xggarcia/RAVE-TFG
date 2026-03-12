"""
Post-install patch for acids-rave compatibility with scipy >= 1.12.

scipy >= 1.12 moved `kaiser` from `scipy.signal` to `scipy.signal.windows`.
This script patches rave/pqmf.py so the import works on modern scipy.

Run this after installing acids-rave:
    python install/patch_rave.py
"""

import importlib.util
import os
import sys


def patch():
    # Find rave package without importing it (import may fail due to the very
    # bug we're patching). Use importlib to locate it instead.
    spec = importlib.util.find_spec("rave")
    if spec is None or spec.origin is None:
        print("[patch] acids-rave not installed, skipping.")
        return

    rave_dir = os.path.dirname(spec.origin)
    pqmf_path = os.path.join(rave_dir, "pqmf.py")
    if not os.path.exists(pqmf_path):
        print(f"[patch] pqmf.py not found at {pqmf_path}, skipping.")
        return

    with open(pqmf_path, "r", encoding="utf-8") as f:
        content = f.read()

    old_import = "from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord"
    new_import = (
        "from scipy.signal import firwin, kaiser_beta, kaiserord\n"
        "from scipy.signal.windows import kaiser"
    )

    if old_import not in content:
        print("[patch] pqmf.py already patched or import line differs, skipping.")
        return

    content = content.replace(old_import, new_import)

    with open(pqmf_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[patch] Patched {pqmf_path} for scipy >= 1.12 compatibility.")


if __name__ == "__main__":
    patch()
