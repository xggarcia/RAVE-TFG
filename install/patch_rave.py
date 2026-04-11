"""
Post-install patch for acids-rave compatibility with modern scipy and for
running preprocessing safely on Windows.

Fixes applied:

1. `rave/pqmf.py`: `kaiser` was moved from `scipy.signal` to
   `scipy.signal.windows` in scipy >= 1.12.
2. `rave/pqmf.py`: `scipy.signal.firwin` removed the deprecated `nyq`
   keyword argument (replaced by `fs=2*nyq`) in scipy >= 1.12.
3. `scripts/preprocess.py`: `multiprocessing.Pool()` spawned one worker per
   CPU core. On Windows each worker imports CUDA torch, which reserves
   several GB of virtual address space; with many cores this blows past the
   page file limit and crashes with `WinError 1455`. We cap the worker
   count to something sensible (env var `RAVE_PREPROCESS_WORKERS`, default 4)
   since the work is I/O-bound on ffmpeg/ffprobe anyway.

Run this after installing acids-rave:
    python install/patch_rave.py
"""

import importlib.util
import os


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def patch_pqmf(rave_dir):
    pqmf_path = os.path.join(rave_dir, "pqmf.py")
    if not os.path.exists(pqmf_path):
        print(f"[patch] pqmf.py not found at {pqmf_path}, skipping.")
        return

    content = _read(pqmf_path)
    original = content
    changes = []

    # Fix 1: kaiser import location (scipy >= 1.12).
    old_import = "from scipy.signal import firwin, kaiser, kaiser_beta, kaiserord"
    new_import = (
        "from scipy.signal import firwin, kaiser_beta, kaiserord\n"
        "from scipy.signal.windows import kaiser"
    )
    if old_import in content:
        content = content.replace(old_import, new_import)
        changes.append("kaiser import")

    # Fix 2: firwin no longer accepts `nyq` (removed in scipy >= 1.12).
    # The only call site uses `nyq=np.pi`, which is equivalent to `fs=2*np.pi`.
    old_firwin = "firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)"
    new_firwin = "firwin(N, wc, window=('kaiser', beta), scale=False, fs=2*np.pi)"
    if old_firwin in content:
        content = content.replace(old_firwin, new_firwin)
        changes.append("firwin nyq->fs")

    if content == original:
        print("[patch] pqmf.py already patched or source differs, skipping.")
        return

    _write(pqmf_path, content)
    print(f"[patch] Patched {pqmf_path} ({', '.join(changes)}).")


def patch_preprocess():
    # scripts/preprocess.py lives in the `scripts` package installed alongside
    # rave (acids-rave ships both).
    spec = importlib.util.find_spec("scripts.preprocess")
    if spec is None or spec.origin is None:
        print("[patch] scripts.preprocess not found, skipping worker cap.")
        return

    preprocess_path = spec.origin
    content = _read(preprocess_path)

    old_pool = "    pool = multiprocessing.Pool()"
    new_pool = (
        "    _rave_max_workers = int(os.environ.get('RAVE_PREPROCESS_WORKERS', '4'))\n"
        "    _rave_cpu_count = os.cpu_count() or 1\n"
        "    pool = multiprocessing.Pool(processes=max(1, min(_rave_cpu_count, _rave_max_workers)))"
    )

    if old_pool not in content:
        print("[patch] preprocess.py Pool() line not found or already patched, skipping.")
        return

    content = content.replace(old_pool, new_pool)
    _write(preprocess_path, content)
    print(f"[patch] Patched {preprocess_path} (worker cap).")


def patch():
    # Find rave package without importing it (import may fail due to the very
    # bug we're patching). Use importlib to locate it instead.
    spec = importlib.util.find_spec("rave")
    if spec is None or spec.origin is None:
        print("[patch] acids-rave not installed, skipping.")
        return

    rave_dir = os.path.dirname(spec.origin)
    patch_pqmf(rave_dir)
    patch_preprocess()


if __name__ == "__main__":
    patch()
