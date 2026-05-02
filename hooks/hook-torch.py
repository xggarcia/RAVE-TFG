"""
PyInstaller hook for torch — inference-only bundle for RAVE-TFG.

Rule: NEVER exclude a submodule that is re-exported in any torch package's
__init__.py — those imports are unconditional and will raise
"partially initialized module" errors in the frozen app.

torch/utils/__init__.py re-exports: backcompat, collect_env, data,
deterministic, hooks — all must remain in the bundle.
"""
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
    logger,
)
from PyInstaller import compat

# pyz+py mode lets PyInstaller keep .py sources alongside compiled .pyc —
# required for torch >= 2.0 because it inspects its own source at runtime.
module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

# Collect all submodules, then strip training/dev-only ones.
# Do NOT exclude anything re-exported by a torch __init__.py.
hiddenimports = collect_submodules("torch")

_SAFE_TO_EXCLUDE = {
    # Compiler / graph transforms — not used for JIT inference
    "torch._dynamo",
    "torch._inductor",
    "torch._functorch",
    "torch._higher_order_ops",
    "torch._export",
    "torch._library",
    "torch._sources",
    # Distributed training
    "torch.distributed",
    # FX graph manipulation
    "torch.fx",
    # ONNX export
    "torch.onnx",
    # Profiling (cupti DLLs alone are ~200 MB)
    "torch.profiler",
    # TensorBoard integration
    "torch.utils.tensorboard",
    # Heavy dev utilities (safe — NOT re-exported by torch.utils.__init__)
    "torch.utils.benchmark",
    "torch.utils.bottleneck",
    "torch.utils.cpp_extension",
    "torch.utils.cpp_tracker",
    "torch.utils.triton",
    # Mobile / edge backends
    "torch.backends._coreml",
    "torch.backends._nnapi",
    "torch.backends.mps",
    "torch.backends.openvino",
    "torch.backends.executorch",
    "torch.backends.xnnpack",
    # Lazy tensors
    "torch.lazy",
    # AO quantization stack
    "torch.ao",
    # Testing infrastructure (~200 MB Python files)
    "torch.testing",
    # XLA
    "torch_xla",
}

hiddenimports = [
    m for m in hiddenimports
    if not any(m == exc or m.startswith(exc + ".") for exc in _SAFE_TO_EXCLUDE)
]

# Collect CUDA / cuDNN / cuBLAS DLLs (needed for CUDA inference)
datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
    ],
)

if compat.is_win:
    try:
        binaries = collect_dynamic_libs("torch")
    except Exception:
        logger.warning("hook-torch: failed to collect torch DLLs", exc_info=True)
        binaries = []
