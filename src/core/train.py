# -*- coding: utf-8 -*-
"""
Train RAVE models
"""
import os
import subprocess


def _detect_gpu_flag():
    # rave's own setup_gpu() uses GPUtil with a 5% VRAM threshold, which
    # excludes GPUs already used by the Windows desktop compositor. We pick
    # the first CUDA device ourselves so training always lands on the GPU
    # when one exists, and falls back to CPU (--gpu -1) otherwise.
    try:
        import torch
    except ImportError:
        return ["-1"]
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return ["0"]
    return ["-1"]


def find_latest_run(out_path, name):
    """Return the newest `<out_path>/<name>_<hash>` folder that contains
    at least one .ckpt, or None if no such run exists."""
    if not os.path.isdir(out_path):
        return None
    prefix = f"{name}_"
    candidates = []
    for entry in os.listdir(out_path):
        if not entry.startswith(prefix):
            continue
        run_dir = os.path.join(out_path, entry)
        if not os.path.isdir(run_dir):
            continue
        has_ckpt = False
        for root, _, files in os.walk(run_dir):
            if any(f.endswith(".ckpt") for f in files):
                has_ckpt = True
                break
        if has_ckpt:
            candidates.append((os.path.getmtime(run_dir), run_dir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def TrainModel(
    name="my_model",
    config="v2_small",
    extra_configs=None,
    db_path="preprocessed_data",
    out_path="models/user_model/checkpoints",
    channels=1,
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8,
    ckpt=None,
    overrides=None,
):
    """
    Train a RAVE model.

    Args:
        name: Model name
        config: Architecture configuration (v2_small, v2, v3, etc.)
        extra_configs: Additional RAVE configs to apply, e.g. ["noise", "causal"]
        db_path: Path to preprocessed dataset
        out_path: Output path for checkpoints
        channels: Number of audio channels
        val_every: Validation frequency (steps)
        save_every: Save checkpoint frequency (steps)
        max_steps: Maximum training steps
        batch_size: Batch size for training
        ckpt: Checkpoint to resume from. Accepts:
            - None: start a fresh run
            - str: explicit path to a .ckpt file or a run folder
            - True: auto-detect the latest run matching `name` under `out_path`

    Returns:
        Path to output directory
    """

    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    resolved_ckpt = None
    if ckpt is True:
        resolved_ckpt = find_latest_run(out_path, name)
        if resolved_ckpt is None:
            print("[resume] No previous run found - starting fresh.")
    elif isinstance(ckpt, str) and ckpt:
        resolved_ckpt = ckpt

    # Build command
    cmd = [
        "rave",
        "train",
        "--config", config,
        *[arg for c in (extra_configs or []) for arg in ("--config", c)],
        "--db_path", db_path,
        "--out_path", out_path,
        "--name", name,
        "--channels", str(channels),
        "--val_every", str(val_every),
        "--save_every", str(save_every),
        "--max_steps", str(max_steps),
        "--batch", str(batch_size),
    ]
    for gpu_id in _detect_gpu_flag():
        cmd.extend(["--gpu", gpu_id])
    for override in (overrides or []):
        cmd.extend(["--override", override])
    if resolved_ckpt:
        cmd.extend(["--ckpt", resolved_ckpt])

    print(f"Training model: {name}")
    print(f"Config: {config}")
    print(f"Dataset: {db_path}")
    print(f"Output: {out_path}")
    print(f"Checkpoint every: {val_every} steps")
    print(f"Save every: {save_every} steps")
    if resolved_ckpt:
        print(f"Resuming from: {resolved_ckpt}")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, check=True)

    print(f"Training completed!")
    return out_path
