"""Train RAVE models by invoking the `rave train` CLI."""

import glob
import os
import subprocess


def find_latest_run(out_path, name):
    """Return the newest `<out_path>/<name>_<hash>` folder that contains a .ckpt, or None."""
    if not os.path.isdir(out_path):
        return None

    runs_with_ckpt = []
    for entry in os.listdir(out_path):
        if not entry.startswith(f"{name}_"):
            continue
        run_dir = os.path.join(out_path, entry)
        if not os.path.isdir(run_dir):
            continue
        if glob.glob(os.path.join(run_dir, "**", "*.ckpt"), recursive=True):
            runs_with_ckpt.append(run_dir)

    if not runs_with_ckpt:
        return None
    return max(runs_with_ckpt, key=os.path.getmtime)


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
    """Train a RAVE model.

    `ckpt` can be:
        - None: start fresh
        - str: explicit path to a .ckpt file or run folder
        - True: auto-detect the latest matching run under `out_path`
    """
    os.makedirs(out_path, exist_ok=True)

    resolved_ckpt = None
    if ckpt is True:
        resolved_ckpt = find_latest_run(out_path, name)
        if resolved_ckpt is None:
            print("[resume] No previous run found - starting fresh.")
    elif isinstance(ckpt, str) and ckpt:
        resolved_ckpt = ckpt

    cmd = [
        "rave", "train",
        "--config", config,
    ]
    for c in extra_configs or []:
        cmd.extend(["--config", c])
    cmd.extend([
        "--db_path", db_path,
        "--out_path", out_path,
        "--name", name,
        "--channels", str(channels),
        "--val_every", str(val_every),
        "--save_every", str(save_every),
        "--max_steps", str(max_steps),
        "--batch", str(batch_size),
        "--gpu", "0",
    ])
    for override in overrides or []:
        cmd.extend(["--override", override])
    if resolved_ckpt:
        cmd.extend(["--ckpt", resolved_ckpt])

    print(f"Training model: {name}")
    print(f"Config: {config}")
    print(f"Dataset: {db_path}")
    print(f"Output: {out_path}")
    if resolved_ckpt:
        print(f"Resuming from: {resolved_ckpt}")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, check=True)

    print("Training completed!")
    return out_path
