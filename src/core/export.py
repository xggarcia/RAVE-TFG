"""Export trained RAVE models to TorchScript."""

import glob
import os
import shutil
import subprocess


def _find_latest_run_dir(base_dir):
    """Return the newest immediate subfolder of `base_dir` that contains a .ckpt."""
    candidates = []
    for entry in os.listdir(base_dir):
        run_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        if glob.glob(os.path.join(run_dir, "checkpoints", "*.ckpt")):
            candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def ExportModel(run_path=None, output_path="models/user_model/exported_model", streaming=True):
    """Export a trained RAVE model to TorchScript."""
    os.makedirs(output_path, exist_ok=True)

    if run_path is None:
        checkpoints_dir = os.path.abspath("models/user_model/checkpoints")
        if not os.path.isdir(checkpoints_dir):
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
        run_path = _find_latest_run_dir(checkpoints_dir)
        if run_path is None:
            raise FileNotFoundError(f"No training runs with checkpoints found in {checkpoints_dir}")
        print(f"Auto-detected run: {run_path}")

    run_path = os.path.abspath(run_path)
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt files in {checkpoint_dir}. Train the model first.")

    cmd = ["rave", "export", "--run", run_path]
    if streaming:
        cmd.append("--streaming")

    print(f"Exporting model from: {run_path}")
    print(f"Checkpoints found: {checkpoint_files}")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, check=True)

    exported_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ts")]
    if exported_files:
        latest_export = max(
            (os.path.join(checkpoint_dir, f) for f in exported_files),
            key=os.path.getmtime,
        )
        dest_path = os.path.join(output_path, os.path.basename(latest_export))
        shutil.move(latest_export, dest_path)
        print(f"Model moved to: {dest_path}")

    print("Export completed!")
    return output_path
