# -*- coding: utf-8 -*-
"""
Train RAVE models
"""
import os
import subprocess


def TrainModel(
    name="my_model",
    config="v2_small",
    db_path="preprocessed_data",
    out_path="models/user_model/checkpoints",
    channels=1,
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8
):
    """
    Train a RAVE model.
    
    Args:
        name: Model name
        config: Architecture configuration (v2_small, v2, v3, etc.)
        db_path: Path to preprocessed dataset
        out_path: Output path for checkpoints
        channels: Number of audio channels
        val_every: Validation frequency (steps)
        save_every: Save checkpoint frequency (steps)
        max_steps: Maximum training steps
        batch_size: Batch size for training
    
    Returns:
        Path to output directory
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    
    # Build command
    cmd = [
        "rave",
        "train",
        "--config", config,
        "--db_path", db_path,
        "--out_path", out_path,
        "--name", name,
        "--channels", str(channels),
        "--val_every", str(val_every),
        "--save_every", str(save_every),
        "--max_steps", str(max_steps),
        "--batch", str(batch_size)
    ]
    
    print(f"Training model: {name}")
    print(f"Config: {config}")
    print(f"Dataset: {db_path}")
    print(f"Output: {out_path}")
    print(f"Checkpoint every: {val_every} steps")
    print(f"Save every: {save_every} steps")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, check=True)
    
    print(f"Training completed!")
    return out_path
