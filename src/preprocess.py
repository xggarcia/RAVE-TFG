# -*- coding: utf-8 -*-
"""
Preprocess audio datasets for RAVE training
"""
import os
import subprocess


def PreprocessDataset(audio_path, channels=1, lazy=False, max_db_size=10):
    """
    Preprocess the dataset.

    Args:
        audio_path: Path to folder containing audio files
        channels: Number of audio channels (1=mono, 2=stereo)
        lazy: If True, store only file paths and decode on-the-fly during
            training (slower, especially on Windows). Defaults to False so
            the dataset is fully decoded to int16 upfront for fast training.
        max_db_size: Maximum database size in GB

    Returns:
        Path to preprocessed data directory
    """
    # Use absolute paths to avoid RAVE bug with relative paths
    audio_path = os.path.abspath(audio_path)
    data_path = os.path.abspath("preprocessed_data")
    
    # Create directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    cmd = [
        "rave",
        "preprocess",
        "--input_path", audio_path,
        "--output_path", data_path,
        "--channels", str(channels),
        "--max_db_size", str(max_db_size)
    ]
    
    if lazy:
        cmd.append("--lazy")
    
    print(f"Preprocessing: {audio_path} -> {data_path}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Use input="y\n" to automatically confirm the lazy dataset prompt on Windows/macOS
    if lazy:
        subprocess.run(cmd, check=True, input="y\n", text=True)
    else:
        subprocess.run(cmd, check=True)

    print(f"Preprocessing completed!")
    return data_path
