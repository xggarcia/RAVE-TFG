# -*- coding: utf-8 -*-
"""
Export trained RAVE models to TorchScript
"""
import os
import shutil
import subprocess


def ExportModel(run_path=None, output_path="models/user_model/exported_model", streaming=True):
    """
    Export a trained RAVE model to a TorchScript file.
    
    Args:
        run_path: Path to training run folder (auto-detects if None)
        output_path: Output directory for exported model
        streaming: Enable streaming mode for real-time applications
    
    Returns:
        Path to output directory
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # If no run path provided, try to find the latest one
    if run_path is None:
        checkpoints_dir = os.path.abspath("models/user_model/checkpoints")
        if os.path.exists(checkpoints_dir):
            # Find the most recent run
            runs = []
            for model_dir in os.listdir(checkpoints_dir):
                model_path = os.path.join(checkpoints_dir, model_dir)
                if os.path.isdir(model_path):
                    for version_dir in os.listdir(model_path):
                        version_path = os.path.join(model_path, version_dir)
                        if os.path.isdir(version_path) and version_dir.startswith("version_"):
                            runs.append(version_path)
            
            if runs:
                # Sort by modification time, get the latest
                run_path = max(runs, key=os.path.getmtime)
                print(f"Auto-detected run: {run_path}")
            else:
                raise FileNotFoundError("No training runs found in models/user_model/checkpoints")
        else:
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Convert to absolute path (RAVE has issues with relative paths)
    run_path = os.path.abspath(run_path)
    
    # Check if checkpoint files exist in the run directory
    # PyTorch Lightning saves checkpoints in a 'checkpoints' subfolder
    checkpoint_dir = os.path.join(run_path, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    else:
        # Fallback: check directly in run_path (for older RAVE versions or manual setups)
        checkpoint_files = [f for f in os.listdir(run_path) if f.endswith('.ckpt')]
    
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files (.ckpt) found in {run_path} or {run_path}/checkpoints.\n"
            "You need to train the model first. Run TrainModel() and let it train "
            "for at least a few epochs until a checkpoint is saved."
        )
    
    # Build command
    cmd = [
        "rave",
        "export",
        "--run", run_path,
    ]
    
    # Add streaming flag for realtime processing (avoids clicking in Max/MSP)
    if streaming:
        cmd.append("--streaming")
    
    print(f"Exporting model from: {run_path}")
    print(f"Checkpoints found: {checkpoint_files}")
    print(f"Streaming mode: {streaming}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, check=True)
    
    # Find the exported .ts file and move it to the output directory
    # RAVE exports to the checkpoints directory by default
    export_search_dir = checkpoint_dir if os.path.exists(checkpoint_dir) else run_path
    exported_files = [f for f in os.listdir(export_search_dir) if f.endswith('.ts')]
    
    if exported_files:
        # Get the most recently created .ts file
        exported_files_full = [os.path.join(export_search_dir, f) for f in exported_files]
        latest_export = max(exported_files_full, key=os.path.getmtime)
        
        # Move to output directory
        dest_path = os.path.join(output_path, os.path.basename(latest_export))
        shutil.move(latest_export, dest_path)
        print(f"Model moved to: {dest_path}")
    
    print(f"Export completed!")
    return output_path
