import os
from IPython.display import Audio, display
import librosa as li
import soundfile as sf
import numpy as np
import requests
import torch
import subprocess

torch.set_grad_enabled(False)


def UseModel(model_path = "models/demo/demo_model.ts", audio_path='data/demo/audio1.wav', sr=44100, random = True, output_name = "Output_audio", duration=30):
    """
    Generate audio from random latent vectors using a RAVE model.
    
    Args:
        model_path: Path to the RAVE model (.ts file)
        audio_path: Path to a sample audio file (used to get latent dimensions)
        sr: Sample rate for audio
        random: Whether to generate random latent vectors or use the latent vector of the input audio
        duration: Duration in seconds for random generation (default: 30)
    """
    model = torch.jit.load(model_path).eval()

    # Load a sample audio to get the latent shape
    x, sr = li.load(audio_path, sr=sr)
    x = torch.from_numpy(x).reshape(1, 1, -1)
    
    # Encode to get latent representation and its shape
    z = model.encode(x)
    
    # Generate audio from random numbers
    if random:
        # Calculate latent size for desired duration (30s default)
        # RAVE typically has a compression ratio, estimate from the loaded audio
        latent_per_second = z.shape[-1] / (len(x.reshape(-1)) / sr)
        target_latent_length = int(latent_per_second * duration)
        z_shape = (z.shape[0], z.shape[1], target_latent_length)
        z_random = torch.from_numpy(np.random.randn(*z_shape).astype(np.float32))
    else:
        z_random = z
    x_hat = model.decode(z_random).numpy().reshape(-1)
    
    # Save and display the generated audio

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"{output_name}.wav")
    sf.write(output_path, x_hat, sr)
    print(f"Audio generated and saved to: {output_path}")
    return output_path



def PreprocessDataset(audio_path, channels=1, lazy=True, max_db_size=10):
    """
    Preprocess the dataset.
    
    Args:
        audio_path: Path to folder containing audio files
        channels: Number of audio channels (1 for mono, 2 for stereo)
        lazy: If True, uses lazy loading (trains directly on raw files)
        max_db_size: Maximum database size in GB (default 10, LMDB pre-allocates this on Windows)
    
    Returns:
        data_path: Path to the preprocessed dataset
    """
    # Use absolute path to avoid RAVE bug with relative paths
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
    if(lazy):
        subprocess.run(cmd, check=True, input="y\n", text=True)

    print(f"Preprocessing completed!")
    return data_path


def TrainModel(
    name="my_model",
    config="v2_small",
    db_path="preprocessed_data",
    out_path="models/checkpoints",
    channels=1,
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8
):
    """
    Train a RAVE model.
    
    Args:
        name: Name for the model
        config: Architecture configuration (v1, v2, v2_small, discrete, etc.)
        db_path: Path to preprocessed dataset
        out_path: Output path for model checkpoints
        channels: Number of audio channels (1 for mono, 2 for stereo)
        val_every: Checkpoint/validate every N steps (default: 1000 for frequent saves)
        save_every: Save model every N steps (default: 10000)
        max_steps: Maximum training steps (default: 6000000)
        batch_size: Batch size (default: 8)
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


def ExportModel(run_path=None, output_path="models/trained", streaming=True):
    """
    Export a trained RAVE model to a TorchScript file.
    
    Args:
        run_path: Path to the training run folder (e.g., models/checkpoints/model_name/version_X)
                  If None, will look for the latest run in models/checkpoints
        output_path: Path where to save the exported .ts file (default: models/trained)
        streaming: If True, enables cached convolutions for realtime processing.
                   Required to avoid clicking artifacts in Max/MSP and other realtime environments.
    
    Returns:
        output_path: Path to the exported model
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # If no run path provided, try to find the latest one
    if run_path is None:
        checkpoints_dir = os.path.abspath("models/checkpoints")
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
                raise FileNotFoundError("No training runs found in models/checkpoints")
        else:
            raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Convert to absolute path (RAVE has issues with relative paths)
    run_path = os.path.abspath(run_path)
    
    # Check if checkpoint files exist in the run directory
    checkpoint_files = [f for f in os.listdir(run_path) if f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files (.ckpt) found in {run_path}.\n"
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
    print(f"Checkpoint found: {checkpoint_files[-1]}")
    print(f"Streaming mode: {streaming}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, check=True)
    
    print(f"Export completed!")
    return output_path


def train_workflow(
    audio_path,
    model_name="my_model",
    channels=1,
    lazy=True,
    max_db_size=10,
    config="v2_small",
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8,
    streaming=True
):
    """
    Complete training workflow: preprocess → train → export
    
    Args:
        audio_path: Path to folder containing audio files
        model_name: Name for the model
        channels: Number of audio channels (1 for mono, 2 for stereo)
        lazy: If True, uses lazy loading for preprocessing
        max_db_size: Maximum database size in GB
        config: Architecture configuration (v1, v2, v2_small, discrete, etc.)
        val_every: Checkpoint every N steps
        save_every: Save model every N steps
        max_steps: Maximum training steps
        batch_size: Batch size for training
        streaming: If True, export with streaming mode for realtime use
    
    Returns:
        exported_path: Path to the exported model
    """
    print("=" * 50)
    print("RAVE Training Workflow")
    print("=" * 50)
    
    # Step 1: Preprocess
    print("\n[Step 1/3] Preprocessing dataset...")
    data_path = PreprocessDataset(audio_path, channels, lazy, max_db_size)
    
    # Step 2: Train
    print("\n[Step 2/3] Training model...")
    out_path = TrainModel(
        name=model_name,
        config=config,
        db_path=data_path,
        out_path="models/checkpoints",
        channels=channels,
        val_every=val_every,
        save_every=save_every,
        max_steps=max_steps,
        batch_size=batch_size
    )
    
    # Step 3: Export
    print("\n[Step 3/3] Exporting model...")
    exported_path = ExportModel(streaming=streaming)
    
    print("\n" + "=" * 50)
    print("Workflow completed!")
    print(f"Exported model saved to: {exported_path}")
    print("=" * 50)
    
    return exported_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAVE Training and Inference CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess audio dataset")
    preprocess_parser.add_argument("audio_path", help="Path to folder containing audio files")
    preprocess_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    preprocess_parser.add_argument("--no-lazy", action="store_true", help="Disable lazy loading")
    preprocess_parser.add_argument("--max-db-size", type=int, default=10, help="Max database size in GB (default: 10)")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a RAVE model")
    train_parser.add_argument("--name", default="my_model", help="Model name (default: my_model)")
    train_parser.add_argument("--config", default="v2_small", help="Architecture config (default: v2_small)")
    train_parser.add_argument("--db-path", default="preprocessed_data", help="Path to preprocessed dataset")
    train_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    train_parser.add_argument("--val-every", type=int, default=1000, help="Checkpoint every N steps (default: 1000)")
    train_parser.add_argument("--save-every", type=int, default=10000, help="Save every N steps (default: 10000)")
    train_parser.add_argument("--max-steps", type=int, default=6000000, help="Max training steps (default: 6000000)")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export trained model to TorchScript")
    export_parser.add_argument("--run-path", help="Path to training run folder (auto-detects if not provided)")
    export_parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode")
    
    # Workflow command (full pipeline)
    workflow_parser = subparsers.add_parser("workflow", help="Run complete workflow: preprocess → train → export")
    workflow_parser.add_argument("audio_path", help="Path to folder containing audio files")
    workflow_parser.add_argument("--name", default="my_model", help="Model name (default: my_model)")
    workflow_parser.add_argument("--config", default="v2_small", help="Architecture config (default: v2_small)")
    workflow_parser.add_argument("--channels", type=int, default=1, help="Number of audio channels (default: 1)")
    workflow_parser.add_argument("--val-every", type=int, default=1000, help="Checkpoint every N steps (default: 1000)")
    workflow_parser.add_argument("--max-steps", type=int, default=6000000, help="Max training steps (default: 6000000)")
    
    # Generate command (use model)
    generate_parser = subparsers.add_parser("generate", help="Generate audio using a trained model")
    generate_parser.add_argument("--model", default="models/demo/demo_model.ts", help="Path to model file")
    generate_parser.add_argument("--audio", default="input_data/demo_data/audio1.wav", help="Path to sample audio file")
    generate_parser.add_argument("--output", default="generated", help="Output filename (without extension)")
    generate_parser.add_argument("--no-random", action="store_true", help="Use input audio's latent instead of random")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        PreprocessDataset(
            audio_path=args.audio_path,
            channels=args.channels,
            lazy=not args.no_lazy,
            max_db_size=args.max_db_size
        )
    
    elif args.command == "train":
        TrainModel(
            name=args.name,
            config=args.config,
            db_path=args.db_path,
            channels=args.channels,
            val_every=args.val_every,
            save_every=args.save_every,
            max_steps=args.max_steps,
            batch_size=args.batch_size
        )
    
    elif args.command == "export":
        ExportModel(
            run_path=args.run_path,
            streaming=not args.no_streaming
        )
    
    elif args.command == "workflow":
        train_workflow(
            audio_path=args.audio_path,
            model_name=args.name,
            config=args.config,
            channels=args.channels,
            val_every=args.val_every,
            max_steps=args.max_steps
        )
    
    elif args.command == "generate":
        UseModel(
            model_path=args.model,
            audio_path=args.audio,
            output_name=args.output,
            random=not args.no_random
        )
    