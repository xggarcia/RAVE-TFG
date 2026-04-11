# -*- coding: utf-8 -*-
"""
Complete training workflow: preprocess → train → export
"""
from src.preprocess import PreprocessDataset
from src.train import TrainModel
from src.export import ExportModel


def train_workflow(
    audio_path,
    model_name="my_model",
    channels=1,
    lazy=False,
    max_db_size=10,
    config="v2_small",
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8,
    streaming=True,
    ckpt=None,
):
    """
    Complete training workflow: preprocess → train → export
    
    Args:
        audio_path: Path to folder containing audio files
        model_name: Name for the model
        channels: Number of audio channels
        lazy: Use lazy loading for preprocessing
        max_db_size: Max database size in GB
        config: Architecture configuration
        val_every: Validation frequency
        save_every: Save frequency
        max_steps: Maximum training steps
        batch_size: Batch size
        streaming: Enable streaming mode for export
    
    Returns:
        Path to exported model
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
        out_path="models/user_model/checkpoints",
        channels=channels,
        val_every=val_every,
        save_every=save_every,
        max_steps=max_steps,
        batch_size=batch_size,
        ckpt=ckpt,
    )
    
    # Step 3: Export
    print("\n[Step 3/3] Exporting model...")
    exported_path = ExportModel(streaming=streaming)
    
    print("\n" + "=" * 50)
    print("Workflow completed!")
    print(f"Exported model saved to: {exported_path}")
    print("=" * 50)
    
    return exported_path
