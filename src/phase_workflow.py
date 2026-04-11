# -*- coding: utf-8 -*-
"""
Phase-aware training workflow.

Trains a RAVE model on phase-organized audio data, then generates
pre-computed latent anchors from each phase for use in the streaming GUI.

Expected data layout:
    input_data/user_data/
        soft_rain/   (audio files for phase 1)
        rain/        (audio files for phase 2)
        storm/       (audio files for phase 3)
"""

import os

import torch

from src.preprocess import PreprocessDataset
from src.train import TrainModel
from src.export import ExportModel
from src.streaming.phase_control import generate_anchors_from_folders, save_phase_anchors


def discover_phase_folders(base_path):
    """Find subfolders in base_path, treating each as a phase.

    Returns list of (folder_name, full_path) sorted alphabetically.
    """
    if not os.path.isdir(base_path):
        return []

    folders = []
    for name in sorted(os.listdir(base_path)):
        full = os.path.join(base_path, name)
        if os.path.isdir(full):
            folders.append((name, full))

    return folders


def merge_phase_audio(phase_folders, merged_path):
    """Create symlinks or copies of all phase audio into a single folder.

    RAVE expects one flat folder of audio for preprocessing. This merges
    all phase subfolders into merged_path while preserving originals.

    Returns the merged folder path.
    """
    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
    os.makedirs(merged_path, exist_ok=True)

    count = 0
    for label, folder in phase_folders:
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() not in audio_exts:
                continue
            src = os.path.join(folder, fname)
            # Prefix with phase label to avoid name collisions
            dst_name = f"{label}__{fname}"
            dst = os.path.join(merged_path, dst_name)
            if not os.path.exists(dst):
                # Use hard link where possible, fall back to copy
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
            count += 1

    return count


def phase_train_workflow(
    audio_base_path,
    phase_labels=None,
    model_name="my_phase_model",
    config="v2_small",
    channels=1,
    lazy=False,
    max_db_size=10,
    val_every=1000,
    save_every=10000,
    max_steps=6000000,
    batch_size=8,
    streaming=True,
    sr=44100,
):
    """Full phase-aware training: preprocess -> train -> export -> generate anchors.

    Args:
        audio_base_path: Folder containing one subfolder per phase.
        phase_labels: Optional ordered list of subfolder names (phase order).
                      If None, subfolders are sorted alphabetically.
        model_name: Name for the RAVE model.
        config: RAVE architecture config.
        channels: Audio channels.
        lazy: Use lazy preprocessing.
        max_db_size: Max LMDB size in GB.
        val_every: Validation frequency.
        save_every: Checkpoint save frequency.
        max_steps: Max training steps.
        batch_size: Batch size.
        streaming: Enable streaming export.
        sr: Sample rate for anchor generation.

    Returns:
        Tuple of (exported_model_path, anchors_path) or (None, None) on failure.
    """
    print("=" * 60)
    print("  Phase-Aware Training Workflow")
    print("=" * 60)

    # --- Discover phases ---
    phase_folders = discover_phase_folders(audio_base_path)
    if len(phase_folders) < 2:
        print(f"\n[X] Need at least 2 phase subfolders in {audio_base_path}")
        print("    Create subfolders like: soft_rain/, rain/, storm/")
        print("    Each containing audio files for that phase.")
        return None, None

    # Apply custom ordering if provided
    if phase_labels:
        ordered = []
        for label in phase_labels:
            match = [(l, p) for l, p in phase_folders if l == label]
            if match:
                ordered.append(match[0])
            else:
                print(f"[!] Phase folder '{label}' not found, skipping")
        phase_folders = ordered if len(ordered) >= 2 else phase_folders

    print(f"\n[*] Found {len(phase_folders)} phases:")
    for i, (label, path) in enumerate(phase_folders):
        n_files = len([f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in {".wav", ".mp3", ".flac", ".ogg"}])
        print(f"    {i + 1}. {label} ({n_files} audio files)")

    # --- Step 1: Merge audio into single folder ---
    print(f"\n[Step 1/5] Merging phase audio into single training folder...")
    merged_path = os.path.join("input_data", f"_merged_{model_name}")
    n_merged = merge_phase_audio(phase_folders, merged_path)
    print(f"  [OK] {n_merged} files merged into {merged_path}")

    # --- Step 2: Preprocess ---
    print(f"\n[Step 2/5] Preprocessing merged dataset...")
    data_path = PreprocessDataset(merged_path, channels, lazy, max_db_size)

    # --- Step 3: Train ---
    print(f"\n[Step 3/5] Training RAVE model on all phases...")
    TrainModel(
        name=model_name,
        config=config,
        db_path=data_path,
        out_path="models/user_model/checkpoints",
        channels=channels,
        val_every=val_every,
        save_every=save_every,
        max_steps=max_steps,
        batch_size=batch_size,
    )

    # --- Step 4: Export ---
    print(f"\n[Step 4/5] Exporting model...")
    exported_path = ExportModel(streaming=streaming)

    if not exported_path or not os.path.exists(exported_path):
        print("[X] Export failed, cannot generate phase anchors.")
        return None, None

    # --- Step 5: Generate phase anchors ---
    print(f"\n[Step 5/5] Generating phase anchors from reference audio...")
    model = torch.jit.load(exported_path).eval()
    anchors = generate_anchors_from_folders(model, phase_folders, sr=sr)

    if len(anchors) < 2:
        print("[X] Could not generate enough phase anchors.")
        return exported_path, None

    anchors_path = os.path.splitext(exported_path)[0] + "_phases.json"
    save_phase_anchors(anchors, anchors_path)

    print("\n" + "=" * 60)
    print("  Phase-Aware Training Complete")
    print("=" * 60)
    print(f"\n  Model:   {exported_path}")
    print(f"  Anchors: {anchors_path}")
    print(f"  Phases:  {' -> '.join(a['label'] for a in anchors)}")
    print(f"\n  In the streaming GUI, load the model and the anchors")
    print(f"  will be detected automatically.")
    print("=" * 60)

    return exported_path, anchors_path


def generate_anchors_only(model_path, audio_base_path, phase_labels=None, sr=44100):
    """Generate phase anchors for an already-exported model.

    Useful when the model was trained with the normal workflow and you
    want to add phase control after the fact.

    Args:
        model_path: Path to exported .ts model.
        audio_base_path: Folder with phase subfolders.
        phase_labels: Optional ordered list of phase names.
        sr: Sample rate.

    Returns:
        Path to saved anchors file, or None.
    """
    print("\n--- Generate Phase Anchors ---")

    phase_folders = discover_phase_folders(audio_base_path)
    if len(phase_folders) < 2:
        print(f"[X] Need at least 2 phase subfolders in {audio_base_path}")
        return None

    if phase_labels:
        ordered = [(l, p) for l, p in phase_folders if l in phase_labels]
        if len(ordered) >= 2:
            phase_folders = sorted(ordered, key=lambda x: phase_labels.index(x[0]))

    print(f"[*] Loading model: {model_path}")
    model = torch.jit.load(model_path).eval()

    print(f"[*] Encoding {len(phase_folders)} phases...")
    anchors = generate_anchors_from_folders(model, phase_folders, sr=sr)

    if len(anchors) < 2:
        print("[X] Not enough phases encoded.")
        return None

    anchors_path = os.path.splitext(model_path)[0] + "_phases.json"
    save_phase_anchors(anchors, anchors_path)

    print(f"[OK] Phase anchors saved: {anchors_path}")
    print(f"     Phases: {' -> '.join(a['label'] for a in anchors)}")
    return anchors_path
