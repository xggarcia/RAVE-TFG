"""Phase workflow helpers called by the desktop UI workers."""
import json
import os

import torch


def generate_anchors_only(model_path, base_folder, phase_labels=None, output_dir=None):
    """Generate phase latent anchors + PCA scatter from organized audio folders.

    Args:
        model_path: Path to a TorchScript RAVE model (.ts).
        base_folder: Root folder whose immediate subdirectories are phase folders.
        phase_labels: Ordered list of phase names. If None, subdirs are sorted alphabetically.
        output_dir: Where to save the output files. Defaults to dirname(model_path).

    Returns:
        Path to the saved phase bundle JSON.
    """
    from src.streaming.phase_control import (
        compute_pca_scatter,
        generate_anchors_from_folders,
        save_phase_bundle,
    )

    subdirs = [
        d for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d))
    ]

    if phase_labels:
        subdirs_map = {d: os.path.join(base_folder, d) for d in subdirs}
        phase_dirs = [(lbl, subdirs_map[lbl]) for lbl in phase_labels if lbl in subdirs_map]
    else:
        subdirs.sort()
        phase_dirs = [(d, os.path.join(base_folder, d)) for d in subdirs]

    if not phase_dirs:
        raise ValueError(f"No phase subfolders found in {base_folder}")

    print(f"Phases: {[lbl for lbl, _ in phase_dirs]}")

    print(f"Loading model: {model_path}")
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    sr = getattr(model, "sr", None) or getattr(model, "sample_rate", None) or 44100
    if isinstance(sr, torch.Tensor):
        sr = int(sr.item())
    print(f"Sample rate: {sr} Hz")

    print("Encoding phase anchors…")
    anchors = generate_anchors_from_folders(model, phase_dirs, sr=sr)

    if not anchors:
        raise RuntimeError("No anchors generated — check that phase folders contain audio files.")

    print("Computing PCA projection…")
    pca_data = compute_pca_scatter(model, phase_dirs, anchors, sr=sr)

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    out_dir = output_dir or os.path.dirname(os.path.abspath(model_path))
    os.makedirs(out_dir, exist_ok=True)

    bundle_path = os.path.join(out_dir, f"{model_stem}_phases.json")
    pca_path = os.path.join(out_dir, f"{model_stem}_phases_pca.json")

    save_phase_bundle(anchors, bundle_path, phase_pca=pca_data)
    print(f"Bundle saved → {bundle_path}")

    with open(pca_path, "w", encoding="utf-8") as f:
        json.dump(pca_data, f, indent=2)
    print(f"PCA scatter saved → {pca_path}")

    return bundle_path
