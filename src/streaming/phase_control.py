"""Phase control via latent space interpolation.

Encodes reference audio clips into latent anchors and interpolates
between them at inference time to steer generation toward a target
sound texture (e.g. soft rain -> storm).
"""

import json
import os

import librosa
import torch


def encode_phase_anchor(model, audio_path, sr):
    """Encode a reference audio clip into a latent anchor.

    Args:
        model: Loaded TorchScript RAVE model with .encode() method.
        audio_path: Path to the reference audio file.
        sr: Sample rate to load audio at.

    Returns:
        Dict with 'mean_z' and 'std_z' tensors of shape (1, latent_size, 1).
    """
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        z = model.encode(audio_tensor)

    mean_z = z.mean(dim=2, keepdim=True)
    std_z = z.std(dim=2, keepdim=True).clamp(min=0.01)

    return {"mean_z": mean_z, "std_z": std_z}


def apply_phase_bias(z, blended_mean, blended_std):
    """Shift a latent tensor toward a target distribution.

    Centres z by subtracting its cross-channel mean and rescales the
    residual with the anchor's per-channel std before adding the anchor
    mean. The engine operates with latent_length=1 in practice; the
    formula is well-defined for any latent_length.

    Args:
        z: Latent tensor of shape (1, latent_size, latent_length).
        blended_mean: Target mean, shape (1, latent_size, 1).
        blended_std: Target std, shape (1, latent_size, 1).

    Returns:
        Biased latent tensor, same shape as z.
    """
    return blended_mean + (z - z.mean(dim=1, keepdim=True)) * blended_std.clamp(min=0.01)


def save_phase_anchors(anchors, path):
    """Save phase anchors to a JSON file.

    Args:
        anchors: List of anchor dicts with 'label', 'mean_z', 'std_z'.
        path: Output file path (.json).
    """
    data = []
    for a in anchors:
        data.append({
            "label": a["label"],
            "mean_z": a["mean_z"].squeeze().tolist(),
            "std_z": a["std_z"].squeeze().tolist(),
        })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"phase_anchors": data}, f, indent=2)


def save_phase_bundle(anchors, path, phase_pca=None):
    """Save a single combined phase file with anchors and optional 2-D map.

    Args:
        anchors: List of anchor dicts with 'label', 'mean_z', 'std_z'.
        path: Output file path (.json).
        phase_pca: Optional list of PCA entries:
                   [{"label": str, "points": [[x, y], ...], "anchor_xy": [x, y]}]
    """
    anchor_data = []
    for a in anchors:
        anchor_data.append({
            "label": a["label"],
            "mean_z": a["mean_z"].squeeze().tolist(),
            "std_z": a["std_z"].squeeze().tolist(),
        })

    bundle = {
        "format": "rave_phase_bundle",
        "version": 1,
        "phase_anchors": anchor_data,
        "phase_pca": phase_pca if isinstance(phase_pca, list) else [],
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)


def load_phase_anchors(path):
    """Load phase anchors from a JSON file.

    Args:
        path: Path to anchors JSON file.

    Returns:
        List of anchor dicts with 'label', 'mean_z', 'std_z' tensors.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    anchors = []
    for entry in data["phase_anchors"]:
        mean_z = torch.tensor(entry["mean_z"]).float().unsqueeze(0).unsqueeze(-1)
        std_z = torch.tensor(entry["std_z"]).float().unsqueeze(0).unsqueeze(-1)
        anchors.append({"label": entry["label"], "mean_z": mean_z, "std_z": std_z})

    return anchors


def compute_pca_scatter(model, phase_dirs, anchors, sr=44100):
    """Encode all phase audio, PCA-project to 2D, return scatter data for the UI.

    Returns a list of dicts:
        {"label": str, "points": [[x, y], ...], "anchor_xy": [x, y]}
    One entry per phase, points are all encoded latent frames projected to 2D.
    """
    import numpy as np
    from collections import defaultdict
    from sklearn.decomposition import PCA

    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
    all_frames: list = []
    frame_labels: list = []

    for label, folder in phase_dirs:
        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in audio_exts
        ]
        for fpath in files:
            try:
                audio, _ = librosa.load(fpath, sr=sr, mono=True)
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    z = model.encode(audio_tensor)          # (1, L, T)
                frames = z.squeeze(0).permute(1, 0).cpu().numpy()  # (T, L)
                all_frames.append(frames)
                frame_labels.extend([label] * frames.shape[0])
            except Exception as exc:
                print(f"  [!] PCA: skipping {fpath}: {exc}")

    if not all_frames:
        return []

    matrix = np.concatenate(all_frames, axis=0)            # (N, L)
    n_comp = min(2, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(matrix)                     # (N, 2)

    # Project each anchor mean into the same PCA space
    anchor_2d: dict[str, list] = {}
    for anchor in anchors:
        lbl = anchor["label"]
        mean_z = anchor["mean_z"].squeeze().cpu().numpy().reshape(1, -1)
        try:
            xy = pca.transform(mean_z)[0]
            anchor_2d[lbl] = [float(xy[0]), float(xy[1])]
        except Exception:
            anchor_2d[lbl] = [0.0, 0.0]

    # Group scatter points by phase
    phase_pts: dict = defaultdict(list)
    for (x, y), lbl in zip(coords, frame_labels):
        phase_pts[lbl].append([float(x), float(y)])

    result = []
    for label, _ in phase_dirs:
        result.append({
            "label": label,
            "points": phase_pts.get(label, []),
            "anchor_xy": anchor_2d.get(label, [0.0, 0.0]),
        })
    return result


def generate_anchors_from_folders(model, phase_dirs, sr=44100):
    """Encode phase anchors from organized audio folders.

    Each folder in phase_dirs represents one phase. All audio files within
    a folder are encoded and their latents are averaged to produce a single
    anchor per phase.

    Args:
        model: Loaded TorchScript RAVE model.
        phase_dirs: List of (label, folder_path) tuples, ordered by phase intensity.
        sr: Sample rate.

    Returns:
        List of anchor dicts with 'label', 'mean_z', 'std_z'.
    """
    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
    anchors = []

    for label, folder in phase_dirs:
        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in audio_exts
        ]

        if not files:
            print(f"  [!] No audio files in {folder}, skipping phase '{label}'")
            continue

        all_z = []
        for fpath in files:
            audio, _ = librosa.load(fpath, sr=sr, mono=True)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                z = model.encode(audio_tensor)
            all_z.append(z)

        combined = torch.cat(all_z, dim=2)
        mean_z = combined.mean(dim=2, keepdim=True)
        std_z = combined.std(dim=2, keepdim=True).clamp(min=0.01)

        anchors.append({"label": label, "mean_z": mean_z, "std_z": std_z})
        print(f"  [OK] Phase '{label}': {len(files)} files encoded")

    return anchors
