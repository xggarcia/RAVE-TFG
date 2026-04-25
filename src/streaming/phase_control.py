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


def interpolate_phase(anchors, phase_value):
    """Interpolate between phase anchors based on slider position.

    Anchors are evenly spaced along [0.0, 1.0]. With N anchors, anchor i
    sits at position i / (N - 1). The two bracketing anchors are blended
    linearly.

    Args:
        anchors: List of anchor dicts, each with 'mean_z' and 'std_z'.
        phase_value: Float in [0.0, 1.0] from the phase slider.

    Returns:
        Tuple of (blended_mean, blended_std), each shape (1, latent_size, 1).
    """
    n = len(anchors)
    phase_value = max(0.0, min(1.0, phase_value))

    if n < 2:
        return anchors[0]["mean_z"], anchors[0]["std_z"]

    segment_length = 1.0 / (n - 1)
    idx = min(int(phase_value / segment_length), n - 2)
    t = (phase_value - idx * segment_length) / segment_length
    t = max(0.0, min(1.0, t))

    left = anchors[idx]
    right = anchors[idx + 1]

    blended_mean = (1 - t) * left["mean_z"] + t * right["mean_z"]
    blended_std = (1 - t) * left["std_z"] + t * right["std_z"]

    return blended_mean, blended_std


def apply_phase_bias(z, blended_mean, blended_std):
    """Shift a latent tensor toward a target distribution.

    Normalizes z per-channel to zero mean / unit variance, then rescales
    to match the blended anchor distribution.

    Args:
        z: Latent tensor of shape (1, latent_size, latent_length).
        blended_mean: Target mean, shape (1, latent_size, 1).
        blended_std: Target std, shape (1, latent_size, 1).

    Returns:
        Biased latent tensor, same shape as z.
    """
    if z.shape[2] == 1:
        # Single frame — skip normalisation, just shift toward anchor mean
        return blended_mean + (z - z.mean(dim=1, keepdim=True)) * blended_std.clamp(min=0.01)

    z_mean = z.mean(dim=2, keepdim=True)
    # correction=0 (population std) avoids the degrees-of-freedom error when latent_length=1
    z_std = z.std(dim=2, keepdim=True, correction=0).clamp(min=0.01)

    z_norm = (z - z_mean) / z_std
    return z_norm * blended_std + blended_mean


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
