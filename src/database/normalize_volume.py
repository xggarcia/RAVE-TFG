"""Normalize audio files to a consistent RMS volume level.

Scans a directory tree for audio files (WAV, FLAC, OGG, MP3) and normalizes
each one to a target RMS (dBFS). MP3/OGG files are converted to WAV in place.

Usage:
    python -m src.database.normalize_volume --input-dir rainSounds
    python -m src.database.normalize_volume --input-dir rainSounds --target-db -18.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 numpy array shape (samples, channels) and sample rate.

    Uses soundfile for lossless formats (WAV/FLAC/OGG/AIFF) and librosa for MP3.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr)
    except Exception:
        # Fallback for MP3 and other formats soundfile cannot decode.
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim == 1:
            data = y[:, np.newaxis]          # (samples,) → (samples, 1)
        else:
            data = y.T                       # (channels, samples) → (samples, channels)
        return data.astype(np.float32), int(sr)


def _rms_db(data: np.ndarray) -> float:
    """Return RMS level in dBFS. Returns -100.0 for silence."""
    rms = float(np.sqrt(np.mean(data ** 2)))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)


def _apply_normalization(data: np.ndarray, target_db: float) -> np.ndarray:
    """Scale data so its RMS equals target_db. Reduces gain if clipping would occur."""
    current_db = _rms_db(data)
    if current_db < -99.0:
        return data  # silence, nothing to do

    gain = 10.0 ** ((target_db - current_db) / 20.0)
    normalized = data * gain

    peak = float(np.max(np.abs(normalized)))
    if peak > 1.0:
        normalized /= peak

    return normalized


def normalize_file(path: Path, target_db: float) -> bool:
    """Normalize one audio file and save as WAV (16-bit PCM). Returns True on success."""
    try:
        data, sr = _load_audio(path)
    except Exception as exc:
        print(f"  [skip] Cannot read: {exc}")
        return False

    before_db = _rms_db(data)
    if before_db < -99.0:
        print(f"  [skip] File appears to be silence.")
        return False

    normalized = _apply_normalization(data, target_db)
    after_db = _rms_db(normalized)

    out_path = path.with_suffix(".wav")
    try:
        sf.write(str(out_path), normalized, sr, subtype="PCM_16")
    except Exception as exc:
        print(f"  [error] Cannot write {out_path.name}: {exc}")
        return False

    # Remove source if it was a different format (MP3, OGG, etc.)
    if path.suffix.lower() != ".wav" and path.exists():
        path.unlink()

    print(f"  {path.name} → {out_path.name}  ({before_db:.1f} → {after_db:.1f} dBFS)")
    return True


def find_audio_files(root: Path) -> list[Path]:
    """Recursively find all audio files under root, deduplicating by resolved path."""
    seen: set[Path] = set()
    files: list[Path] = []
    for ext in AUDIO_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(p)
    return sorted(files)


def normalize_directory(root: Path, target_db: float = -20.0) -> tuple[int, int]:
    """Normalize all audio files under root. Returns (ok_count, total_count)."""
    files = find_audio_files(root)
    if not files:
        return 0, 0
    ok = 0
    for idx, path in enumerate(files, start=1):
        print(f"  [{idx}/{len(files)}] {path.name}")
        if normalize_file(path, target_db):
            ok += 1
    return ok, len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize audio files to a consistent RMS volume level."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing audio files (searched recursively).",
    )
    parser.add_argument(
        "--target-db",
        type=float,
        default=-20.0,
        help="Target RMS level in dBFS (default: -20.0).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        print(f"Directory not found: {input_dir}")
        return 1

    files = find_audio_files(input_dir)
    if not files:
        print(f"No audio files found in {input_dir}")
        return 0

    print(f"Target RMS: {args.target_db:.1f} dBFS\n")
    ok, total = normalize_directory(input_dir, args.target_db)
    if total == 0:
        print(f"No audio files found in {input_dir}")
        return 0
    print(f"\nDone. Normalized {ok}/{total} file(s).")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
