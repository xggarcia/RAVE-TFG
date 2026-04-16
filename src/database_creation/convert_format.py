"""Convert audio files to a consistent format.

Scans a directory tree for audio files and converts each one to the target
format using:
- Resampling with librosa
- Channel mapping (default mono)
- WAV writing with soundfile

Usage:
    python -m src.database_creation.convert_format --input-dir input_data/user_data/rain
    python -m src.database_creation.convert_format --input-dir input_data/user_data/rain --target-sr 44100 --target-channels 1 --target-subtype PCM_16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 array with shape (samples, channels)."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr)
    except Exception:
        # Fallback for compressed formats that may not decode via soundfile.
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim == 1:
            data = y[:, np.newaxis]
        else:
            data = y.T
        return data.astype(np.float32), int(sr)


def _to_target_channels(data: np.ndarray, target_channels: int) -> np.ndarray:
    """Map input channels to target channel count."""
    if target_channels < 1:
        raise ValueError("target_channels must be >= 1")

    samples, channels = data.shape
    if channels == target_channels:
        return data

    if target_channels == 1:
        return np.mean(data, axis=1, keepdims=True, dtype=np.float32)

    if channels == 1:
        return np.repeat(data, target_channels, axis=1)

    if channels > target_channels:
        return data[:, :target_channels]

    missing = target_channels - channels
    pad = np.repeat(data[:, -1:], missing, axis=1)
    return np.concatenate([data, pad], axis=1)


def _resample_if_needed(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample to target sample rate if needed."""
    if target_sr < 1:
        raise ValueError("target_sr must be >= 1")
    if orig_sr == target_sr:
        return data

    channels_first = data.T
    resampled = librosa.resample(channels_first, orig_sr=orig_sr, target_sr=target_sr)
    return resampled.T.astype(np.float32)


def convert_file(
    path: Path,
    target_sr: int,
    target_channels: int,
    target_subtype: str,
) -> bool:
    """Convert one file to WAV with target format. Returns True on success."""
    try:
        data, sr = _load_audio(path)
    except Exception as exc:
        print(f"  [skip] Cannot read: {exc}")
        return False

    converted = _resample_if_needed(data, sr, target_sr)
    converted = _to_target_channels(converted, target_channels)

    out_path = path.with_suffix(".wav")
    try:
        sf.write(str(out_path), converted, target_sr, subtype=target_subtype)
    except Exception as exc:
        print(f"  [error] Cannot write {out_path.name}: {exc}")
        return False

    if path != out_path and path.exists():
        path.unlink()

    print(
        f"  {path.name} -> {out_path.name}  "
        f"({sr}Hz/{data.shape[1]}ch -> {target_sr}Hz/{target_channels}ch {target_subtype})"
    )
    return True


def find_audio_files(root: Path) -> list[Path]:
    """Recursively find supported audio files under root."""
    seen: set[Path] = set()
    files: list[Path] = []
    for ext in AUDIO_EXTENSIONS:
        for p in root.rglob(f"*{ext}"):
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(p)
    return sorted(files)


def convert_directory(
    root: Path,
    target_sr: int = 44100,
    target_channels: int = 1,
    target_subtype: str = "PCM_16",
) -> tuple[int, int]:
    """Convert all audio files under root. Returns (ok_count, total_count)."""
    files = find_audio_files(root)
    if not files:
        return 0, 0

    ok = 0
    for idx, path in enumerate(files, start=1):
        print(f"  [{idx}/{len(files)}] {path.name}")
        if convert_file(path, target_sr, target_channels, target_subtype):
            ok += 1

    return ok, len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert audio files to a consistent WAV format."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing audio files (searched recursively).",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=44100,
        help="Target sample rate in Hz (default: 44100).",
    )
    parser.add_argument(
        "--target-channels",
        type=int,
        default=1,
        help="Target number of channels (default: 1).",
    )
    parser.add_argument(
        "--target-subtype",
        default="PCM_16",
        help="Target soundfile subtype for WAV output (default: PCM_16).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        print(f"Directory not found: {input_dir}")
        return 1
    if args.target_sr < 1:
        print("target-sr must be >= 1")
        return 1
    if args.target_channels < 1:
        print("target-channels must be >= 1")
        return 1

    files = find_audio_files(input_dir)
    if not files:
        print(f"No audio files found in {input_dir}")
        return 0

    print(
        "Target format: "
        f"WAV, {args.target_channels} channel(s), {args.target_sr} Hz, {args.target_subtype}\n"
    )
    ok, total = convert_directory(
        root=input_dir,
        target_sr=args.target_sr,
        target_channels=args.target_channels,
        target_subtype=args.target_subtype,
    )

    print(f"\nDone. Converted {ok}/{total} file(s).")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())