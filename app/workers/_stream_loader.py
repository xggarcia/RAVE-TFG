"""Model-loading, audio-encoding, and recording helpers for StreamWorker."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.workers._stream_models import _SlotState, _detect_model_sr

if TYPE_CHECKING:
    from app.workers.stream_worker import StreamWorker


def _latent_size_hint_from_path(model_path: str) -> int | None:
    m = re.search(r"[_\-]z(\d{1,4})(?:[_\-.]|$)", model_path, re.IGNORECASE)
    if not m:
        return None
    try:
        hinted = int(m.group(1))
        return hinted if hinted > 0 else None
    except ValueError:
        return None


def _detect_model_latent_size(model, model_path: str, block_size: int) -> int:
    import torch

    for test_samples in (block_size, 2048, 8192):
        try:
            probe = torch.zeros(1, 1, test_samples).float()
            encoded = model.encode(probe)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            if torch.is_tensor(encoded) and encoded.dim() >= 2:
                channels = int(encoded.shape[1])
                if channels > 0:
                    return channels
        except Exception:
            continue

    hinted = _latent_size_hint_from_path(model_path)
    valid_channels: list[int] = []
    with torch.no_grad():
        for test_channels in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            try:
                test_z = torch.randn(1, test_channels, 128)
                _ = model.decode(test_z)
                valid_channels.append(test_channels)
            except Exception:
                continue

    if hinted is not None and hinted in valid_channels:
        return hinted
    if valid_channels:
        return max(valid_channels)

    raise RuntimeError("Could not determine model latent channel size")


def _clear_slot_model(slot: _SlotState):
    slot.model = None
    slot.model_path = None
    slot.is_active = False
    slot.status_var = "Inactive"
    slot.cached_audio = None
    slot.prev_z = None


def _validate_active_slots(active_slots: list[_SlotState]) -> set:
    output_lengths = {slot.output_length for slot in active_slots}
    if len(output_lengths) > 1:
        lengths_msg = ", ".join(str(l) for l in sorted(output_lengths))
        raise RuntimeError(f"All active models must produce same chunk size. Detected: {lengths_msg}")

    detected_srs = {slot.model_sr for slot in active_slots if slot.model_sr is not None}
    if len(detected_srs) > 1:
        srs_msg = ", ".join(str(r) for r in sorted(detected_srs))
        raise RuntimeError(f"All active models must use same sample rate. Detected: {srs_msg}")

    return detected_srs


def load_slot_anchors(slot: _SlotState, warn_fn, path: str):
    import json
    import torch

    try:
        with open(path, "r") as f:
            data = json.load(f)
        entries = data.get("phase_anchors", data) if isinstance(data, dict) else data
        anchors = []
        for e in entries:
            mean_z = torch.tensor(e["mean_z"], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            std_z  = torch.tensor(e["std_z"],  dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            anchors.append({"label": e.get("label", ""), "mean_z": mean_z, "std_z": std_z})
    except Exception as exc:
        warn_fn(f"Slot {slot.slot_id + 1}: failed to load anchors — {exc}")
        return
    slot.phase_anchors = anchors
    slot.phase_enabled = len(anchors) >= 2


def load_slot_phase_map_anchor(slot: _SlotState, mean_z: list, std_z: list):
    import torch

    if not mean_z:
        slot.phase_map_anchor = None
    else:
        slot.phase_map_anchor = {
            "mean_z": torch.tensor(mean_z, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
            "std_z":  torch.tensor(std_z,  dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
        }


def _encode_audio_for_slot(worker: "StreamWorker", index: int, path: str):
    import torch
    import soundfile as sf

    slot = worker._slots[index]
    if slot.model is None:
        return

    try:
        audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        model_sr = slot.model_sr or worker._sample_rate
        if file_sr != model_sr:
            import torchaudio
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(file_sr, model_sr)
            audio = resampler(audio_t).squeeze(0).numpy()

        audio_t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            encoded = slot.model.encode(audio_t)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]

        # DEBUG temporal — diagnóstico de velocidad 2× en modo audio
        input_samples = int(audio_t.shape[-1])
        input_duration_s = input_samples / model_sr if model_sr else 0.0
        encoded_frames = int(encoded.shape[-1])
        implied_encoder_ratio = input_samples / encoded_frames if encoded_frames else 0.0
        ratio_check = (
            slot.output_length / implied_encoder_ratio if implied_encoder_ratio else 0.0
        )
        _debug_msg = (
            f"[DEBUG encode] slot={index + 1} "
            f"file_sr={file_sr} model_sr={model_sr} "
            f"input_samples={input_samples} input_duration={input_duration_s:.2f}s "
            f"encoded_frames={encoded_frames} "
            f"implied_encoder_ratio={implied_encoder_ratio:.1f} "
            f"output_length={slot.output_length} "
            f"ratio_check={ratio_check:.3f}"
        )
        print(_debug_msg, flush=True)
        worker.log.emit("INFO", _debug_msg)

        with worker._lock:
            if slot.audio_file_path == path:
                slot.encoded_latents = encoded
                slot.raw_audio = audio
                slot.latent_position = 0
                slot.audio_sample_pos = 0
        worker.log.emit("INFO", f"Slot {index + 1}: audio encoded ({encoded.shape[-1]} frames)")
    except Exception as exc:
        worker.warning.emit(f"Slot {index + 1}: audio encode failed — {exc}")


def _load_slot_model(worker: "StreamWorker", index: int, model_path: str):
    import torch

    model = torch.jit.load(model_path).eval()
    latent_size = None
    latent_length = 1
    output_length = None

    with torch.no_grad():
        latent_size = _detect_model_latent_size(model, model_path, worker._block_size)

        while latent_length <= 16384:
            try:
                test_z = torch.randn(1, latent_size, latent_length)
                test_output = model.decode(test_z)
                output_length = int(test_output.shape[-1])
                break
            except Exception:
                latent_length *= 2

        if output_length is None:
            raise RuntimeError("Could not determine model output length")

    slot = worker._slots[index]
    slot.model = model
    slot.model_path = model_path
    slot.latent_size = latent_size
    slot.latent_length = latent_length
    slot.output_length = output_length
    slot.model_sr = _detect_model_sr(model, model_path)
    slot.latent_scale = [1.0] * latent_size

    # DEBUG temporal — diagnóstico de velocidad 2× en modo audio
    _debug_msg = (
        f"[DEBUG load] slot={index + 1} "
        f"latent_size={slot.latent_size} "
        f"latent_length={slot.latent_length} "
        f"output_length={slot.output_length} "
        f"model_sr={slot.model_sr}"
    )
    print(_debug_msg, flush=True)
    worker.log.emit("INFO", _debug_msg)

    slot.prev_z = None
    slot.cached_audio = None
    slot.is_active = True
    slot.status_var = "Active"

    worker.slotInfo.emit(index, latent_size)

    if slot.audio_file_path and slot.encoded_latents is None:
        _encode_audio_for_slot(worker, index, slot.audio_file_path)


def _process_pending_model_changes(worker: "StreamWorker"):
    with worker._lock:
        pending = list(worker._pending_model_changes)
        worker._pending_model_changes.clear()

    for index, model_path in pending:
        if not model_path:
            _clear_slot_model(worker._slots[index])
            worker.log.emit("INFO", f"Slot {index + 1} unloaded")
            continue
        try:
            _load_slot_model(worker, index, model_path)
            if worker._engine is not None and worker._engine.is_running:
                slot = worker._slots[index]
                if slot.output_length != worker._block_size:
                    _clear_slot_model(slot)
                    raise RuntimeError(
                        f"Model output {slot.output_length} does not match running chunk size {worker._block_size}"
                    )
                if slot.model_sr is not None and slot.model_sr != worker._sample_rate:
                    _clear_slot_model(slot)
                    raise RuntimeError(
                        f"Model sample rate {slot.model_sr} does not match running rate {worker._sample_rate}"
                    )
            worker.log.emit("INFO", f"Slot {index + 1} loaded {model_path}")
        except Exception as exc:
            worker.warning.emit(f"Slot {index + 1} load failed: {exc}")


def _recording_start(worker: "StreamWorker", output_path: str | None = None) -> str:
    with worker._record_lock:
        if worker._record_enabled:
            return worker._record_path or ""

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            rec_dir = Path("outputs") / "recordings"
            rec_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = rec_dir / f"stream_recording_{stamp}.wav"

        worker._record_chunks.clear()
        worker._record_path = str(path)
        worker._record_enabled = True

    worker.log.emit("INFO", f"Recording started: {worker._record_path}")
    return worker._record_path


def _recording_stop(worker: "StreamWorker") -> str | None:
    with worker._record_lock:
        if not worker._record_enabled:
            return None
        worker._record_enabled = False
        chunks = list(worker._record_chunks)
        worker._record_chunks.clear()
        out_path = worker._record_path

    if not out_path:
        return None

    if not chunks:
        worker.log.emit("WARN", "Recording stopped with no captured audio")
        return out_path

    import soundfile as sf

    try:
        with sf.SoundFile(out_path, mode="w", samplerate=worker._sample_rate, channels=1, subtype="PCM_16") as f:
            for chunk in chunks:
                if chunk is None or len(chunk) == 0:
                    continue
                f.write(np.asarray(chunk, dtype=np.float32))
        worker.log.emit("INFO", f"Recording saved: {out_path}")
    except Exception as exc:
        worker.warning.emit(f"Recording save failed: {exc}")

    return out_path
