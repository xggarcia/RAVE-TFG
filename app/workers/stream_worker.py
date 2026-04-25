import re
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal


def _detect_model_sr(model, model_path: str) -> int | None:
    # 1. Try model.sr attribute (RAVE exports this in most versions)
    for attr in ("sr", "sample_rate", "sampling_rate"):
        try:
            val = int(getattr(model, attr))
            if val > 0:
                return val
        except Exception:
            pass

    # 2. Try model.sr() as a callable (some TorchScript exports)
    try:
        val = int(model.sr())
        if val > 0:
            return val
    except Exception:
        pass

    # 3. Fall back to filename heuristic: _r44100_ or _22050_ etc.
    for pattern in (r"_r(\d{4,6})_", r"[_\-](\d{4,6})[_\-]", r"(\d{4,6})hz"):
        m = re.search(pattern, model_path, re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1))
                if 8000 <= val <= 192000:
                    return val
            except ValueError:
                pass

    return None


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _SlotState:
    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.model = None
        self.model_path = None
        self.model_sr = None
        self.is_active = _Var(False)
        self.status_var = _Var("Inactive")

        self.latent_size = 16
        self.latent_length = 1
        self.output_length = 1
        self.prev_z = None

        self.gain = _Var(0.6)
        self.temperature = _Var(0.6)
        self.smoothing = _Var(0.4)
        self.dry_wet = _Var(1.0)  # 0 = dry (passthrough silence), 1 = fully wet

        self.input_mode = _Var("random")
        self.audio_file_path = None
        self.encoded_latents = None
        self.raw_audio = None          # float32 numpy array, resampled to model SR
        self.latent_position = 0
        self.audio_sample_pos = 0      # advances by chunk_samples in sync with latent_position
        self.loop_audio = _Var(True)
        self.random_intensity = _Var(1.0)
        self.density = _Var(1.0)
        self.held_z = None
        self.density_hold_frames = 0

        self.latent_bias: list[float] = []   # per-dim additive bias
        self.latent_scale: list[float] = []  # per-dim multiplicative scale
        self.latent_global_bias = _Var(0.0)  # additive bias applied to all latent dims

        self.phase_enabled = _Var(False)
        self.phase_value = _Var(0.0)
        self.phase_anchors = []
        self.phase_map_anchor = None   # {"mean_z": tensor, "std_z": tensor} or None

        self.use_prior = _Var(False)
        self.prior_model = None
        self.embedded_prior_available = False
        self.prior_seed_channels = None
        self.embedded_prior_seed_channels = None
        self.prior_temperature = _Var(1.0)
        self.prior_needs_warmup = True
        self.prior_chunks_generated = 0

        self.cached_audio = None
        self.last_decode_cycle = -1


class StreamWorker(QThread):
    log = Signal(str, str)  # level, message
    finished = Signal(dict)
    failed = Signal(str, str)
    stage = Signal(str, bool, bool)  # label, done, current
    slotVu = Signal(int, float, float)
    slotSpectrogram = Signal(int, object)   # index, np.ndarray audio chunk
    slotInfo = Signal(int, int)             # index, latent_size
    masterVu = Signal(float, float, float)  # left, right, latency_ms
    warning = Signal(str)
    running = Signal(bool)

    def __init__(self, model_paths: list[str | None], sample_rate: int = 44100, block_size: int = 256):
        super().__init__()
        self._model_paths = list(model_paths)
        self._sample_rate = sample_rate
        self._block_size = block_size

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._stream = None
        self._engine = None
        self._master_volume = 0.75
        self._slots: list[_SlotState] = [_SlotState(i) for i in range(len(self._model_paths))]
        self._phase_xy = (0.5, 0.5)
        self._pending_model_changes: list[tuple[int, str | None]] = []

    @staticmethod
    def _coerce_prior_latent(z, latent_size: int, latent_length: int):
        import torch

        if isinstance(z, (tuple, list)):
            tensors = [item for item in z if torch.is_tensor(item)]
            if not tensors:
                raise ValueError("Prior output tuple/list has no tensor")
            z = tensors[0]

        if not torch.is_tensor(z):
            raise ValueError("Prior output is not a tensor")

        if z.dim() == 2:
            z = z.unsqueeze(0)
        elif z.dim() != 3:
            raise ValueError(f"Unsupported prior latent rank: {z.dim()}")

        if z.shape[1] != latent_size and z.shape[2] == latent_size:
            z = z.transpose(1, 2)

        if z.shape[1] != latent_size:
            raise ValueError(f"Prior latent channels mismatch: got {z.shape[1]}, expected {latent_size}")

        if z.shape[2] > latent_length:
            z = z[:, :, :latent_length]
        elif z.shape[2] < latent_length:
            z = torch.nn.functional.pad(z, (0, latent_length - z.shape[2]))

        return z

    def _find_compatible_prior_seed(self, prior_model, decoder_model, latent_size: int, latent_length: int):
        import torch

        with torch.no_grad():
            for seed_channels in (1, 1024):
                try:
                    seed = torch.zeros(1, seed_channels, latent_length).float()
                    z_out = prior_model.prior(seed)
                    z = self._coerce_prior_latent(z_out, latent_size, latent_length)
                    _ = decoder_model.decode(z)
                    return seed_channels
                except Exception:
                    continue
        return None

    def stop(self):
        self._stop_event.set()
        if self._engine is not None:
            self._engine.stop()

    def set_slot_params(self, index: int, gain: float | None = None, temp: float | None = None,
                        smooth: float | None = None, dry_wet: float | None = None,
                        noise: float | None = None, bias: float | None = None):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            slot = self._slots[index]
            if gain is not None:
                slot.gain.set(max(0.0, min(1.2, gain)))
            if temp is not None:
                slot.temperature.set(max(0.1, min(3.0, temp)))
            if smooth is not None:
                slot.smoothing.set(max(0.0, min(0.95, smooth)))
            if dry_wet is not None:
                slot.dry_wet.set(max(0.0, min(1.0, dry_wet)))
            if noise is not None:
                slot.random_intensity.set(max(0.0, min(3.0, noise)))
            if bias is not None:
                slot.latent_global_bias.set(max(-2.0, min(2.0, bias)))

    def set_master_volume(self, value: float):
        self._master_volume = max(0.0, min(1.2, value))
        if self._engine is not None:
            self._engine.master_volume = self._master_volume

    def set_phase_xy(self, x: float, y: float):
        with self._lock:
            self._phase_xy = (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
            for slot in self._slots:
                slot.phase_value.set(self._phase_xy[0])

    def set_slot_model(self, index: int, model_path: str | None):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            self._pending_model_changes.append((index, model_path))

    def set_slot_enabled(self, index: int, enabled: bool):
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].is_active.set(enabled)

    def set_slot_input_mode(self, index: int, mode: str):
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].input_mode.set(mode)

    def set_slot_anchors(self, index: int, path: str):
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
            self.warning.emit(f"Slot {index + 1}: failed to load anchors — {exc}")
            return
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].phase_anchors = anchors
                self._slots[index].phase_enabled.set(len(anchors) >= 2)

    def set_slot_latent_dim(self, index: int, dim: int, bias: float, scale: float):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            slot = self._slots[index]
            if 0 <= dim < len(slot.latent_bias):
                slot.latent_bias[dim] = max(-2.0, min(2.0, bias))
            if 0 <= dim < len(slot.latent_scale):
                slot.latent_scale[dim] = max(0.0, min(3.0, scale))

    def set_slot_use_prior(self, index: int, enabled: bool):
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].use_prior.set(enabled)

    def set_slot_phase_map_anchor(self, index: int, mean_z: list, std_z: list):
        import torch
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            slot = self._slots[index]
            if not mean_z:
                slot.phase_map_anchor = None
            else:
                slot.phase_map_anchor = {
                    "mean_z": torch.tensor(mean_z, dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                    "std_z":  torch.tensor(std_z,  dtype=torch.float32).unsqueeze(0).unsqueeze(-1),
                }

    def set_slot_phase(self, index: int, value: float):
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].phase_value.set(max(0.0, min(1.0, value)))

    def set_slot_audio_file(self, index: int, path: str):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            slot = self._slots[index]
            slot.audio_file_path = path
            slot.encoded_latents = None
            slot.raw_audio = None
            slot.latent_position = 0
            slot.audio_sample_pos = 0

        # Encode off the lock so the audio thread isn't blocked
        self._encode_audio_for_slot(index, path)

    def _encode_audio_for_slot(self, index: int, path: str):
        import torch
        import soundfile as sf
        import numpy as np

        slot = self._slots[index]
        if slot.model is None:
            # Model not loaded yet; encoding will happen when model loads
            return

        try:
            audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            model_sr = slot.model_sr or self._sample_rate
            if file_sr != model_sr:
                import torchaudio
                audio_t = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(file_sr, model_sr)
                audio = resampler(audio_t).squeeze(0).numpy()

            audio_t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            with torch.no_grad():
                encoded = slot.model.encode(audio_t)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]

            with self._lock:
                if slot.audio_file_path == path:   # still the same file
                    slot.encoded_latents = encoded
                    slot.raw_audio = audio
                    slot.latent_position = 0
                    slot.audio_sample_pos = 0
            self.log.emit("INFO", f"Slot {index + 1}: audio encoded ({encoded.shape[-1]} frames)")
        except Exception as exc:
            self.warning.emit(f"Slot {index + 1}: audio encode failed — {exc}")

    def _get_active_slots(self):
        with self._lock:
            return [s for s in self._slots if s.is_active.get() and s.model is not None]

    def _schedule_ui_callback(self, cb):
        cb()

    def _log_msg(self, message: str):
        self.log.emit("INFO", message)

    @staticmethod
    def _latent_size_hint_from_path(model_path: str) -> int | None:
        m = re.search(r"[_\-]z(\d{1,4})(?:[_\-.]|$)", model_path, re.IGNORECASE)
        if not m:
            return None
        try:
            hinted = int(m.group(1))
            return hinted if hinted > 0 else None
        except ValueError:
            return None

    def _detect_model_latent_size(self, model, model_path: str) -> int:
        import torch

        # Prefer encode() output when available: it reflects the model's true latent channels.
        for test_samples in (self._block_size, 2048, 8192):
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

        hinted = self._latent_size_hint_from_path(model_path)
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
            # Some scripted decoders can accept multiple channel counts; prefer larger plausible latent size.
            return max(valid_channels)

        raise RuntimeError("Could not determine model latent channel size")

    def _load_slot_model(self, index: int, model_path: str):
        import torch

        model = torch.jit.load(model_path).eval()
        latent_size = None
        latent_length = 1
        output_length = None

        with torch.no_grad():
            latent_size = self._detect_model_latent_size(model, model_path)

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

        slot = self._slots[index]
        slot.model = model
        slot.model_path = model_path
        slot.latent_size = latent_size
        slot.latent_length = latent_length
        slot.output_length = output_length
        slot.model_sr = _detect_model_sr(model, model_path)
        slot.latent_bias = [0.0] * latent_size
        slot.latent_scale = [1.0] * latent_size
        slot.prior_model = None
        slot.prior_seed_channels = None
        slot.embedded_prior_available = False
        slot.embedded_prior_seed_channels = None
        slot.prior_needs_warmup = True
        slot.prior_chunks_generated = 0

        if hasattr(model, "prior"):
            embedded_seed = self._find_compatible_prior_seed(
                prior_model=model,
                decoder_model=model,
                latent_size=latent_size,
                latent_length=latent_length,
            )
            if embedded_seed is not None:
                slot.embedded_prior_available = True
                slot.embedded_prior_seed_channels = embedded_seed
                self.log.emit("INFO", f"Slot {index + 1}: embedded prior available (seed channels: {embedded_seed})")

        slot.prev_z = None
        slot.cached_audio = None
        slot.is_active.set(True)
        slot.status_var.set("Active")

        self.slotInfo.emit(index, latent_size)

        # If a file was already picked before the model loaded, encode it now
        if slot.audio_file_path and slot.encoded_latents is None:
            self._encode_audio_for_slot(index, slot.audio_file_path)

    def _clear_slot_model(self, index: int):
        slot = self._slots[index]
        slot.model = None
        slot.model_path = None
        slot.is_active.set(False)
        slot.status_var.set("Inactive")
        slot.cached_audio = None
        slot.prev_z = None
        slot.prior_model = None
        slot.prior_seed_channels = None
        slot.embedded_prior_available = False
        slot.embedded_prior_seed_channels = None
        slot.prior_needs_warmup = True
        slot.prior_chunks_generated = 0

    def _process_pending_model_changes(self):
        with self._lock:
            pending = list(self._pending_model_changes)
            self._pending_model_changes.clear()

        for index, model_path in pending:
            if not model_path:
                self._clear_slot_model(index)
                self.log.emit("INFO", f"Slot {index + 1} unloaded")
                continue
            try:
                self._load_slot_model(index, model_path)
                if self._engine is not None and self._engine.is_running:
                    slot = self._slots[index]
                    if slot.output_length != self._block_size:
                        self._clear_slot_model(index)
                        raise RuntimeError(
                            f"Model output {slot.output_length} does not match running chunk size {self._block_size}"
                        )
                    if slot.model_sr is not None and slot.model_sr != self._sample_rate:
                        self._clear_slot_model(index)
                        raise RuntimeError(
                            f"Model sample rate {slot.model_sr} does not match running rate {self._sample_rate}"
                        )
                self.log.emit("INFO", f"Slot {index + 1} loaded {model_path}")
            except Exception as exc:
                self.warning.emit(f"Slot {index + 1} load failed: {exc}")

    @staticmethod
    def _validate_active_slots(active_slots: list[_SlotState]):
        output_lengths = {slot.output_length for slot in active_slots}
        if len(output_lengths) > 1:
            lengths_msg = ", ".join(str(length) for length in sorted(output_lengths))
            raise RuntimeError(f"All active models must produce same chunk size. Detected: {lengths_msg}")

        detected_srs = {slot.model_sr for slot in active_slots if slot.model_sr is not None}
        if len(detected_srs) > 1:
            srs_msg = ", ".join(str(rate) for rate in sorted(detected_srs))
            raise RuntimeError(f"All active models must use same sample rate. Detected: {srs_msg}")

        return detected_srs

    def run(self):
        try:
            from src.streaming.engine import StreamingEngine

            self._stop_event.clear()

            if self._model_paths:
                for idx, model_path in enumerate(self._model_paths):
                    if not model_path:
                        continue
                    label = f"Load model {idx + 1}"
                    self.stage.emit(label, False, True)

                    self._load_slot_model(idx, model_path)

                    self.stage.emit(label, True, False)
                    self.log.emit("INFO", f"Loaded {model_path}")

            active_slots = self._get_active_slots()
            if not active_slots:
                raise RuntimeError("Please load and activate at least one model before starting")

            detected_srs = self._validate_active_slots(active_slots)
            self._block_size = active_slots[0].output_length
            if len(detected_srs) == 1:
                model_sr = next(iter(detected_srs))
                if self._sample_rate != model_sr:
                    self.log.emit(
                        "INFO",
                        f"Adjusting sample rate from {self._sample_rate} Hz to {model_sr} Hz to match loaded model",
                    )
                    self._sample_rate = model_sr

            self.stage.emit("Open audio devices", False, True)
            import sounddevice as sd

            _ = sd.query_devices()
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._block_size,
            )
            self._stream.start()
            self.stage.emit("Open audio devices", True, False)

            self.stage.emit("Allocate buffers", False, True)
            self._engine = StreamingEngine(self._log_msg, self._get_active_slots, self._schedule_ui_callback)
            self._engine.master_volume = self._master_volume
            self._engine.configure(
                stream=self._stream,
                sr=self._sample_rate,
                chunk_samples=self._block_size,
                performance_mode="Balanced",
            )
            self.stage.emit("Allocate buffers", True, False)
            self._engine.start()
            self.running.emit(True)

            underruns_seen = 0
            while not self._stop_event.is_set() and self._engine.is_running:
                self._process_pending_model_changes()
                with self._lock:
                    for i, slot in enumerate(self._slots):
                        level = 0.0
                        peak = 0.0
                        if slot.cached_audio is not None:
                            audio = slot.cached_audio
                            level = float(np.sqrt(np.mean(np.square(audio))))
                            peak = float(np.max(np.abs(audio)))
                            self.slotSpectrogram.emit(i, audio * slot.gain.get())
                        self.slotVu.emit(i, min(1.0, level * 2.0), min(1.0, peak))

                    if self._engine.metrics["write_ms"]:
                        latency = float(self._engine.metrics["write_ms"][-1])
                    else:
                        latency = (self._block_size / self._sample_rate) * 1000.0

                if self._engine is not None:
                    chunk = self._engine.get_last_output_chunk()
                else:
                    chunk = None
                if chunk is not None and len(chunk) > 0:
                    rms = float(np.sqrt(np.mean(np.square(chunk))))
                    peak = float(np.max(np.abs(chunk)))
                    level = min(1.0, rms * 2.0)
                    peak_level = min(1.0, peak)
                    self.masterVu.emit(level, peak_level, latency)
                else:
                    self.masterVu.emit(0.0, 0.0, latency)

                underruns_now = int(self._engine.metrics.get("underruns", 0))
                if underruns_now > underruns_seen:
                    underruns_seen = underruns_now
                    self.warning.emit(f"Buffer underrun detected ({underruns_now})")

                time.sleep(0.1)

            if self._engine is not None:
                self._engine.stop()

            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            self.running.emit(False)
            self.finished.emit(
                {
                    "sample_rate": self._sample_rate,
                    "block_size": self._block_size,
                    "loaded_models": [p for p in self._model_paths],
                    "underruns": underruns_seen,
                }
            )
        except Exception as exc:
            self.running.emit(False)
            self.failed.emit(str(exc), traceback.format_exc())
