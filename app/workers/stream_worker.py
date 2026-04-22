import math
import re
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal


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

        self.input_mode = _Var("random")
        self.audio_file_path = None
        self.encoded_latents = None
        self.latent_position = 0
        self.loop_audio = _Var(True)
        self.random_intensity = _Var(1.0)
        self.density = _Var(1.0)
        self.held_z = None
        self.density_hold_frames = 0

        self.phase_enabled = _Var(False)
        self.phase_value = _Var(0.0)
        self.phase_anchors = []

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
    masterVu = Signal(float, float, float)  # left, right, latency_ms
    warning = Signal(str)
    running = Signal(bool)

    def __init__(self, model_paths: list[str | None], sample_rate: int = 44100, block_size: int = 256):
        super().__init__()
        self._model_paths = (model_paths[:4] + [None, None, None, None])[:4]
        self._sample_rate = sample_rate
        self._block_size = block_size

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._stream = None
        self._engine = None
        self._slots: list[_SlotState] = [_SlotState(i) for i in range(4)]
        self._phase_xy = (0.5, 0.5)
        self._pending_model_changes: list[tuple[int, str | None]] = []

    def stop(self):
        self._stop_event.set()
        if self._engine is not None:
            self._engine.stop()

    def set_slot_params(self, index: int, gain: float | None = None, temp: float | None = None, smooth: float | None = None):
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

    def _get_active_slots(self):
        with self._lock:
            return [s for s in self._slots if s.is_active.get() and s.model is not None]

    def _schedule_ui_callback(self, cb):
        cb()

    def _log_msg(self, message: str):
        self.log.emit("INFO", message)

    def _load_slot_model(self, index: int, model_path: str):
        import torch

        model = torch.jit.load(model_path).eval()
        latent_size = None
        latent_length = 1
        output_length = None

        with torch.no_grad():
            for test_channels in [1, 2, 8, 16, 32, 64, 128]:
                try:
                    test_z = torch.randn(1, test_channels, 128)
                    _ = model.decode(test_z)
                    latent_size = test_channels
                    break
                except Exception:
                    continue

            if latent_size is None:
                raise RuntimeError("Could not determine model latent channel size")

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
        slot.model_sr = None
        name_match = re.search(r"_r(\d+)_", model_path)
        if name_match:
            try:
                slot.model_sr = int(name_match.group(1))
            except ValueError:
                slot.model_sr = None

        slot.prev_z = None
        slot.cached_audio = None
        slot.is_active.set(True)
        slot.status_var.set("Active")

    def _clear_slot_model(self, index: int):
        slot = self._slots[index]
        slot.model = None
        slot.model_path = None
        slot.is_active.set(False)
        slot.status_var.set("Inactive")
        slot.cached_audio = None
        slot.prev_z = None

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
                        self.slotVu.emit(i, min(1.0, level * 2.0), min(1.0, peak))

                    if self._engine.metrics["write_ms"]:
                        latency = float(self._engine.metrics["write_ms"][-1])
                    else:
                        latency = (self._block_size / self._sample_rate) * 1000.0

                left = 0.35 + 0.25 * math.sin(time.perf_counter() * 2.1)
                right = 0.33 + 0.28 * math.sin(time.perf_counter() * 1.8 + 0.2)
                self.masterVu.emit(max(0.0, left), max(0.0, right), latency)

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
