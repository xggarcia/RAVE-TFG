import threading
import time
import traceback
from collections import deque

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal

from app.workers._stream_models import _SlotState
from app.workers._stream_loader import (
    _clear_slot_model,
    _encode_audio_for_slot,
    _load_slot_model,
    _process_pending_model_changes,
    _recording_start,
    _recording_stop,
    _validate_active_slots,
    load_slot_anchors,
    load_slot_phase_map_anchor,
)


class StreamWorker(QThread):
    log = Signal(str, str)  # level, message
    finished = Signal(dict)
    failed = Signal(str, str)
    stage = Signal(str, bool, bool)  # label, done, current
    slotVu = Signal(int, float, float)
    slotSpectrogram = Signal(int, object)   # index, np.ndarray audio chunk
    slotInfo = Signal(int, int)             # index, latent_size
    slotSubbandPosition = Signal(int, int, int)  # index, current_step, total_steps
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
        self._loading_threads: list = []

        self._subband_intensity = 1.0  # Global subband control intensity (0.0-1.0)
        self._dynamic_stride_enabled = True

        self._record_lock = threading.Lock()
        self._record_enabled = False
        self._record_chunks = deque()
        self._record_path: str | None = None

    def stop(self):
        self._stop_event.set()
        if self._engine is not None:
            self._engine.stop()

    def add_slot(self):
        with self._lock:
            self._slots.append(_SlotState(len(self._slots)))

    def load_model_async(self, index: int, model_path: str | None):
        """Load (or clear) a slot model in a background thread — no need to start streaming first."""
        while len(self._slots) <= index:
            self.add_slot()
        if not model_path:
            with self._lock:
                _clear_slot_model(self._slots[index])
            self.slotInfo.emit(index, 0)
            return
        t = _ModelLoadThread(self, index, model_path)
        self._loading_threads.append(t)
        t.finished.connect(lambda: self._loading_threads.remove(t) if t in self._loading_threads else None)
        t.start()

    def _cleanup_loading_threads(self):
        for t in list(self._loading_threads):
            if t.isRunning():
                t.wait()

    def set_slot_params(self, index: int, gain: float | None = None, temp: float | None = None,
                        smooth: float | None = None, dry_wet: float | None = None,
                        noise: float | None = None, bias: float | None = None):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            slot = self._slots[index]
            if gain is not None:
                slot.gain = max(0.0, min(1.2, gain))
            if temp is not None:
                slot.temperature = max(0.1, min(3.0, temp))
            if smooth is not None:
                slot.smoothing = max(0.0, min(0.95, smooth))
            if dry_wet is not None:
                slot.dry_wet = max(0.0, min(1.0, dry_wet))
            if noise is not None:
                slot.random_intensity = max(0.0, min(3.0, noise))
            if bias is not None:
                slot.latent_global_bias = max(-2.0, min(2.0, bias))

    def set_master_volume(self, value: float):
        self._master_volume = max(0.0, min(1.2, value))
        if self._engine is not None:
            self._engine.master_volume = self._master_volume

    def set_phase_xy(self, x: float, y: float):
        with self._lock:
            self._phase_xy = (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
            for slot in self._slots:
                slot.phase_value = self._phase_xy[0]

    def set_slot_model(self, index: int, model_path: str | None):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            self._pending_model_changes.append((index, model_path))

    def set_slot_enabled(self, index: int, enabled: bool):
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].is_active = bool(enabled)

    def set_slot_input_mode(self, index: int, mode: str):
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].input_mode = mode

    def set_slot_anchors(self, index: int, path: str):
        with self._lock:
            slot = self._slots[index] if 0 <= index < len(self._slots) else None
        if slot is not None:
            load_slot_anchors(slot, self.warning.emit, path)

    def set_slot_latent_dim(self, index: int, dim: int, scale: float):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            slot = self._slots[index]
            if 0 <= dim < len(slot.latent_scale):
                slot.latent_scale[dim] = max(0.0, min(3.0, scale))

    def set_slot_phase_map_anchor(self, index: int, mean_z: list, std_z: list):
        with self._lock:
            if not (0 <= index < len(self._slots)):
                return
            load_slot_phase_map_anchor(self._slots[index], mean_z, std_z)

    def set_slot_phase(self, index: int, value: float):
        with self._lock:
            if 0 <= index < len(self._slots):
                self._slots[index].phase_value = max(0.0, min(1.0, value))

    def set_slot_subband_pattern(self, index: int, pattern: torch.Tensor | np.ndarray):
        """Set the subband pattern for a slot (num_subbands, n_timesteps)."""
        with self._lock:
            if 0 <= index < len(self._slots):
                slot = self._slots[index]
                if isinstance(pattern, torch.Tensor):
                    slot.subband_pattern = pattern.float()
                else:
                    slot.subband_pattern = torch.from_numpy(np.asarray(pattern, dtype=np.float32))
                slot.subband_position = 0

    def set_subband_intensity(self, intensity: float):
        """Set global subband intensity (0.0-1.0)."""
        with self._lock:
            self._subband_intensity = max(0.0, min(1.0, float(intensity)))
            # Update engine if running
            if self._engine is not None:
                self._engine.subband_intensity = self._subband_intensity

    def set_dynamic_stride_enabled(self, enabled: bool):
        with self._lock:
            self._dynamic_stride_enabled = bool(enabled)
            if self._engine is not None:
                self._engine.dynamic_stride_enabled = self._dynamic_stride_enabled

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
        _encode_audio_for_slot(self, index, path)

    def _get_active_slots(self):
        with self._lock:
            return [s for s in self._slots if s.is_active and s.model is not None]

    def _schedule_ui_callback(self, cb):
        cb()

    def _on_played_chunk(self, chunk):
        with self._record_lock:
            if not self._record_enabled:
                return
            self._record_chunks.append(np.array(chunk, dtype=np.float32, copy=True))

    def start_recording(self, output_path: str | None = None) -> str:
        return _recording_start(self, output_path)

    def stop_recording(self) -> str | None:
        return _recording_stop(self)

    def _log_msg(self, message: str):
        # DEBUG temporal — print a stdout además de emitir el signal
        # para poder ver todos los logs en la terminal durante diagnóstico.
        print(f"[stream] {message}", flush=True)
        self.log.emit("INFO", message)

    def run(self):
        try:
            from src.streaming.engine import StreamingEngine

            self._stop_event.clear()

            # Wait for any background model loads to finish before starting audio
            self._cleanup_loading_threads()

            # Load models for any slot that was assigned but not yet loaded
            for idx in range(len(self._slots)):
                slot = self._slots[idx]
                if slot.model is not None:
                    # Pre-loaded by load_model_async — emit stage as done
                    self.stage.emit(f"Load model {idx + 1}", True, False)
                    continue
                model_path = self._model_paths[idx] if idx < len(self._model_paths) else None
                if not model_path:
                    continue
                label = f"Load model {idx + 1}"
                self.stage.emit(label, False, True)
                _load_slot_model(self, idx, model_path)
                self.stage.emit(label, True, False)
                self.log.emit("INFO", f"Loaded {model_path}")

            active_slots = self._get_active_slots()
            if not active_slots:
                raise RuntimeError("Please load and activate at least one model before starting")

            detected_srs = _validate_active_slots(active_slots)
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
            self._engine.subband_intensity = self._subband_intensity
            self._engine.dynamic_stride_enabled = self._dynamic_stride_enabled
            self._engine.on_chunk_played = self._on_played_chunk
            self._engine.configure(
                stream=self._stream,
                sr=self._sample_rate,
                chunk_samples=self._block_size,
            )
            self.stage.emit("Allocate buffers", True, False)
            self._engine.start()
            self.running.emit(True)

            underruns_seen = 0
            while not self._stop_event.is_set() and self._engine.is_running:
                _process_pending_model_changes(self)
                with self._lock:
                    for i, slot in enumerate(self._slots):
                        level = 0.0
                        peak = 0.0
                        if slot.cached_audio is not None:
                            audio = slot.cached_audio
                            level = float(np.sqrt(np.mean(np.square(audio))))
                            peak = float(np.max(np.abs(audio)))
                            self.slotSpectrogram.emit(i, audio * slot.gain)
                        self.slotVu.emit(i, min(1.0, level * 2.0), min(1.0, peak))
                        if slot.input_mode == "gesture" and slot.subband_pattern is not None:
                            total_steps = int(slot.subband_pattern.shape[1]) if slot.subband_pattern.ndim == 2 else 0
                            if total_steps > 0:
                                self.slotSubbandPosition.emit(i, int(slot.subband_position) % total_steps, total_steps)

                    if self._engine.last_write_ms > 0.0:
                        latency = float(self._engine.last_write_ms)
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

            with self._record_lock:
                rec_was_on = self._record_enabled
            if rec_was_on:
                self.stop_recording()

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


class _ModelLoadThread(QThread):
    """Loads a single model into a slot in a background thread."""

    def __init__(self, worker: StreamWorker, index: int, model_path: str):
        super().__init__()
        self._worker = worker
        self._index = index
        self._model_path = model_path

    def run(self):
        try:
            _load_slot_model(self._worker, self._index, self._model_path)
        except Exception as exc:
            self._worker.warning.emit(f"Slot {self._index + 1} load failed: {exc}")
