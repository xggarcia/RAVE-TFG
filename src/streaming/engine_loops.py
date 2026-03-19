import queue
import time

import numpy as np
import torch


def deactivate_finished_audio_slot(self, slot):
    slot.is_active.set(False)
    slot.status_var.set("Loaded (Audio Finished)")
    self.log(f"Slot {slot.slot_id + 1}: Audio playback finished")


def generate_mixed_chunk(self, active_slots):
    mixed_audio = np.zeros(self.chunk_samples, dtype=np.float32)
    decode_total_ms = 0.0

    for slot in active_slots:
        should_decode = (slot.cached_audio is None) or ((self._producer_cycle + slot.slot_id) % self.decode_stride == 0)

        if should_decode:
            decode_start = time.perf_counter()
            with torch.no_grad():
                if slot.input_mode.get() == "audio" and slot.encoded_latents is not None:
                    total_latent_frames = slot.encoded_latents.shape[-1]

                    if slot.latent_position + slot.latent_length <= total_latent_frames:
                        z = slot.encoded_latents[:, :, slot.latent_position:slot.latent_position + slot.latent_length]
                        slot.latent_position += slot.latent_length
                    else:
                        if slot.loop_audio.get():
                            slot.latent_position = 0
                            z = slot.encoded_latents[:, :, slot.latent_position:slot.latent_position + slot.latent_length]
                            slot.latent_position += slot.latent_length
                        else:
                            self.schedule_ui_callback(lambda s=slot: deactivate_finished_audio_slot(self, s))
                            continue

                    z = z * slot.temperature.get()
                else:
                    z = torch.randn(1, slot.latent_size, slot.latent_length)
                    z = z * slot.random_intensity.get() * slot.temperature.get()

                if slot.prev_z is not None:
                    smooth = slot.smoothing.get()
                    z = smooth * slot.prev_z + (1 - smooth) * z

                slot.prev_z = z
                audio = slot.model.decode(z).cpu().numpy().flatten()

            decode_total_ms += (time.perf_counter() - decode_start) * 1000.0

            if len(audio) != self.chunk_samples:
                if len(audio) > self.chunk_samples:
                    audio = audio[:self.chunk_samples]
                else:
                    audio = np.pad(audio, (0, self.chunk_samples - len(audio)))

            slot.cached_audio = audio.astype(np.float32, copy=False)
            slot.last_decode_cycle = self._producer_cycle

        if slot.cached_audio is not None:
            mixed_audio += slot.cached_audio * slot.gain.get()

    if len(active_slots) > 1:
        max_val = np.abs(mixed_audio).max()
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val

    mixed_audio = np.clip(mixed_audio, -1.0, 1.0).astype(np.float32)
    self.metrics["decode_ms"].append(decode_total_ms)
    self._producer_cycle += 1
    return mixed_audio


def update_overload_state(self, producer_ms):
    chunk_budget_ms = (self.chunk_samples / self.sr) * 1000.0

    if producer_ms > chunk_budget_ms * 0.95:
        self._overload_cycles += 1
        self._stable_cycles = 0
    else:
        self._stable_cycles += 1
        self._overload_cycles = max(0, self._overload_cycles - 1)

    if self._overload_cycles >= 3 and self.decode_stride < self.max_decode_stride:
        self.decode_stride += 1
        self._overload_cycles = 0
        self.log(f"[PERF] Overload detected. Increasing decode stride to {self.decode_stride}.")

    if self._stable_cycles >= 25 and self.decode_stride > self.base_decode_stride:
        self.decode_stride -= 1
        self._stable_cycles = 0
        self.log(f"[PERF] Stable again. Decreasing decode stride to {self.decode_stride}.")


def log_metrics_snapshot(self):
    avg_prod = np.mean(self.metrics["producer_ms"]) if self.metrics["producer_ms"] else 0.0
    avg_decode = np.mean(self.metrics["decode_ms"]) if self.metrics["decode_ms"] else 0.0
    avg_write = np.mean(self.metrics["write_ms"]) if self.metrics["write_ms"] else 0.0
    budget_ms = (self.chunk_samples / self.sr) * 1000.0
    self.log(
        "[METRICS] "
        f"slots={len(self.get_active_slots())} | "
        f"budget={budget_ms:.1f}ms | prod={avg_prod:.1f}ms | decode={avg_decode:.1f}ms | "
        f"write={avg_write:.1f}ms | q={self.audio_queue.qsize()}/{self.queue_maxsize} | "
        f"underruns={self.metrics['underruns']} | dropped={self.metrics['dropped_chunks']}"
    )


def producer_loop(self):
    try:
        while self.is_running and not self.stop_event.is_set():
            loop_start = time.perf_counter()
            active_slots = self.get_active_slots()

            if not active_slots:
                chunk = np.zeros(self.chunk_samples, dtype=np.float32)
            else:
                chunk = generate_mixed_chunk(self, active_slots)

            producer_ms = (time.perf_counter() - loop_start) * 1000.0
            self.metrics["producer_ms"].append(producer_ms)
            self.metrics["generated_chunks"] += 1
            update_overload_state(self, producer_ms)

            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    _ = self.audio_queue.get_nowait()
                    self.metrics["dropped_chunks"] += 1
                except queue.Empty:
                    pass
                self.audio_queue.put_nowait(chunk)

            if self.audio_queue.qsize() >= self.queue_maxsize - 1:
                time.sleep((self.chunk_samples / self.sr) * 0.1)

    except Exception as e:
        self.log(f"ERROR in streaming: {str(e)}")
        self.is_running = False
        self.stop_event.set()


def consumer_loop(self):
    try:
        while self.is_running and not self.stop_event.is_set():
            timeout_s = (self.chunk_samples / self.sr) * 1.5
            try:
                chunk = self.audio_queue.get(timeout=timeout_s)
            except queue.Empty:
                self.metrics["underruns"] += 1
                chunk = self.last_good_chunk
                if chunk is None:
                    chunk = np.zeros(self.chunk_samples, dtype=np.float32)

            write_start = time.perf_counter()
            self.stream.write(chunk)
            write_ms = (time.perf_counter() - write_start) * 1000.0
            self.metrics["write_ms"].append(write_ms)
            self.metrics["played_chunks"] += 1
            self.last_good_chunk = chunk

            if self.metrics["played_chunks"] % 20 == 0:
                log_metrics_snapshot(self)

    except Exception as e:
        self.log(f"ERROR in audio output: {str(e)}")
        self.is_running = False
        self.stop_event.set()
