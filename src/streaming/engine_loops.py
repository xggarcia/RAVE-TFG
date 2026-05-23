import queue
import time

import numpy as np
import torch


def deactivate_finished_audio_slot(self, slot):
    slot.is_active = False
    slot.status_var = "Loaded (Audio Finished)"
    self.log(f"Slot {slot.slot_id + 1}: Audio playback finished")


def _project_subband_pattern_to_latent(pattern, latent_size: int, latent_length: int, step_index: int):
    """Map a painted (num_subbands, n_timesteps) pattern to a latent chunk.

    Each output frame samples one consecutive column of the pattern starting
    at ``step_index``, so a spike at column N produces a latent excursion as
    the playhead crosses column N.

    Each subband row drives a contiguous slice of the latent dimensions,
    weighted by row position (top band strongest, bottom band weakest).
    """
    if pattern is None:
        return torch.zeros(1, latent_size, latent_length)

    if torch.is_tensor(pattern):
        pattern_np = pattern.detach().cpu().float().numpy()
    else:
        pattern_np = np.asarray(pattern, dtype=np.float32)

    if pattern_np.ndim != 2 or latent_size <= 0 or latent_length <= 0:
        return torch.zeros(1, max(1, latent_size), max(1, latent_length))

    num_rows, num_steps = pattern_np.shape
    if num_rows == 0 or num_steps == 0:
        return torch.zeros(1, latent_size, latent_length)

    usable_rows = min(num_rows, latent_size)
    row_weights = np.linspace(1.0, 0.30, usable_rows, dtype=np.float32)
    dim_slices = np.array_split(np.arange(latent_size), usable_rows)

    step_idx = int(step_index) % num_steps
    window_indices = [(step_idx + i) % num_steps for i in range(latent_length)]
    step_window = np.clip(pattern_np[:, window_indices].astype(np.float32), 0.0, 1.0)
    excursion = step_window * 2.0

    z_np = np.zeros((latent_size, latent_length), dtype=np.float32)
    for row_idx in range(usable_rows):
        row_excursion = excursion[row_idx, :] * float(row_weights[row_idx])
        dims = dim_slices[row_idx]
        for k, dim in enumerate(dims):
            sign = 1.0 if ((k + row_idx) % 2 == 0) else -1.0
            z_np[dim, :] = row_excursion * sign

    return torch.from_numpy(z_np).unsqueeze(0)


def generate_mixed_chunk(self, active_slots):
    mixed_audio = np.zeros(self.chunk_samples, dtype=np.float32)
    decode_total_ms = 0.0
    for slot in active_slots:
        should_decode = (slot.cached_audio is None) or ((self._producer_cycle + slot.slot_id) % self.decode_stride == 0)

        if should_decode:
            decode_start = time.perf_counter()
            with torch.no_grad():
                if slot.input_mode == "audio" and slot.encoded_latents is None:
                    # File not loaded yet — output silence for this slot
                    slot.cached_audio = np.zeros(self.chunk_samples, dtype=np.float32)
                    continue

                if slot.input_mode == "audio" and slot.encoded_latents is not None:
                    total_latent_frames = slot.encoded_latents.shape[-1]

                    if slot.latent_position + slot.latent_length <= total_latent_frames:
                        z = slot.encoded_latents[:, :, slot.latent_position:slot.latent_position + slot.latent_length]
                        dry_pos = slot.audio_sample_pos
                        slot.latent_position += slot.latent_length
                        slot.audio_sample_pos += self.chunk_samples
                    else:
                        if slot.loop_audio:
                            slot.latent_position = 0
                            slot.audio_sample_pos = 0
                            z = slot.encoded_latents[:, :, 0:slot.latent_length]
                            dry_pos = 0
                            slot.latent_position += slot.latent_length
                            slot.audio_sample_pos += self.chunk_samples
                        else:
                            self.schedule_ui_callback(lambda s=slot: deactivate_finished_audio_slot(self, s))
                            continue

                else:
                    if slot.input_mode == "gesture":
                        pattern = slot.subband_pattern
                        if pattern is None:
                            z = torch.zeros(1, slot.latent_size, slot.latent_length)
                        else:
                            z = _project_subband_pattern_to_latent(
                                pattern, slot.latent_size, slot.latent_length, slot.subband_position
                            )
                            z = z * slot.subband_intensity
                            if pattern.shape[1] > 0:
                                slot.subband_position = (slot.subband_position + slot.latent_length) % pattern.shape[1]
                    else:
                        z = torch.randn(1, slot.latent_size, slot.latent_length)
                        z = z * slot.random_intensity

                z = z * slot.temperature

                map_anchor = getattr(slot, "phase_map_anchor", None)
                if map_anchor is not None:
                    from .phase_control import apply_phase_bias
                    z = apply_phase_bias(z, map_anchor["mean_z"], map_anchor["std_z"])

                if slot.prev_z is not None:
                    smooth = slot.smoothing
                    z = smooth * slot.prev_z + (1 - smooth) * z

                slot.prev_z = z

                global_bias = float(getattr(slot, "latent_global_bias", 0.0))
                if global_bias != 0.0:
                    z = z + global_bias
                # Per-dim latent scale
                scale_list = getattr(slot, "latent_scale", [])
                if scale_list:
                    z = z.clone()
                    for _dim in range(z.shape[1]):
                        _s = scale_list[_dim] if _dim < len(scale_list) else 1.0
                        if _s != 1.0:
                            z[:, _dim, :] = z[:, _dim, :] * _s

                audio = slot.model.decode(z).cpu().numpy().flatten()

            decode_total_ms += (time.perf_counter() - decode_start) * 1000.0
            if len(audio) != self.chunk_samples:
                if len(audio) > self.chunk_samples:
                    audio = audio[:self.chunk_samples]
                else:
                    audio = np.pad(audio, (0, self.chunk_samples - len(audio)))

            audio = audio.astype(np.float32, copy=False)
            # Dry/wet blend — only meaningful in audio mode with raw audio available
            if slot.input_mode == "audio" and slot.raw_audio is not None:
                w = float(slot.dry_wet)
                raw = slot.raw_audio
                n = len(audio)
                end = dry_pos + n
                if end <= len(raw):
                    dry_chunk = raw[dry_pos:end]
                else:
                    # loop-wrap
                    part1 = raw[dry_pos:]
                    part2 = raw[: end - len(raw)]
                    dry_chunk = np.concatenate([part1, part2])
                if len(dry_chunk) < n:
                    dry_chunk = np.pad(dry_chunk, (0, n - len(dry_chunk)))
                audio = (1.0 - w) * dry_chunk + w * audio
            slot.cached_audio = audio
            slot.last_decode_cycle = self._producer_cycle
        if slot.cached_audio is not None:
            mixed_audio += slot.cached_audio * slot.gain

    if len(active_slots) > 1:
        max_val = np.abs(mixed_audio).max()
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val

    mixed_audio = np.clip(mixed_audio, -1.0, 1.0).astype(np.float32)
    self.last_decode_ms = decode_total_ms
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

    if not self.dynamic_stride_enabled:
        # Keep stride at base; user disabled adaptive behavior.
        if self.decode_stride != self.base_decode_stride:
            self.decode_stride = self.base_decode_stride
        return

    if self._overload_cycles >= 3 and self.decode_stride < self.max_decode_stride:
        self.decode_stride += 1
        self._overload_cycles = 0
        self.log(f"[PERF] Overload detected. Increasing decode stride to {self.decode_stride}.")

    if self._stable_cycles >= 25 and self.decode_stride > self.base_decode_stride:
        self.decode_stride -= 1
        self._stable_cycles = 0
        self.log(f"[PERF] Stable again. Decreasing decode stride to {self.decode_stride}.")


def log_metrics_snapshot(self):
    budget_ms = (self.chunk_samples / self.sr) * 1000.0
    self.log(
        "[METRICS] "
        f"slots={len(self.get_active_slots())} | "
        f"budget={budget_ms:.1f}ms | prod={self.last_producer_ms:.1f}ms | "
        f"decode={self.last_decode_ms:.1f}ms | write={self.last_write_ms:.1f}ms | "
        f"q={self.audio_queue.qsize()}/{self.queue_maxsize} | "
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
                chunk = (chunk * getattr(self, "master_volume", 1.0)).astype(np.float32)

            producer_ms = (time.perf_counter() - loop_start) * 1000.0
            self.last_producer_ms = producer_ms
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
                chunk = self.get_last_output_chunk()
                if chunk is None:
                    chunk = np.zeros(self.chunk_samples, dtype=np.float32)

            write_start = time.perf_counter()
            self.stream.write(chunk)
            self.last_write_ms = (time.perf_counter() - write_start) * 1000.0
            self.metrics["played_chunks"] += 1
            self.set_last_output_chunk(chunk)

            cb = getattr(self, "on_chunk_played", None)
            if callable(cb):
                try:
                    cb(chunk)
                except Exception:
                    pass

            if self.metrics["played_chunks"] % 20 == 0:
                log_metrics_snapshot(self)

    except Exception as e:
        self.log(f"ERROR in audio output: {str(e)}")
        self.is_running = False
        self.stop_event.set()
