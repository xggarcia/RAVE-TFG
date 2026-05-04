import queue
import random
import time

import numpy as np
import torch


def deactivate_finished_audio_slot(self, slot):
    slot.is_active.set(False)
    slot.status_var.set("Loaded (Audio Finished)")
    self.log(f"Slot {slot.slot_id + 1}: Audio playback finished")


def deactivate_failed_prior_slot(self, slot, reason):
    slot.use_prior.set(False)
    slot.prior_needs_warmup = True
    slot.prior_chunks_generated = 0
    if slot.is_active.get():
        slot.status_var.set("Active (Prior Off)")
    else:
        slot.status_var.set("Loaded (Prior Off)")
    self.log(f"Slot {slot.slot_id + 1}: Prior generation failed, fallback to source mode - {reason}")


def coerce_prior_latent(z, target_channels, target_frames):
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

    if z.shape[1] != target_channels and z.shape[2] == target_channels:
        z = z.transpose(1, 2)

    if z.shape[1] != target_channels:
        raise ValueError(f"Prior channels mismatch: got {z.shape[1]}, expected {target_channels}")

    if z.shape[2] > target_frames:
        z = z[:, :, :target_frames]
    elif z.shape[2] < target_frames:
        z = torch.nn.functional.pad(z, (0, target_frames - z.shape[2]))

    return z


def _interp_curve(curve: list, t_norm: float) -> float:
    if not curve:
        return 0.5
    if t_norm <= curve[0][0]:
        return curve[0][1]
    for i in range(1, len(curve)):
        x0, y0 = curve[i - 1]
        x1, y1 = curve[i]
        if t_norm <= x1:
            if x1 <= x0:
                return y1
            a = (t_norm - x0) / (x1 - x0)
            return y0 + a * (y1 - y0)
    return curve[-1][1]


def _sample_curve_at(curve: list, n: int):
    import numpy as _np
    if not curve or n <= 0:
        return _np.zeros(n, dtype=_np.float32)
    xs = _np.linspace(0.0, 1.0, n)
    ys = [_interp_curve(curve, float(x)) for x in xs]
    return _np.array(ys, dtype=_np.float32)


def _project_subband_pattern_to_latent(pattern, latent_size: int, latent_length: int, step_index: int):
    """Map a painted (num_subbands, n_timesteps) pattern to a latent chunk.

    Each of the ``latent_length`` output frames samples one consecutive column
    of the pattern starting at ``step_index``. This makes the timeline 1:1: a
    spike painted at column N produces a latent excursion exactly when the
    playhead crosses column N.

    Each subband row drives a contiguous slice of the latent dimensions,
    weighted by ``row_weights`` (top band strongest, bottom band weakest).
    A faint texture per row breaks the all-DC excursion that would otherwise
    sound monotonous, but the painted curve is the dominant signal.
    """
    import numpy as _np
    import torch as _torch

    if pattern is None:
        return _torch.zeros(1, latent_size, latent_length)

    if torch.is_tensor(pattern):
        pattern = pattern.detach().cpu().float().numpy()
    else:
        pattern = _np.asarray(pattern, dtype=_np.float32)

    if pattern.ndim != 2 or latent_size <= 0 or latent_length <= 0:
        return _torch.zeros(1, max(1, latent_size), max(1, latent_length))

    num_rows, num_steps = pattern.shape
    if num_rows == 0 or num_steps == 0:
        return _torch.zeros(1, latent_size, latent_length)

    usable_rows = min(num_rows, latent_size)
    # Weights match the visual band heights so what looks dominant sounds dominant.
    row_weights = _np.array([1.0, 0.78, 0.58, 0.42, 0.30], dtype=_np.float32)
    if usable_rows != row_weights.size:
        row_weights = _np.linspace(1.0, 0.30, usable_rows, dtype=_np.float32)
    else:
        row_weights = row_weights[:usable_rows]
    dim_slices = _np.array_split(_np.arange(latent_size), usable_rows)

    step_idx = int(step_index) % num_steps
    window_indices = [(step_idx + i) % num_steps for i in range(latent_length)]
    # step_window[row, frame] is the painted value at that (band, time) pair.
    step_window = _np.clip(pattern[:, window_indices].astype(_np.float32), 0.0, 1.0)

    # painted=0 -> 0 (band silent), painted=1 -> +2 (strong).
    # Per-dim alternating sign below spreads the excursion through latent space
    # so a fully-painted row produces timbral movement rather than flat DC.
    excursion = step_window * 2.0

    z_np = _np.zeros((latent_size, latent_length), dtype=_np.float32)
    for row_idx in range(usable_rows):
        row_excursion = excursion[row_idx, :] * float(row_weights[row_idx])
        dims = dim_slices[row_idx]
        for k, dim in enumerate(dims):
            sign = 1.0 if ((k + row_idx) % 2 == 0) else -1.0
            z_np[dim, :] = row_excursion * sign

    return _torch.from_numpy(z_np).unsqueeze(0)


def _project_curve_to_latent(curve: list, latent_size: int, latent_length: int, seed: int = 0, intensity: float = 1.0):
    import numpy as _np
    import torch as _torch
    base = _sample_curve_at(curve, latent_length)
    # Apply intensity scaling to the base curve (makes it more/less dramatic)
    base = base * intensity
    # deterministic per-seed random scales (more dramatic projection)
    rng = _np.random.RandomState(seed)
    # Generate scales: larger variations for more noticeable effect
    # Use a mix of base offset + variable amplitude per dimension
    dim_centers = rng.uniform(-0.5, 0.5, size=(latent_size,)).astype(_np.float32)  # per-dim center
    dim_scales = rng.uniform(0.8, 2.5, size=(latent_size,)).astype(_np.float32)  # per-dim amplitude (0.8-2.5x)
    
    # Project curve to latent space: each dimension gets a scaled + offset version of the curve
    z_np = _np.zeros((latent_size, latent_length), dtype=_np.float32)
    for d in range(latent_size):
        # Curve drives both amplitude and offset
        z_np[d, :] = (base * dim_scales[d]) + dim_centers[d]
    
    z = _torch.from_numpy(z_np).unsqueeze(0)  # (1, latent_size, latent_length)
    return z


def generate_mixed_chunk(self, active_slots):
    mixed_audio = np.zeros(self.chunk_samples, dtype=np.float32)
    decode_total_ms = 0.0
    for slot in active_slots:
        should_decode = (slot.cached_audio is None) or ((self._producer_cycle + slot.slot_id) % self.decode_stride == 0)

        if should_decode:
            decode_start = time.perf_counter()
            prior_chunk_used = False
            with torch.no_grad():
                if slot.input_mode.get() == "audio" and slot.encoded_latents is None:
                    # File not loaded yet — output silence for this slot
                    slot.cached_audio = np.zeros(self.chunk_samples, dtype=np.float32)
                    continue

                if slot.input_mode.get() == "audio" and slot.encoded_latents is not None:
                    total_latent_frames = slot.encoded_latents.shape[-1]

                    if slot.latent_position + slot.latent_length <= total_latent_frames:
                        z = slot.encoded_latents[:, :, slot.latent_position:slot.latent_position + slot.latent_length]
                        dry_pos = slot.audio_sample_pos
                        slot.latent_position += slot.latent_length
                        slot.audio_sample_pos += self.chunk_samples
                    else:
                        if slot.loop_audio.get():
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
                    if slot.input_mode.get() == "gesture":
                        pattern = slot.subband_pattern
                        if pattern is None:
                            z = torch.zeros(1, slot.latent_size, slot.latent_length)
                        else:
                            z = _project_subband_pattern_to_latent(
                                pattern, slot.latent_size, slot.latent_length, slot.subband_position
                            )
                            z = z * slot.subband_intensity.get()
                            if pattern.shape[1] > 0:
                                slot.subband_position = (slot.subband_position + slot.latent_length) % pattern.shape[1]
                    else:
                        density = slot.density.get()
                        if slot.held_z is None or random.random() < density:
                            z = torch.randn(1, slot.latent_size, slot.latent_length)
                            z = z * slot.random_intensity.get()
                            slot.held_z = z.clone()
                            slot.density_hold_frames = 0
                        else:
                            slot.density_hold_frames += 1
                            fade = max(0.0, 1.0 - slot.density_hold_frames * 0.12)
                            z = slot.held_z * fade
                
                z = z * slot.temperature.get()

                map_anchor = getattr(slot, "phase_map_anchor", None)
                if map_anchor is not None:
                    from .phase_control import apply_phase_bias
                    z = apply_phase_bias(z, map_anchor["mean_z"], map_anchor["std_z"])

                if slot.use_prior.get() and (slot.prior_model is not None or slot.embedded_prior_available):
                    try:
                        if slot.prior_model is not None:
                            prior_model = slot.prior_model
                            seed_channels = slot.prior_seed_channels if slot.prior_seed_channels in (1, 1024) else 1
                        else:
                            prior_model = slot.model
                            seed_channels = (
                                slot.embedded_prior_seed_channels if slot.embedded_prior_seed_channels in (1, 1024) else 1
                            )

                        if slot.prior_needs_warmup:
                            warmup_length = max(slot.latent_length * 4, slot.latent_length + 64)
                            warmup_seed_candidates = []

                            # Candidate 1: source-conditioned seed when channel layout allows it.
                            if seed_channels == slot.latent_size:
                                repeats = max(2, int(np.ceil(warmup_length / slot.latent_length)))
                                warmup_seed_candidates.append(z.repeat(1, 1, repeats)[:, :, :warmup_length])

                            # Candidate 2: robust zero seed fallback.
                            warmup_seed_candidates.append(torch.zeros(1, seed_channels, warmup_length).float())

                            z_warm = None
                            warmup_error = None
                            for seed in warmup_seed_candidates:
                                try:
                                    z_prior = prior_model.prior(seed)
                                    z_warm = coerce_prior_latent(z_prior, slot.latent_size, warmup_length)
                                    break
                                except Exception as exc:
                                    warmup_error = exc

                            if z_warm is None:
                                raise warmup_error if warmup_error is not None else RuntimeError("Prior warmup failed")

                            z = z_warm[:, :, -slot.latent_length:]
                            slot.prior_needs_warmup = False
                            slot.prior_chunks_generated = 0
                        else:
                            seed_candidates = []
                            if seed_channels == slot.latent_size:
                                seed_candidates.append(z)
                            seed_candidates.append(torch.zeros(1, seed_channels, slot.latent_length).float())

                            z_out = None
                            prior_error = None
                            for seed in seed_candidates:
                                try:
                                    z_prior = prior_model.prior(seed)
                                    z_out = coerce_prior_latent(z_prior, slot.latent_size, slot.latent_length)
                                    break
                                except Exception as exc:
                                    prior_error = exc

                            if z_out is None:
                                raise prior_error if prior_error is not None else RuntimeError("Prior inference failed")

                            z = z_out

                        z = z * slot.prior_temperature.get()
                        prior_chunk_used = True
                    except Exception as prior_exc:
                        self.schedule_ui_callback(lambda s=slot, msg=str(prior_exc): deactivate_failed_prior_slot(self, s, msg))
                        # Keep source latent z for this chunk and continue streaming.
                if slot.prev_z is not None:
                    smooth = slot.smoothing.get()
                    z = smooth * slot.prev_z + (1 - smooth) * z

                slot.prev_z = z

                global_bias_var = getattr(slot, "latent_global_bias", None)
                gesture_bias_var = getattr(slot, "gesture_bias", None)
                global_bias = float(global_bias_var.get()) if global_bias_var is not None else 0.0
                gesture_bias = float(gesture_bias_var.get()) if gesture_bias_var is not None else 0.0
                total_bias = global_bias + gesture_bias
                if total_bias != 0.0:
                    z = z + total_bias
                # Per-dim latent controls (bias / scale)
                bias_list  = getattr(slot, "latent_bias",  [])
                scale_list = getattr(slot, "latent_scale", [])
                if bias_list or scale_list:
                    z = z.clone()
                    for _dim in range(z.shape[1]):
                        _b = bias_list[_dim]  if _dim < len(bias_list)  else 0.0
                        _s = scale_list[_dim] if _dim < len(scale_list) else 1.0
                        if _b != 0.0 or _s != 1.0:
                            z[:, _dim, :] = z[:, _dim, :] * _s + _b

                audio = slot.model.decode(z).cpu().numpy().flatten()

            if prior_chunk_used and slot.prior_chunks_generated == 0:
                fade = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
                audio = audio * fade

            if prior_chunk_used:
                slot.prior_chunks_generated += 1
            decode_total_ms += (time.perf_counter() - decode_start) * 1000.0
            if len(audio) != self.chunk_samples:
                if len(audio) > self.chunk_samples:
                    audio = audio[:self.chunk_samples]
                else:
                    audio = np.pad(audio, (0, self.chunk_samples - len(audio)))

            audio = audio.astype(np.float32, copy=False)
            # Dry/wet blend — only meaningful in audio mode with raw audio available
            if slot.input_mode.get() == "audio" and slot.raw_audio is not None:
                w = float(slot.dry_wet.get())
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
                chunk = (chunk * getattr(self, "master_volume", 1.0)).astype(np.float32)

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
                chunk = self.get_last_output_chunk()
                if chunk is None:
                    chunk = np.zeros(self.chunk_samples, dtype=np.float32)

            write_start = time.perf_counter()
            self.stream.write(chunk)
            write_ms = (time.perf_counter() - write_start) * 1000.0
            self.metrics["write_ms"].append(write_ms)
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
