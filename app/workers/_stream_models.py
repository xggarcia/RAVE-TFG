"""Data models and helpers shared by StreamWorker."""
import re


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

        # Subband control: grid of (num_subbands, n_timesteps) with 0.0 or 1.0
        self.subband_pattern = None  # torch tensor shaped (num_subbands, n_timesteps)
        self.subband_position = 0    # current timestep in pattern (loopback)
        self.subband_intensity = _Var(1.0)  # global control: 0.0=muted, 1.0=full

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
