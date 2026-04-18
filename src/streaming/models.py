try:
    import tkinter as tk
    _TK_AVAILABLE = True
except ModuleNotFoundError:
    _TK_AVAILABLE = False


class _Var:
    """Fallback for _BooleanVar/DoubleVar/StringVar on headless servers."""
    def __init__(self, value=None):
        self._value = value
    def get(self):
        return self._value
    def set(self, value):
        self._value = value


def _BooleanVar(value=False):
    return _BooleanVar(value=value) if _TK_AVAILABLE else _Var(value)

def _DoubleVar(value=0.0):
    return _DoubleVar(value=value) if _TK_AVAILABLE else _Var(value)

def _StringVar(value=""):
    return _StringVar(value=value) if _TK_AVAILABLE else _Var(value)


class ModelSlot:
    """Represents a single model slot with its own parameters and state."""

    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.model = None
        self.model_path = None
        self.model_sr = None
        self.is_active = _BooleanVar(value=False)
        self.is_loaded = False

        # Model-specific dimensions
        self.latent_size = None
        self.latent_length = None
        self.output_length = None
        self.prev_z = None

        # Individual parameters
        self.gain = _DoubleVar(value=0.5)
        self.temperature = _DoubleVar(value=1.0)
        self.smoothing = _DoubleVar(value=0.0)

        # Input source parameters
        self.input_mode = _StringVar(value="random")
        self.audio_file_path = None
        self.encoded_latents = None
        self.latent_position = 0
        self.loop_audio = _BooleanVar(value=True)
        self.random_intensity = _DoubleVar(value=1.0)
        self.density = _DoubleVar(value=1.0)
        self.held_z = None
        self.density_hold_frames = 0

        # Phase interpolation control
        self.phase_enabled = _BooleanVar(value=False)
        self.phase_value = _DoubleVar(value=0.0)
        self.phase_anchors = []

        # Prior generation parameters/state
        self.use_prior = _BooleanVar(value=False)
        self.prior_model = None
        self.prior_model_path = None
        self.prior_seed_channels = None
        self.embedded_prior_available = False
        self.embedded_prior_seed_channels = None
        self.prior_temperature = _DoubleVar(value=1.0)
        self.prior_status_var = _StringVar(value="No Prior Loaded")
        self.prior_needs_warmup = True
        self.prior_chunks_generated = 0

        # Streaming cache for load shedding
        self.cached_audio = None
        self.last_decode_cycle = -1

        # UI references
        self.model_var = _StringVar(value="[No Model Loaded]")
        self.status_var = _StringVar(value="Inactive")
