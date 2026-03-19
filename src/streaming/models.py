import tkinter as tk


class ModelSlot:
    """Represents a single model slot with its own parameters and state."""

    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.model = None
        self.model_path = None
        self.is_active = tk.BooleanVar(value=False)
        self.is_loaded = False

        # Model-specific dimensions
        self.latent_size = None
        self.latent_length = None
        self.output_length = None
        self.prev_z = None

        # Individual parameters
        self.gain = tk.DoubleVar(value=0.5)
        self.temperature = tk.DoubleVar(value=1.0)
        self.smoothing = tk.DoubleVar(value=0.0)

        # Input source parameters
        self.input_mode = tk.StringVar(value="random")
        self.audio_file_path = None
        self.encoded_latents = None
        self.latent_position = 0
        self.loop_audio = tk.BooleanVar(value=True)
        self.random_intensity = tk.DoubleVar(value=1.0)

        # Streaming cache for load shedding
        self.cached_audio = None
        self.last_decode_cycle = -1

        # UI references
        self.model_var = tk.StringVar(value="[No Model Loaded]")
        self.status_var = tk.StringVar(value="Inactive")
