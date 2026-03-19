import tkinter as tk

from src.streaming import QuickCalibrator, StreamingEngine

from .model_io import StreamGUIModelIOMixin
from .runtime import StreamGUIRuntimeMixin
from .slots import StreamGUISlotsMixin
from .ui import StreamGUIUIMixin


class RAVEStreamGUI(StreamGUIUIMixin, StreamGUISlotsMixin, StreamGUIModelIOMixin, StreamGUIRuntimeMixin):
    """Composed GUI controller with focused mixins by concern."""

    def __init__(self, root):
        self.root = root
        self.root.title("RAVE Multi-Model Streaming")
        self.root.geometry("1000x850")
        self.root.resizable(True, True)

        # Streaming state
        self.is_streaming = False
        self.stream = None
        self.stream_thread = None

        # Global audio settings
        self.sr = tk.IntVar(value=44100)
        self.chunk_duration = tk.DoubleVar(value=1.0)
        self.chunk_samples = None
        self.performance_mode = tk.StringVar(value="Balanced")
        self.auto_calibrate_on_start = tk.BooleanVar(value=True)
        self.calibration_summary = tk.StringVar(value="Calibration: not run")
        self.last_calibration_signature = None
        self.last_calibration_result = None

        # Dynamic model slots
        self.model_slots = []
        self.next_slot_id = 0
        self.slots_container = None

        # Available models list
        self.available_models = []

        # Service objects
        self.calibrator = QuickCalibrator(logger=self.log_status, validate_slots=self._validate_active_slots)
        self.engine = StreamingEngine(
            logger=self.log_status,
            get_active_slots=self._get_active_loaded_slots,
            schedule_ui_callback=lambda fn: self.root.after(0, fn),
        )

        self.create_widgets()
        self.discover_models()
