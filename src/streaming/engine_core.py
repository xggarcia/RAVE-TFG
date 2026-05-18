import queue
import threading

from .engine_loops import consumer_loop, producer_loop


# Engine defaults. These used to be selected by a "performance_mode" preset
# (Quality / Balanced / Max Stability). Now only one configuration exists and
# dynamic stride is what adapts to load at runtime — toggleable from the UI.
DEFAULT_QUEUE_SIZE = 6
DEFAULT_BASE_STRIDE = 1
DEFAULT_MAX_STRIDE = 4


class StreamingEngine:
    """Realtime producer/consumer engine decoupled from GUI widgets."""

    def __init__(self, logger, get_active_slots, schedule_ui_callback):
        self.log = logger
        self.get_active_slots = get_active_slots
        self.schedule_ui_callback = schedule_ui_callback

        self.stream = None
        self.sr = 44100
        self.chunk_samples = None

        self.is_running = False
        self.producer_thread = None
        self.consumer_thread = None
        self.stop_event = threading.Event()
        self.audio_queue = None

        self.master_volume = 1.0
        self.base_decode_stride = DEFAULT_BASE_STRIDE
        self.max_decode_stride = DEFAULT_MAX_STRIDE
        self.decode_stride = DEFAULT_BASE_STRIDE
        self.queue_maxsize = DEFAULT_QUEUE_SIZE
        self.dynamic_stride_enabled = True

        self._producer_cycle = 0
        self._overload_cycles = 0
        self._stable_cycles = 0
        self.last_good_chunk = None
        self._last_chunk_lock = threading.Lock()
        self.on_chunk_played = None

        # Aggregate counters and the most recent per-stage timing.
        # We used to keep 120-sample deques per stage but the only place that
        # consumed them showed the most recent value anyway.
        self.metrics = {
            "generated_chunks": 0,
            "played_chunks": 0,
            "underruns": 0,
            "dropped_chunks": 0,
        }
        self.last_producer_ms = 0.0
        self.last_decode_ms = 0.0
        self.last_write_ms = 0.0

    def configure(self, stream, sr, chunk_samples):
        self.stream = stream
        self.sr = sr
        self.chunk_samples = chunk_samples

        self.queue_maxsize = DEFAULT_QUEUE_SIZE
        self.base_decode_stride = DEFAULT_BASE_STRIDE
        self.max_decode_stride = DEFAULT_MAX_STRIDE
        self.decode_stride = self.base_decode_stride

        self.audio_queue = queue.Queue(maxsize=self.queue_maxsize)
        self.stop_event.clear()
        self._reset_metrics()

    def _reset_metrics(self):
        self.metrics["generated_chunks"] = 0
        self.metrics["played_chunks"] = 0
        self.metrics["underruns"] = 0
        self.metrics["dropped_chunks"] = 0
        self.last_producer_ms = 0.0
        self.last_decode_ms = 0.0
        self.last_write_ms = 0.0
        self._producer_cycle = 0
        self._overload_cycles = 0
        self._stable_cycles = 0
        self.last_good_chunk = None

    def set_last_output_chunk(self, chunk):
        with self._last_chunk_lock:
            self.last_good_chunk = chunk

    def get_last_output_chunk(self):
        with self._last_chunk_lock:
            if self.last_good_chunk is None:
                return None
            return self.last_good_chunk.copy()

    def start(self):
        self.is_running = True
        self.producer_thread = threading.Thread(target=producer_loop, args=(self,), daemon=True)
        self.consumer_thread = threading.Thread(target=consumer_loop, args=(self,), daemon=True)
        self.producer_thread.start()
        self.consumer_thread.start()

    def stop(self):
        self.is_running = False
        self.stop_event.set()

        if self.producer_thread:
            self.producer_thread.join(timeout=2.0)
            self.producer_thread = None

        if self.consumer_thread:
            self.consumer_thread.join(timeout=2.0)
            self.consumer_thread = None

        self.audio_queue = None
