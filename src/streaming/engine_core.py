import collections
import queue
import threading

from .engine_loops import consumer_loop, producer_loop


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
        self.base_decode_stride = 1
        self.max_decode_stride = 2
        self.decode_stride = 1
        self.queue_maxsize = 6
        self._producer_cycle = 0
        self._overload_cycles = 0
        self._stable_cycles = 0
        self.last_good_chunk = None
        self._last_chunk_lock = threading.Lock()
        self.on_chunk_played = None

        self.metrics = {
            "generated_chunks": 0,
            "played_chunks": 0,
            "underruns": 0,
            "dropped_chunks": 0,
            "producer_ms": collections.deque(maxlen=120),
            "decode_ms": collections.deque(maxlen=120),
            "write_ms": collections.deque(maxlen=120),
        }

    @staticmethod
    def profile_for_mode(mode):
        if mode == "Quality":
            return {"queue_size": 4, "base_stride": 1, "max_stride": 2}
        if mode == "Max Stability":
            return {"queue_size": 10, "base_stride": 2, "max_stride": 6}
        return {"queue_size": 6, "base_stride": 1, "max_stride": 4}

    def configure(self, stream, sr, chunk_samples, performance_mode, fixed_stride: int | None = None):
        self.stream = stream
        self.sr = sr
        self.chunk_samples = chunk_samples

        profile = self.profile_for_mode(performance_mode)
        self.queue_maxsize = profile["queue_size"]
        if fixed_stride is not None:
            self.base_decode_stride = fixed_stride
            self.max_decode_stride = fixed_stride
        else:
            self.base_decode_stride = profile["base_stride"]
            self.max_decode_stride = profile["max_stride"]
        self.decode_stride = self.base_decode_stride

        self.audio_queue = queue.Queue(maxsize=self.queue_maxsize)
        self.stop_event.clear()
        self._reset_metrics()

    def _reset_metrics(self):
        self.metrics["generated_chunks"] = 0
        self.metrics["played_chunks"] = 0
        self.metrics["underruns"] = 0
        self.metrics["dropped_chunks"] = 0
        self.metrics["producer_ms"].clear()
        self.metrics["decode_ms"].clear()
        self.metrics["write_ms"].clear()
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
