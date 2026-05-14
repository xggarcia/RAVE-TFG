from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
import torch

from app.ui.widgets.form import (
    AMBER,
    BG0,
    BG1,
    FG2,
    LINE0,
    PageHeader,
)
from app.ui.widgets.vu import VUMeter
from app.ui.pages._stream_builders import _StreamBuildersMixin
from app.ui.pages._stream_handlers import _StreamHandlersMixin
from app.workers.stream_worker import StreamWorker


class StreamPage(QWidget, _StreamBuildersMixin, _StreamHandlersMixin):
    statusChanged = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: StreamWorker | None = None
        self._models: list[str] = []
        self._slot_assignments: list[str | None] = [None, None]
        self._slot_anchors: list[str | None] = [None, None]
        self._state = "empty"  # empty|loading|live|error
        self._stage_rows: list = []
        self._slot_widgets: list = []
        self._master_vus: tuple[VUMeter, VUMeter] | None = None
        self._latency_lbl = None
        self._selected_slot: int = -1
        self._adv_panel = None
        self._slot_latent_sizes: list[int | None] = [None, None]
        self._slot_adv_state: list[dict] = [self._default_adv_state(), self._default_adv_state()]
        self._slot_param_state: list[dict] = [self._default_slot_params(), self._default_slot_params()]
        # One painted pattern per slot; widget shows the selected slot's pattern.
        self._slot_subband_patterns: list[torch.Tensor | None] = [None, None]
        self._slot_subband_positions: list[int] = [0, 0]
        self._profiler = None
        self._recording_active = False
        self._gesture_intensity = 1.0  # Subband intensity: 0.0-1.0
        self._fixed_stride: int | None = None
        self._stride_combo: QComboBox | None = None
        self._build()
        self._refresh_models()

    @staticmethod
    def _default_adv_state(n_dims: int = 8) -> dict:
        return {"prior_on": False, "scales": [1.0] * n_dims, "active_dim": 0}

    @staticmethod
    def _default_slot_params() -> dict:
        return {"gain": 0.65, "temp": 0.52, "smooth": 0.40, "dry_wet": 1.0,
                "phase": 0.0, "noise": 1.0, "bias": 0.0}

    def _ensure_slot_lists(self, target_len: int):
        while len(self._slot_latent_sizes) < target_len:
            self._slot_latent_sizes.append(None)
        while len(self._slot_adv_state) < target_len:
            self._slot_adv_state.append(self._default_adv_state())
        while len(self._slot_param_state) < target_len:
            self._slot_param_state.append(self._default_slot_params())
        while len(self._slot_subband_patterns) < target_len:
            self._slot_subband_patterns.append(None)
        while len(self._slot_subband_positions) < target_len:
            self._slot_subband_positions.append(0)

    def _resize_adv_state(self, slot_idx: int, n_dims: int):
        if not (0 <= slot_idx < len(self._slot_adv_state)):
            return
        n_dims = max(1, min(8, n_dims))
        state = self._slot_adv_state[slot_idx]
        old_scales = list(state.get("scales", []))
        state["scales"] = [old_scales[i] if i < len(old_scales) else 1.0 for i in range(n_dims)]
        state["active_dim"] = max(0, min(n_dims - 1, int(state.get("active_dim", 0))))

    def _slot_is_loaded(self, index: int) -> bool:
        if not (0 <= index < len(self._slot_assignments)):
            return False
        return self._slot_assignments[index] is not None and self._slot_latent_sizes[index] is not None

    def _on_stride_changed(self, index: int):
        self._fixed_stride = self._stride_combo.itemData(index)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._start_btn = QPushButton("Start stream")
        self._start_btn.setProperty("role", "primary")
        self._start_btn.setFixedHeight(30)
        self._start_btn.clicked.connect(self._start_stream)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("role", "danger")
        self._stop_btn.setFixedHeight(30)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_stream)

        self._scan_btn = QPushButton("Refresh models")
        self._scan_btn.setFixedHeight(30)
        self._scan_btn.clicked.connect(self._refresh_models)

        self._record_btn = QPushButton("Start rec")
        self._record_btn.setFixedHeight(30)
        self._record_btn.setEnabled(False)
        self._record_btn.clicked.connect(self._toggle_recording)

        self._stride_combo = QComboBox()
        self._stride_combo.setFixedHeight(30)
        self._stride_combo.setToolTip("Stride fijo para EXP-06 (Adaptive = comportamiento normal)")
        for label, value in [("Adaptive", None), ("Stride 1", 1), ("Stride 2", 2), ("Stride 4", 4)]:
            self._stride_combo.addItem(label, value)
        self._stride_combo.currentIndexChanged.connect(self._on_stride_changed)

        root.addWidget(
            PageHeader(
                crumbs=["Generate & Stream", "Streaming GUI"],
                title="Multi-model streaming",
                desc="Realtime phase interpolation between RAVE models.",
                actions=[self._scan_btn, self._stride_combo, self._record_btn, self._stop_btn, self._start_btn],
            )
        )

        self._stack_wrap = QScrollArea()
        self._stack_wrap.setWidgetResizable(True)
        self._stack_wrap.setFrameShape(QFrame.NoFrame)
        self._stack_wrap.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(24, 24, 24, 24)
        self._content_layout.setSpacing(18)

        self._stack_wrap.setWidget(self._content)
        root.addWidget(self._stack_wrap, 1)

    def _clear_content(self):
        self._slot_widgets = []
        self._master_vus = None
        self._latency_lbl = None
        self._adv_panel = None
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _ensure_worker(self):
        """Create and wire up the StreamWorker if it doesn't exist yet."""
        if self._worker is not None:
            return
        from app.workers.stream_worker import StreamWorker
        n = max(len(self._slot_assignments), 1)
        self._worker = StreamWorker([None] * n, fixed_stride=self._fixed_stride)
        self._worker.stage.connect(self._on_stage)
        self._worker.log.connect(self._on_log)
        self._worker.slotVu.connect(self._on_slot_vu)
        self._worker.slotSpectrogram.connect(self._on_slot_spectrogram)
        self._worker.slotInfo.connect(self._on_slot_info)
        self._worker.slotSubbandPosition.connect(self._on_subband_position)
        self._worker.masterVu.connect(self._on_master_vu)
        self._worker.warning.connect(self._on_warning)
        self._worker.running.connect(self._on_running)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)

    def _is_streaming(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _refresh_models(self):
        from app._paths import get_repo_root
        root = get_repo_root()
        candidates = []
        for folder in [root / "models" / "user_model" / "exported_model", root / "models" / "demo_model"]:
            if folder.exists():
                candidates.extend(str(p) for p in folder.glob("*.ts"))
        self._models = sorted(candidates)
        self._set_live_preview_state(streaming=self._is_streaming())

    def _make_add_slot_btn(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{BG1}; border:1px dashed {LINE0}; border-radius:6px;")
        w.setMinimumHeight(80)
        lay = QVBoxLayout(w)
        lay.setAlignment(Qt.AlignCenter)
        btn = QPushButton("+  Add slot")
        btn.setFixedSize(110, 34)
        btn.clicked.connect(self._add_slot)
        lay.addWidget(btn, alignment=Qt.AlignCenter)
        return w

    def _add_slot(self):
        self._slot_assignments.append(None)
        self._slot_anchors.append(None)
        self._slot_latent_sizes.append(None)
        self._slot_adv_state.append(self._default_adv_state())
        self._slot_param_state.append(self._default_slot_params())
        self._slot_subband_patterns.append(None)
        self._ensure_worker()
        self._worker.add_slot()
        self._set_live_preview_state(streaming=self._is_streaming())

    def _stop_stream(self):
        if self._worker:
            if self._recording_active:
                self._toggle_recording()
            self._worker.stop()
            self.statusChanged.emit("Stopping stream...", AMBER)

    def _clear_subband(self):
        if hasattr(self, "_gesture_widget"):
            self._gesture_widget.clear()

    def _invert_subband(self):
        if hasattr(self, "_gesture_widget"):
            self._gesture_widget.invert()

    @Slot(torch.Tensor)
    def _on_subband_pattern_changed(self, pattern: torch.Tensor):
        idx = self._selected_slot
        # Persist pattern in the per-slot store and push only to that slot.
        if 0 <= idx < len(self._slot_subband_patterns):
            self._slot_subband_patterns[idx] = pattern.clone()
        if self._worker and 0 <= idx < len(self._slot_assignments):
            self._worker.set_slot_subband_pattern(idx, pattern)
        self.statusChanged.emit(f"Subband pattern updated (slot {chr(65 + idx)})", FG2)

    @Slot(int, int, int)
    def _on_subband_position(self, index: int, position: int, total_steps: int):
        if 0 <= index < len(self._slot_subband_positions) and total_steps > 0:
            self._slot_subband_positions[index] = position % total_steps
        if self._state == "live" and hasattr(self, "_gesture_widget") and index == self._selected_slot:
            self._gesture_widget.set_playhead_column(self._slot_subband_positions[index])

    @Slot(float)
    def _on_subband_intensity_changed(self, value: float):
        self._gesture_intensity = max(0.0, min(1.0, value))
        if self._worker:
            self._worker.set_subband_intensity(self._gesture_intensity)
        self.statusChanged.emit(f"Subband intensity: {self._gesture_intensity * 100:.0f}%", FG2)
