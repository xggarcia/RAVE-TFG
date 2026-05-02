from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

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
        self._recording_active = False
        self._gesture_enabled = True
        self._gesture_curve: list[tuple[float, float]] = [(0.0, 0.5), (1.0, 0.5)]
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

        root.addWidget(
            PageHeader(
                crumbs=["Generate & Stream", "Streaming GUI"],
                title="Multi-model streaming",
                desc="Realtime phase interpolation between RAVE models.",
                actions=[self._scan_btn, self._record_btn, self._stop_btn, self._start_btn],
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

    def _refresh_models(self):
        from app._paths import get_repo_root
        root = get_repo_root()
        candidates = []
        for folder in [root / "models" / "user_model" / "exported_model", root / "models" / "demo_model"]:
            if folder.exists():
                candidates.extend(str(p) for p in folder.glob("*.ts"))
        self._models = sorted(candidates)

        # Always show the streaming UI regardless of model presence
        if self._worker:
            self._set_live_preview_state(streaming=True)
        else:
            self._set_live_preview_state()

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
        self._set_live_preview_state()

    def _stop_stream(self):
        if self._worker:
            if self._recording_active:
                self._toggle_recording()
            self._worker.stop()
            self.statusChanged.emit("Stopping stream...", AMBER)

    def _toggle_gesture(self):
        self._gesture_enabled = not self._gesture_enabled
        if hasattr(self, "_gesture_toggle_btn"):
            self._gesture_toggle_btn.setText("Gesture ON" if self._gesture_enabled else "Gesture OFF")
        self._apply_gesture_to_worker()
        self.statusChanged.emit(f"Gesture control {'enabled' if self._gesture_enabled else 'disabled'}", FG2)

    def _clear_gesture(self):
        if hasattr(self, "_gesture_widget"):
            self._gesture_widget.clear_curve()

    @Slot(list)
    def _on_gesture_curve_changed(self, points: list):
        self._gesture_curve = [(float(x), float(y)) for x, y in points]
        self._apply_gesture_to_worker()

    def _apply_gesture_to_worker(self):
        if not self._worker:
            return
        self._worker.set_gesture_curve(self._gesture_curve)
        self._worker.set_gesture_loop_seconds(4.0)
        self._worker.set_gesture_enabled(self._gesture_enabled)
