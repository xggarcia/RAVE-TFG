from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.form import (
    ACID,
    AMBER,
    BG0,
    BG1,
    BG2,
    FG0,
    FG1,
    FG2,
    FG3,
    LINE0,
    MONO,
    PageHeader,
    Panel,
    _lbl,
    section_title,
)
from app.ui.widgets.knob import Knob
from app.ui.widgets.phase_pad import PhasePad
from app.ui.widgets.vu import VUMeter
from app.ui.pages._stream_slot import SlotPanel
from app.workers.stream_worker import StreamWorker


class StreamPage(QWidget):
    statusChanged = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: StreamWorker | None = None
        self._models: list[str] = []
        self._slot_assignments: list[str | None] = [None, None, None, None]
        self._state = "empty"  # empty|loading|live|error
        self._stage_rows: list[QLabel] = []
        self._slot_widgets: list[_SlotPanel] = []
        self._master_vus: tuple[VUMeter, VUMeter] | None = None
        self._latency_lbl: QLabel | None = None
        self._build()
        self._refresh_models()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background:{BG0};")

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

        root.addWidget(
            PageHeader(
                crumbs=["Generate & Stream", "Streaming GUI"],
                title="Multi-model streaming",
                desc="Realtime phase interpolation between RAVE models.",
                actions=[self._scan_btn, self._stop_btn, self._start_btn],
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
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _refresh_models(self):
        candidates = []
        for folder in [
            Path("models/user_model/exported_model"),
            Path("models/demo_model"),
        ]:
            if folder.exists():
                candidates.extend(str(p) for p in folder.glob("*.ts"))
        self._models = sorted(candidates)
        if self._models:
            for i in range(4):
                selected = self._slot_assignments[i]
                if selected and selected in self._models:
                    continue
                self._slot_assignments[i] = self._models[i] if i < len(self._models) else None
        else:
            self._slot_assignments = [None, None, None, None]

        if not self._models:
            self._set_empty_state()
        elif self._worker:
            self._set_live_preview_state(streaming=True)
        else:
            self._set_live_preview_state()

    def _set_empty_state(self):
        self._state = "empty"
        self._clear_content()

        panel = Panel()
        panel.add_header(section_title("Streaming empty"))
        body = panel.body_layout()

        icon = QLabel("○")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"color:{FG3}; font-size:34px; background:transparent;")
        body.addWidget(icon)
        body.addWidget(_lbl("No streaming-capable models yet", 16, FG0, bold=True))
        body.addWidget(_lbl("Export at least one model with streaming enabled.", 12, FG2))

        row = QHBoxLayout()
        open_btn = QPushButton("Open demo models")
        open_btn.clicked.connect(self._refresh_models)
        export_btn = QPushButton("Go to export")
        export_btn.setProperty("role", "primary")
        export_btn.clicked.connect(lambda: self.statusChanged.emit("Open Export page", FG2))
        row.addWidget(open_btn)
        row.addWidget(export_btn)
        row.addStretch()
        body.addLayout(row)

        self._content_layout.addWidget(panel)
        self._content_layout.addStretch()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)

    def _set_loading_state(self):
        self._state = "loading"
        self._clear_content()

        panel = Panel()
        panel.add_header(section_title("Initializing audio engine"))
        body = panel.body_layout()

        body.addWidget(_lbl("Warming up streaming engine", 15, FG0, bold=True))
        body.addWidget(_lbl("Loading models, opening audio devices, and allocating buffers.", 12, FG2))

        self._stage_rows = []
        for label in [
            "Open audio devices",
            "Load model 1",
            "Load model 2",
            "Load model 3",
            "Allocate buffers",
        ]:
            row = _lbl(f"○ {label}", 11, FG3, mono=True)
            self._stage_rows.append(row)
            body.addWidget(row)

        self._content_layout.addWidget(panel)
        self._content_layout.addStretch()

    def _set_live_preview_state(self, streaming: bool = False):
        self._state = "live"
        self._clear_content()

        top = Panel()
        top.add_header(section_title("Master"), _lbl("LIVE", 10, ACID, mono=True))
        tb = top.body_layout()

        master_row = QHBoxLayout()
        master_row.addWidget(Knob("MASTER", 0.75))
        master_row.addWidget(Knob("DRY/WET", 0.4, accent=AMBER))
        vu_col = QHBoxLayout()
        vu_l = VUMeter(0.0, 0.0)
        vu_r = VUMeter(0.0, 0.0)
        self._master_vus = (vu_l, vu_r)
        vu_col.addWidget(vu_l)
        vu_col.addWidget(vu_r)
        master_row.addLayout(vu_col)
        self._latency_lbl = _lbl("44.1kHz · block 256 · latency 5.8ms", 10, FG2, mono=True)
        master_row.addWidget(self._latency_lbl)
        master_row.addStretch()
        tb.addLayout(master_row)

        self._content_layout.addWidget(top)

        grid_wrap = QWidget()
        grid = QGridLayout(grid_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        slots = [
            ("A", self._slot_assignments[0], ACID),
            ("B", self._slot_assignments[1], "#40d0d8"),
            ("C", self._slot_assignments[2], "#d840b8"),
            ("D", self._slot_assignments[3], FG3),
        ]

        self._slot_widgets = []
        for i, (name, selected_model, accent) in enumerate(slots):
            slot = SlotPanel(i, name, self._models, selected_model, accent)
            slot.paramsChanged.connect(self._on_slot_params_changed)
            slot.modelChanged.connect(self._on_slot_model_changed)
            self._slot_widgets.append(slot)
            grid.addWidget(slot, i // 2, i % 2)

        side = Panel()
        side.add_header(section_title("Phase control"))
        sb = side.body_layout()
        pad = PhasePad()
        sb.addWidget(pad)
        coords = _lbl("x=0.50  y=0.50", 10, FG2, mono=True)
        sb.addWidget(coords)

        def _update_xy(x: float, y: float):
            coords.setText(f"x={x:.3f}  y={y:.3f}")
            if self._worker:
                self._worker.set_phase_xy(x, y)

        pad.xyChanged.connect(_update_xy)

        row = QHBoxLayout()
        row.addWidget(grid_wrap, 1)
        row.addWidget(side)
        self._content_layout.addLayout(row)
        self._content_layout.addStretch()

        has_model = any(m is not None for m in self._slot_assignments)
        self._start_btn.setEnabled(has_model and not streaming)
        self._stop_btn.setEnabled(streaming)

    def _start_stream(self):
        if self._worker:
            return
        if not any(m is not None for m in self._slot_assignments):
            self._set_empty_state()
            return

        self._set_loading_state()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        selected = [self._slot_assignments[i] for i in range(4)]
        self._worker = StreamWorker(selected)
        self._worker.stage.connect(self._on_stage)
        self._worker.log.connect(self._on_log)
        self._worker.slotVu.connect(self._on_slot_vu)
        self._worker.masterVu.connect(self._on_master_vu)
        self._worker.warning.connect(self._on_warning)
        self._worker.running.connect(self._on_running)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _stop_stream(self):
        if self._worker:
            self._worker.stop()
            self.statusChanged.emit("Stopping stream...", AMBER)

    @Slot(str, bool, bool)
    def _on_stage(self, label: str, done: bool, current: bool):
        for row in self._stage_rows:
            if label in row.text():
                if done:
                    row.setText(f"✓ {label}")
                    row.setStyleSheet(f"color:{ACID}; font-size:11px; {MONO} background:transparent;")
                elif current:
                    row.setText(f"● {label}")
                    row.setStyleSheet(f"color:{AMBER}; font-size:11px; {MONO} background:transparent;")
                else:
                    row.setText(f"○ {label}")
                    row.setStyleSheet(f"color:{FG3}; font-size:11px; {MONO} background:transparent;")

    @Slot(str, str)
    def _on_log(self, level: str, message: str):
        color = FG2 if level == "INFO" else AMBER if level == "WARN" else FG3
        self.statusChanged.emit(message, color)

    @Slot(int, float, float)
    def _on_slot_vu(self, index: int, level: float, peak: float):
        if self._state != "live":
            return
        if 0 <= index < len(self._slot_widgets):
            self._slot_widgets[index].set_vu(level, peak)

    @Slot(float, float, float)
    def _on_master_vu(self, left: float, right: float, latency_ms: float):
        if self._state != "live":
            return
        if self._master_vus:
            self._master_vus[0].set_levels(left, min(1.0, left + 0.08))
            self._master_vus[1].set_levels(right, min(1.0, right + 0.08))
        if self._latency_lbl:
            self._latency_lbl.setText(f"44.1kHz · block 256 · latency {latency_ms:.1f}ms")

    @Slot(str)
    def _on_warning(self, message: str):
        self.statusChanged.emit(message, AMBER)

    @Slot(bool)
    def _on_running(self, running: bool):
        if running:
            self.statusChanged.emit("Streaming live", ACID)
            self._set_live_preview_state(streaming=True)
        else:
            self.statusChanged.emit("Idle", FG3)

    @Slot(int, float, float)
    def _on_slot_params_changed(self, index: int, gain: float, temp: float, smooth: float):
        if self._worker:
            self._worker.set_slot_params(index, gain=gain, temp=temp, smooth=smooth)

    @Slot(int, str)
    def _on_slot_model_changed(self, index: int, model_path: str):
        value = model_path or None
        self._slot_assignments[index] = value

        if model_path and model_path not in self._models:
            self._models.append(model_path)
            self._models.sort()

        self._start_btn.setEnabled(any(m is not None for m in self._slot_assignments))

        if self._worker:
            self._worker.set_slot_model(index, value)
            name = Path(value).name if value else "empty"
            self.statusChanged.emit(f"Slot {index + 1} model set to {name}", FG2)

    @Slot(dict)
    def _on_finished(self, summary: dict):
        self.statusChanged.emit(
            f"Stream stopped · underruns={summary.get('underruns', 0)}",
            FG2,
        )
        self._worker = None
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self._set_live_preview_state()

    @Slot(str, str)
    def _on_failed(self, short: str, traceback: str):
        self.statusChanged.emit(short, FG3)
        self._worker = None
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)

        self._clear_content()
        panel = Panel()
        panel.add_header(section_title("Streaming error"))
        body = panel.body_layout()
        body.addWidget(_lbl(short, 14, FG0, bold=True))
        err = QLabel(traceback)
        err.setStyleSheet(f"background:{BG1}; color:{FG1}; border:1px solid {LINE0}; border-radius:4px; padding:10px; {MONO} font-size:10px;")
        err.setWordWrap(True)
        body.addWidget(err)
        self._content_layout.addWidget(panel)
        self._content_layout.addStretch()
