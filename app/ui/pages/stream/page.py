"""Streaming page — single class, no mixins.

UI construction lives in the `_build_*` and `_set_*_state` methods.
Worker signal handlers live in the `_on_*` methods.
"""
import json
import os
from pathlib import Path

import torch
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
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
    FG0,
    FG1,
    FG2,
    FG3,
    LINE0,
    MONO,
    Panel,
    PageHeader,
    _lbl,
    section_title,
)
from app.ui.widgets.knob import Knob
from app.ui.widgets.vu import VUMeter
from app.ui.pages.stream.subbands import SubbandDrawWidget
from app.ui.pages.stream.slot import SlotPanel
from app.ui.pages.stream.advanced import AdvancedSlotPanel
from app.workers.stream_worker import StreamWorker


class StreamPage(QWidget):
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
        self._dynamic_stride_enabled = True
        self._build()
        self._refresh_models()

    @staticmethod
    def _default_adv_state(n_dims: int = 8) -> dict:
        return {"scales": [1.0] * n_dims, "active_dim": 0}

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

    # ── Top-level layout ─────────────────────────────────────────────────────

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
        self._stack_wrap.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._stack_wrap.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(24, 24, 24, 24)
        self._content_layout.setSpacing(18)

        self._stack_wrap.setWidget(self._content)

        # Sticky right column for the Advanced panel — kept outside the scroll
        # so it stays in the viewport regardless of how many slots are loaded.
        self._adv_host = QWidget()
        self._adv_host.setFixedWidth(380)
        self._adv_host.setStyleSheet(f"background:{BG0};")
        self._adv_host_layout = QVBoxLayout(self._adv_host)
        self._adv_host_layout.setContentsMargins(0, 24, 24, 24)
        self._adv_host_layout.setSpacing(0)
        self._adv_host_layout.setAlignment(Qt.AlignTop)
        self._adv_host.setVisible(False)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._stack_wrap, 1)
        body_layout.addWidget(self._adv_host, 0)
        root.addWidget(body, 1)

    def _clear_content(self):
        self._slot_widgets = []
        self._master_vus = None
        self._latency_lbl = None
        self._adv_panel = None
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        while self._adv_host_layout.count():
            item = self._adv_host_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._adv_host.setVisible(False)

    def _refresh_models(self):
        from app._paths import get_repo_root
        root = get_repo_root()
        candidates = []
        for folder in [root / "models" / "user_model" / "exported_model", root / "demo" / "model"]:
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
        self._slot_subband_patterns.append(None)
        self._set_live_preview_state()

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

    # ── Page state builders (empty / loading / live) ─────────────────────────

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
        self._record_btn.setEnabled(False)
        self._record_btn.setText("Start rec")
        self._recording_active = False

    def _set_loading_state(self):
        self._state = "loading"
        self._clear_content()

        panel = Panel()
        panel.add_header(section_title("Initializing audio engine"))
        body = panel.body_layout()

        body.addWidget(_lbl("Warming up streaming engine", 15, FG0, bold=True))
        body.addWidget(_lbl("Loading models, opening audio devices, and allocating buffers.", 12, FG2))

        self._stage_rows = []
        model_labels = [f"Load model {i + 1}" for i in range(len(self._slot_assignments))]
        for label in ["Open audio devices", *model_labels, "Allocate buffers"]:
            row = _lbl(f"○ {label}", 11, FG3, mono=True)
            self._stage_rows.append(row)
            body.addWidget(row)

        self._content_layout.addWidget(panel)
        self._content_layout.addStretch()

    def _set_live_preview_state(self, streaming: bool = False):
        self._state = "live"
        self._clear_content()
        self._ensure_slot_lists(len(self._slot_assignments))

        top = Panel()
        top.add_header(section_title("Master"), _lbl("LIVE", 10, ACID, mono=True))
        tb = top.body_layout()

        gesture_head = QHBoxLayout()
        gesture_head.addWidget(_lbl("Subband Control", 10, FG2, mono=True))
        gesture_head.addStretch()
        self._gesture_clear_btn = QPushButton("Clear")
        self._gesture_clear_btn.setFixedHeight(24)
        self._gesture_clear_btn.clicked.connect(self._clear_subband)
        self._gesture_invert_btn = QPushButton("Invert")
        self._gesture_invert_btn.setFixedHeight(24)
        self._gesture_invert_btn.clicked.connect(self._invert_subband)
        gesture_head.addWidget(self._gesture_clear_btn)
        gesture_head.addWidget(self._gesture_invert_btn)
        tb.addLayout(gesture_head)

        self._gesture_widget = SubbandDrawWidget(num_subbands=5, n_timesteps=96)
        # Preload the currently-selected slot's pattern (if any).
        sel = self._selected_slot if 0 <= self._selected_slot < len(self._slot_subband_patterns) else 0
        pat = self._slot_subband_patterns[sel] if 0 <= sel < len(self._slot_subband_patterns) else None
        if pat is not None:
            self._gesture_widget.set_pattern(pat)
        self._gesture_widget.subbandPatternChanged.connect(self._on_subband_pattern_changed)
        tb.addWidget(self._gesture_widget, alignment=Qt.AlignHCenter)

        master_row = QHBoxLayout()
        self._master_knob = Knob("MASTER", 0.75)
        self._master_knob.valueChanged.connect(self._on_master_volume_changed)
        self._gesture_intensity_knob = Knob("SUBBAND\nINT", 1.0, accent=ACID)
        self._gesture_intensity_knob.valueChanged.connect(self._on_subband_intensity_changed)
        master_row.addWidget(self._master_knob)
        master_row.addWidget(self._gesture_intensity_knob)
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

        self._dynamic_stride_chk = QCheckBox("Dynamic stride")
        self._dynamic_stride_chk.setChecked(self._dynamic_stride_enabled)
        self._dynamic_stride_chk.toggled.connect(self._on_dynamic_stride_toggled)
        master_row.addWidget(self._dynamic_stride_chk)

        tb.addLayout(master_row)

        self._content_layout.addWidget(top)

        grid_wrap = QWidget()
        grid = QGridLayout(grid_wrap)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        _accents = [ACID, "#40d0d8", "#d840b8", AMBER, "#5090d8", "#e0406a", FG2, "#c8a060"]

        _SLOT_W = 420

        self._slot_widgets = []
        for i, selected_model in enumerate(self._slot_assignments):
            name = chr(65 + i)
            accent = _accents[i % len(_accents)]
            slot = SlotPanel(i, name, self._models, selected_model, accent, params=self._slot_param_state[i])
            slot.setFixedWidth(_SLOT_W)
            slot.paramsChanged.connect(self._on_slot_params_changed)
            slot.modelChanged.connect(self._on_slot_model_changed)
            slot.enabledChanged.connect(self._on_slot_enabled_changed)
            slot.inputModeChanged.connect(self._on_slot_input_mode_changed)
            slot.audioFileChanged.connect(self._on_slot_audio_file_changed)
            slot.anchorsChanged.connect(self._on_slot_anchors_changed)
            slot.phaseChanged.connect(self._on_slot_phase_changed)
            slot.slotSelected.connect(self._select_slot)
            self._slot_widgets.append(slot)
            grid.addWidget(slot, i // 2, i % 2)

        if not streaming:
            n = len(self._slot_assignments)
            add_btn = self._make_add_slot_btn()
            add_btn.setFixedWidth(_SLOT_W)
            grid.addWidget(add_btn, n // 2, n % 2)

        adv_panel = AdvancedSlotPanel()
        adv_panel.latentDimChanged.connect(self._on_adv_latent_dim)
        adv_panel.phaseMapClicked.connect(self._on_phase_map_clicked)
        self._adv_panel = adv_panel

        sel = self._selected_slot if 0 <= self._selected_slot < len(self._slot_assignments) else 0
        self._selected_slot = sel
        name = chr(65 + sel)
        if self._slot_is_loaded(sel):
            adv_panel.set_slot(sel, name)
            adv_panel.load_state(self._slot_adv_state[sel])
        else:
            adv_panel.set_slot_unloaded(sel, name)

        self._content_layout.addWidget(grid_wrap)
        self._content_layout.addStretch()

        # Mount the Advanced panel in the sticky right column so it stays
        # in the viewport even when many slots push the grid below the fold.
        self._adv_host_layout.addWidget(adv_panel, 0, Qt.AlignTop | Qt.AlignHCenter)
        self._adv_host.setVisible(True)

        has_model = any(m is not None for m in self._slot_assignments)
        self._start_btn.setEnabled(has_model and not streaming)
        self._stop_btn.setEnabled(streaming)
        self._record_btn.setEnabled(streaming)
        self._gesture_clear_btn.setEnabled(True)
        self._gesture_invert_btn.setEnabled(True)
        self._gesture_widget.setEnabled(True)
        if not streaming:
            self._record_btn.setText("Start rec")
            self._recording_active = False

    def _try_load_phase_map(self, _index: int, anchors_path: str):
        def _as_anchor_list(raw):
            if isinstance(raw, list):
                return [a for a in raw if isinstance(a, dict)]
            if isinstance(raw, dict):
                entries = raw.get("phase_anchors", raw.get("anchors", []))
                if isinstance(entries, list):
                    return [a for a in entries if isinstance(a, dict)]
            return []

        def _as_pca_list(raw):
            if isinstance(raw, list):
                return [p for p in raw if isinstance(p, dict)]
            if isinstance(raw, dict):
                entries = raw.get("phase_pca", raw.get("pca", []))
                if isinstance(entries, list):
                    return [p for p in entries if isinstance(p, dict)]
            return []

        def _flat_numbers(values):
            out = []
            def _walk(v):
                if isinstance(v, (int, float)):
                    out.append(float(v))
                elif isinstance(v, list):
                    for vv in v:
                        _walk(vv)
            _walk(values)
            return out

        stem = os.path.splitext(anchors_path)[0]
        pca_path = stem + "_pca.json"

        try:
            with open(anchors_path, encoding="utf-8") as f:
                anchors_raw = json.load(f)

            anchors_data = _as_anchor_list(anchors_raw)
            pca_data = _as_pca_list(anchors_raw)

            if not pca_data and os.path.exists(pca_path):
                with open(pca_path, encoding="utf-8") as f:
                    pca_raw = json.load(f)
                pca_data = _as_pca_list(pca_raw)

            if not pca_data and anchors_data:
                pca_data = []
                for anchor in anchors_data:
                    mean_vals = _flat_numbers(anchor.get("mean_z", []))
                    x = mean_vals[0] if len(mean_vals) > 0 else 0.0
                    y = mean_vals[1] if len(mean_vals) > 1 else 0.0
                    pca_data.append({"label": anchor.get("label", "phase"), "points": [], "anchor_xy": [x, y]})
                self.statusChanged.emit("Phase map loaded without PCA cloud (anchor-only fallback)", FG3)

            if hasattr(self, "_adv_panel"):
                if pca_data:
                    self._adv_panel.load_phase_map(pca_data, anchors_data)
                else:
                    self._adv_panel.clear_phase_map()
        except Exception:
            if hasattr(self, "_adv_panel"):
                self._adv_panel.clear_phase_map()

    # ── Streaming lifecycle ──────────────────────────────────────────────────

    def _start_stream(self):
        if self._worker:
            return
        if not any(m is not None for m in self._slot_assignments):
            self._set_empty_state()
            return

        self._set_loading_state()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        selected = list(self._slot_assignments)
        self._worker = StreamWorker(selected)
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
        self._worker.start()

        from tools.stream_profiler import StreamProfiler
        self._profiler = StreamProfiler(get_worker=lambda: self._worker)
        csv_path = self._profiler.start()
        if csv_path:
            self._worker.log.emit("INFO", f"[profiler] logging → {csv_path.name}")

    def _toggle_recording(self):
        if not self._worker:
            self.statusChanged.emit("Start streaming first to record", AMBER)
            return

        try:
            if not self._recording_active:
                out_path = self._worker.start_recording()
                self._recording_active = True
                self._record_btn.setText("Stop rec")
                self.statusChanged.emit(f"Recording: {Path(out_path).name}", ACID)
            else:
                out_path = self._worker.stop_recording()
                self._recording_active = False
                self._record_btn.setText("Start rec")
                if out_path:
                    self.statusChanged.emit(f"Saved recording: {Path(out_path).name}", FG2)
                else:
                    self.statusChanged.emit("Recording stopped", FG2)
        except Exception as exc:
            self._recording_active = False
            self._record_btn.setText("Start rec")
            self.statusChanged.emit(f"Recording error: {exc}", FG3)

    # ── Worker signal handlers ───────────────────────────────────────────────

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

    @Slot(int, object)
    def _on_slot_spectrogram(self, index: int, audio):
        if self._state != "live":
            return
        if 0 <= index < len(self._slot_widgets):
            self._slot_widgets[index].push_spectrogram(audio)

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
            for slot in self._slot_widgets:
                self._worker.set_slot_enabled(slot._index, slot.powered)
                self._worker.set_slot_input_mode(slot._index, slot.input_mode)
                p = slot.slot_params
                self._slot_param_state[slot._index] = p
                self._worker.set_slot_params(
                    slot._index,
                    gain=p["gain"], temp=p["temp"], smooth=p["smooth"],
                    dry_wet=p["dry_wet"], noise=p["noise"], bias=p["bias"],
                )
                self._worker.set_slot_phase(slot._index, p["phase"])
            for i, state in enumerate(self._slot_adv_state):
                scales = state.get("scales", [])
                for dim, scale in enumerate(scales):
                    self._worker.set_slot_latent_dim(i, dim, float(scale))
            if self._adv_panel and 0 <= self._selected_slot < len(self._slot_latent_sizes):
                latent_size = self._slot_latent_sizes[self._selected_slot]
                if latent_size:
                    self._adv_panel.set_latent_size(latent_size)
        else:
            self.statusChanged.emit("Idle", FG3)

    def _select_slot(self, index: int):
        if self._adv_panel is None:
            return
        prev = self._selected_slot
        if 0 <= prev < len(self._slot_adv_state) and self._slot_is_loaded(prev):
            self._slot_adv_state[prev] = self._adv_panel.dump_state()
        self._selected_slot = index
        for w in self._slot_widgets:
            active = (w._index == index)
            w._name_btn.setStyleSheet(
                f"color:#000; background:{ACID}; border:1px solid {ACID}; "
                f"border-radius:3px; font-size:10px; font-family:'JetBrains Mono','Consolas',monospace;"
                if active else
                f"color:{FG0}; background:transparent; border:1px solid {LINE0}; "
                f"border-radius:3px; font-size:10px; font-family:'JetBrains Mono','Consolas',monospace;"
            )
        # Swap the shared subband widget to show *this* slot's pattern.
        if hasattr(self, "_gesture_widget") and self._gesture_widget is not None:
            import numpy as _np
            pat = (self._slot_subband_patterns[index]
                   if 0 <= index < len(self._slot_subband_patterns) else None)
            if pat is not None:
                self._gesture_widget.set_pattern(pat)
            else:
                self._gesture_widget.set_pattern(
                    _np.zeros((self._gesture_widget.num_subbands,
                               self._gesture_widget.n_timesteps), dtype=_np.float32)
                )
            if 0 <= index < len(self._slot_subband_positions):
                self._gesture_widget.set_playhead_column(self._slot_subband_positions[index])
        if prev == index:
            return
        name = chr(65 + index) if 0 <= index < 26 else str(index + 1)
        if self._slot_is_loaded(index):
            self._adv_panel.set_slot(index, name)
            if 0 <= index < len(self._slot_adv_state):
                self._adv_panel.load_state(self._slot_adv_state[index])
        else:
            self._adv_panel.set_slot_unloaded(index, name)

    @Slot(int, int)
    def _on_slot_info(self, index: int, latent_size: int):
        if 0 <= index < len(self._slot_latent_sizes):
            self._slot_latent_sizes[index] = latent_size
        self._resize_adv_state(index, latent_size)
        if self._adv_panel and index == self._selected_slot:
            name = chr(65 + index) if 0 <= index < 26 else str(index + 1)
            self._adv_panel.set_slot(index, name)
            self._adv_panel.load_state(self._slot_adv_state[index])

    @Slot(int, int, float)
    def _on_adv_latent_dim(self, slot_idx: int, dim: int, scale: float):
        if 0 <= slot_idx < len(self._slot_adv_state):
            scales = self._slot_adv_state[slot_idx].get("scales", [])
            if 0 <= dim < len(scales):
                scales[dim] = scale
        if self._worker:
            self._worker.set_slot_latent_dim(slot_idx, dim, scale)

    @Slot(int, float, float, float, float, float, float)
    def _on_slot_params_changed(self, index, gain, temp, smooth, dry_wet, noise, bias):
        self._select_slot(index)
        if 0 <= index < len(self._slot_param_state):
            self._slot_param_state[index].update(
                {"gain": gain, "temp": temp, "smooth": smooth, "dry_wet": dry_wet, "noise": noise, "bias": bias}
            )
        if self._worker:
            self._worker.set_slot_params(index, gain=gain, temp=temp, smooth=smooth,
                                          dry_wet=dry_wet, noise=noise, bias=bias)

    @Slot(float)
    def _on_master_volume_changed(self, value: float):
        if self._worker:
            self._worker.set_master_volume(value)

    @Slot(float)
    def _on_subband_intensity_changed(self, value: float):
        self._gesture_intensity = max(0.0, min(1.0, value))
        if self._worker:
            self._worker.set_subband_intensity(self._gesture_intensity)
        self.statusChanged.emit(f"Subband intensity: {self._gesture_intensity * 100:.0f}%", FG2)

    @Slot(bool)
    def _on_dynamic_stride_toggled(self, enabled: bool):
        self._dynamic_stride_enabled = bool(enabled)
        if self._worker:
            self._worker.set_dynamic_stride_enabled(self._dynamic_stride_enabled)
        state = "ON" if enabled else "OFF"
        self.statusChanged.emit(f"Dynamic stride: {state}", FG2)

    @Slot(int, float)
    def _on_slot_phase_changed(self, index: int, value: float):
        self._select_slot(index)
        if 0 <= index < len(self._slot_param_state):
            self._slot_param_state[index]["phase"] = value
        if self._worker:
            self._worker.set_slot_phase(index, value)

    @Slot(int, str)
    def _on_slot_anchors_changed(self, index: int, path: str):
        self._select_slot(index)
        self._slot_anchors[index] = path
        if self._worker:
            self._worker.set_slot_anchors(index, path)
        self.statusChanged.emit(f"Slot {index + 1} anchors → {Path(path).name}", FG2)
        self._try_load_phase_map(index, path)

    @Slot(int, dict)
    def _on_phase_map_clicked(self, index: int, blend: dict):
        if self._worker:
            self._worker.set_slot_phase_map_anchor(
                index, blend.get("mean_z", []), blend.get("std_z", [])
            )

    @Slot(int, str)
    def _on_slot_audio_file_changed(self, index: int, path: str):
        self._select_slot(index)
        if self._worker:
            self._worker.set_slot_audio_file(index, path)
        self.statusChanged.emit(f"Slot {index + 1} audio → {Path(path).name}", FG2)

    @Slot(int, str)
    def _on_slot_input_mode_changed(self, index: int, mode: str):
        self._select_slot(index)
        if self._worker:
            self._worker.set_slot_input_mode(index, mode)
        self.statusChanged.emit(f"Slot {index + 1} input → {mode}", FG2)

    @Slot(int, bool)
    def _on_slot_enabled_changed(self, index: int, enabled: bool):
        self._select_slot(index)
        if self._worker:
            self._worker.set_slot_enabled(index, enabled)
        self.statusChanged.emit(f"Slot {index + 1} powered {'on' if enabled else 'off'}", FG2)

    @Slot(int, str)
    def _on_slot_model_changed(self, index: int, model_path: str):
        self._select_slot(index)
        value = model_path or None
        self._slot_assignments[index] = value
        if not value and 0 <= index < len(self._slot_latent_sizes):
            self._slot_latent_sizes[index] = None
        if not value and 0 <= index < len(self._slot_adv_state):
            self._slot_adv_state[index] = self._default_adv_state()
            if self._adv_panel and index == self._selected_slot:
                self._adv_panel.load_state(self._slot_adv_state[index])

        if model_path and model_path not in self._models:
            self._models.append(model_path)
            self._models.sort()

        self._start_btn.setEnabled(any(m is not None for m in self._slot_assignments))

        if self._worker:
            self._worker.set_slot_model(index, value)
            name = Path(value).name if value else "empty"
            self.statusChanged.emit(f"Slot {index + 1} model set to {name}", FG2)

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

    @Slot(dict)
    def _on_finished(self, summary: dict):
        if self._profiler:
            self._profiler.stop()
        self._recording_active = False
        self._record_btn.setText("Start rec")
        self._record_btn.setEnabled(False)
        self.statusChanged.emit(
            f"Stream stopped · underruns={summary.get('underruns', 0)}", FG2,
        )
        self._worker = None
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self._set_live_preview_state()

    @Slot(str, str)
    def _on_failed(self, short: str, traceback: str):
        if self._profiler:
            self._profiler.stop()
        self._recording_active = False
        self._record_btn.setText("Start rec")
        self._record_btn.setEnabled(False)
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
        err.setStyleSheet(
            f"background:{BG1}; color:{FG1}; border:1px solid {LINE0}; "
            f"border-radius:4px; padding:10px; {MONO} font-size:10px;"
        )
        err.setWordWrap(True)
        body.addWidget(err)
        self._content_layout.addWidget(panel)
        self._content_layout.addStretch()
