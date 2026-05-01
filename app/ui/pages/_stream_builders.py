"""StreamPage builder and action mixin methods."""
import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.form import (
    ACID,
    AMBER,
    BG0,
    BG1,
    FG0,
    FG2,
    FG3,
    LINE0,
    Panel,
    _lbl,
    section_title,
)
from app.ui.widgets.knob import Knob
from app.ui.widgets.vu import VUMeter
from app.ui.pages._stream_gesture import _GestureDrawWidget
from app.ui.pages._stream_slot import SlotPanel
from app.ui.pages._stream_advanced import AdvancedSlotPanel


class _StreamBuildersMixin:

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
        gesture_head.addWidget(_lbl("Gesture input (MVP)", 10, FG2, mono=True))
        gesture_head.addStretch()
        self._gesture_toggle_btn = QPushButton("Gesture ON" if self._gesture_enabled else "Gesture OFF")
        self._gesture_toggle_btn.setFixedHeight(24)
        self._gesture_toggle_btn.clicked.connect(self._toggle_gesture)
        self._gesture_clear_btn = QPushButton("Clear")
        self._gesture_clear_btn.setFixedHeight(24)
        self._gesture_clear_btn.clicked.connect(self._clear_gesture)
        gesture_head.addWidget(self._gesture_clear_btn)
        gesture_head.addWidget(self._gesture_toggle_btn)
        tb.addLayout(gesture_head)

        self._gesture_widget = _GestureDrawWidget(self._gesture_curve)
        self._gesture_widget.curveChanged.connect(self._on_gesture_curve_changed)
        tb.addWidget(self._gesture_widget)

        master_row = QHBoxLayout()
        self._master_knob = Knob("MASTER", 0.75)
        self._master_knob.valueChanged.connect(self._on_master_volume_changed)
        master_row.addWidget(self._master_knob)
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

        _accents = [ACID, "#40d0d8", "#d840b8", AMBER, "#5090d8", "#e0406a", FG2, "#c8a060"]

        self._slot_widgets = []
        for i, selected_model in enumerate(self._slot_assignments):
            name = chr(65 + i)
            accent = _accents[i % len(_accents)]
            slot = SlotPanel(i, name, self._models, selected_model, accent, params=self._slot_param_state[i])
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

        row_w = QWidget()
        row_w.setStyleSheet("background:transparent;")
        row_lay = QHBoxLayout(row_w)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(16)
        row_lay.addWidget(grid_wrap, 1)
        row_lay.addWidget(adv_panel)
        self._content_layout.addWidget(row_w)
        self._content_layout.addStretch()

        has_model = any(m is not None for m in self._slot_assignments)
        self._start_btn.setEnabled(has_model and not streaming)
        self._stop_btn.setEnabled(streaming)
        self._record_btn.setEnabled(streaming)
        self._gesture_toggle_btn.setEnabled(True)
        self._gesture_clear_btn.setEnabled(True)
        self._gesture_widget.setEnabled(True)
        if not streaming:
            self._record_btn.setText("Start rec")
            self._recording_active = False
            self._apply_gesture_to_worker()

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

    def _start_stream(self):
        from app.workers.stream_worker import StreamWorker

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
        self._worker.masterVu.connect(self._on_master_vu)
        self._worker.warning.connect(self._on_warning)
        self._worker.running.connect(self._on_running)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

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
