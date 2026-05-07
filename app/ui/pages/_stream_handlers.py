"""StreamPage signal-handler and slot-event mixin methods."""
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel

from app.ui.widgets.form import (
    ACID,
    AMBER,
    BG1,
    FG0,
    FG1,
    FG2,
    FG3,
    LINE0,
    MONO,
    Panel,
    _lbl,
    section_title,
)


class _StreamHandlersMixin:

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
                    self._worker.set_slot_latent_dim(i, dim, 0.0, float(scale))
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
            self._worker.set_slot_latent_dim(slot_idx, dim, 0.0, scale)

    @Slot(int, bool)
    def _on_adv_use_prior(self, slot_idx: int, enabled: bool):
        if 0 <= slot_idx < len(self._slot_adv_state):
            self._slot_adv_state[slot_idx]["prior_on"] = enabled

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
