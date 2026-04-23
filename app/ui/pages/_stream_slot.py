from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QWidget

from app.ui.widgets.form import ACID, AMBER, FG0, FG2, FG3, LINE0, Panel, _lbl, section_title
from app.ui.widgets.knob import Knob
from app.ui.widgets.vu import VUMeter

_PWR_ON  = f"color:#000; background:{ACID}; border:1px solid {ACID}; border-radius:4px; font-size:10px; padding:2px 7px;"
_PWR_OFF = f"color:{FG3}; background:transparent; border:1px solid {LINE0}; border-radius:4px; font-size:10px; padding:2px 7px;"

_INPUT_ACTIVE   = "border-radius:3px; font-size:10px; padding:2px 8px; border:1px solid;"
_INPUT_INACTIVE = f"color:{FG3}; background:transparent; border:1px solid {LINE0}; border-radius:3px; font-size:10px; padding:2px 8px;"


class SlotPanel(Panel):
    paramsChanged    = Signal(int, float, float, float, float)  # index, gain, temp, smooth, dry_wet
    modelChanged     = Signal(int, str)
    enabledChanged   = Signal(int, bool)   # index, enabled
    inputModeChanged = Signal(int, str)    # index, "random"|"audio"
    audioFileChanged = Signal(int, str)    # index, absolute path

    def __init__(
        self,
        index: int,
        slot_name: str,
        model_paths: list[str],
        selected_model: str | None,
        accent: str,
        parent=None,
    ):
        super().__init__(parent)
        self._index = index
        self._powered = False
        self._input_mode = "random"

        self._pwr_btn = QPushButton("OFF")
        self._pwr_btn.setFixedSize(34, 22)
        self._pwr_btn.setStyleSheet(_PWR_OFF)
        self._pwr_btn.clicked.connect(self._toggle_power)

        self._btn_random = QPushButton("RANDOM")
        self._btn_random.setFixedHeight(22)
        self._btn_random.clicked.connect(lambda: self._set_input_mode("random"))

        self._btn_audio = QPushButton("AUDIO")
        self._btn_audio.setFixedHeight(22)
        self._btn_audio.clicked.connect(lambda: self._set_input_mode("audio"))

        self._apply_input_style()

        right_row = QHBoxLayout()
        right_row.setSpacing(6)
        right_row.addWidget(self._btn_random)
        right_row.addWidget(self._btn_audio)
        right_row.addWidget(_lbl(slot_name, 10, FG0, mono=True))
        right_row.addWidget(self._pwr_btn)

        right_widget = QWidget()
        right_widget.setStyleSheet("background:transparent;")
        right_widget.setLayout(right_row)

        self._title = section_title(Path(selected_model).name if selected_model else "No model")
        self.add_header(self._title, right_widget)

        # Picker section — always enabled (browse buttons must work when slot is off)
        picker_widget = self.add_body()
        picker_widget.layout().setContentsMargins(14, 8, 14, 4)
        picker_widget.layout().setSpacing(6)

        picker = QHBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(260)
        self._model_combo.currentIndexChanged.connect(self._emit_model)
        picker.addWidget(self._model_combo, 1)

        self._browse_model_btn = QPushButton("Browse…")
        self._browse_model_btn.setFixedHeight(28)
        self._browse_model_btn.clicked.connect(self._browse_model)
        picker.addWidget(self._browse_model_btn)
        picker_widget.layout().addLayout(picker)

        # Audio file picker row — shown only when input mode is "audio"
        self._audio_row = QWidget()
        self._audio_row.setStyleSheet("background:transparent;")
        audio_row_layout = QHBoxLayout(self._audio_row)
        audio_row_layout.setContentsMargins(0, 0, 0, 0)
        audio_row_layout.setSpacing(6)
        self._audio_lbl = QLabel("No file selected")
        self._audio_lbl.setStyleSheet(f"color:{FG3}; font-size:10px; background:transparent;")
        self._audio_lbl.setMinimumWidth(200)
        self._browse_audio_btn = QPushButton("Load audio…")
        self._browse_audio_btn.setFixedHeight(28)
        self._browse_audio_btn.clicked.connect(self._browse_audio)
        audio_row_layout.addWidget(self._audio_lbl, 1)
        audio_row_layout.addWidget(self._browse_audio_btn)
        self._audio_row.setVisible(False)
        picker_widget.layout().addWidget(self._audio_row)

        # Controls section — disabled when slot is off
        self._controls_widget = self.add_body()
        controls = self._controls_widget.layout()
        controls.setContentsMargins(14, 4, 14, 14)
        controls.setSpacing(10)

        knobs = QHBoxLayout()
        self._gain = Knob("GAIN", 0.65, accent=accent)
        self._temp = Knob("TEMP", 0.52, accent=accent)
        self._smooth = Knob("SMOOTH", 0.40, accent=accent)
        self._dry_wet = Knob("DRY/WET", 1.0, accent=AMBER)
        for knob in (self._gain, self._temp, self._smooth, self._dry_wet):
            knob.valueChanged.connect(self._emit_params)
            knobs.addWidget(knob)
        knobs.addStretch()
        controls.addLayout(knobs)

        vu_row = QHBoxLayout()
        self._vu_l = VUMeter(0.0, 0.0)
        self._vu_r = VUMeter(0.0, 0.0)
        vu_row.addWidget(self._vu_l)
        vu_row.addWidget(self._vu_r)
        vu_row.addWidget(_lbl("spectrogram pending", 10, FG3, mono=True))
        vu_row.addStretch()
        controls.addLayout(vu_row)

        self.set_available_models(model_paths, selected_model)

        # DRY/WET only meaningful in audio mode
        self._dry_wet.set_available(False)

        # Initial OFF — disable only the controls, browse buttons stay active
        self._controls_widget.setEnabled(False)

    def set_available_models(self, model_paths: list[str], selected_model: str | None):
        prev = self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItem("— empty slot —", "")
        for path in model_paths:
            self._model_combo.addItem(Path(path).name, path)

        idx = 0
        if selected_model:
            for i in range(self._model_combo.count()):
                if self._model_combo.itemData(i) == selected_model:
                    idx = i
                    break
        self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(prev)

        current = self.current_model()
        self._title.setText(Path(current).name if current else "No model")

    def current_model(self) -> str | None:
        value = self._model_combo.currentData()
        return value if value else None

    def _emit_params(self, _):
        self.paramsChanged.emit(self._index, self._gain.value, self._temp.value, self._smooth.value, self._dry_wet.value)

    def _emit_model(self, _):
        current = self.current_model()
        self._title.setText(Path(current).name if current else "No model")
        self.modelChanged.emit(self._index, current or "")

    def _browse_model(self):
        dlg = QFileDialog(self, "Select model")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("TorchScript (*.ts)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if not dlg.exec():
            return
        files = dlg.selectedFiles()
        if not files:
            return
        path = files[0]

        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == path:
                self._model_combo.setCurrentIndex(i)
                return

        self._model_combo.addItem(Path(path).name, path)
        self._model_combo.setCurrentIndex(self._model_combo.count() - 1)

    def _toggle_power(self):
        self._powered = not self._powered
        self._pwr_btn.setText("ON" if self._powered else "OFF")
        self._pwr_btn.setStyleSheet(_PWR_ON if self._powered else _PWR_OFF)
        self._controls_widget.setEnabled(self._powered)
        self.enabledChanged.emit(self._index, self._powered)

    def _set_input_mode(self, mode: str):
        if mode == self._input_mode:
            return
        self._input_mode = mode
        self._apply_input_style()
        self._audio_row.setVisible(mode == "audio")
        self._dry_wet.set_available(mode == "audio")
        self.inputModeChanged.emit(self._index, mode)

    def _browse_audio(self):
        dlg = QFileDialog(self, "Select audio file")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Audio files (*.wav *.flac *.ogg *.mp3 *.aiff)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if not dlg.exec():
            return
        files = dlg.selectedFiles()
        if not files:
            return
        path = files[0]
        self._audio_lbl.setText(Path(path).name)
        self._audio_lbl.setToolTip(path)
        self._audio_lbl.setStyleSheet(f"color:{FG2}; font-size:10px; background:transparent;")
        self.audioFileChanged.emit(self._index, path)

    def _apply_input_style(self):
        if self._input_mode == "random":
            self._btn_random.setStyleSheet(f"color:#000; background:{AMBER}; {_INPUT_ACTIVE} border-color:{AMBER};")
            self._btn_audio.setStyleSheet(_INPUT_INACTIVE)
        else:
            self._btn_random.setStyleSheet(_INPUT_INACTIVE)
            self._btn_audio.setStyleSheet(f"color:#000; background:{FG2}; {_INPUT_ACTIVE} border-color:{FG2};")

    @property
    def powered(self) -> bool:
        return self._powered

    @property
    def input_mode(self) -> str:
        return self._input_mode

    def set_vu(self, level: float, peak: float):
        if not self._powered:
            return
        self._vu_l.set_levels(level, peak)
        self._vu_r.set_levels(level * 0.92, min(1.0, peak * 0.95))
