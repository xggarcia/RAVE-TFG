from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QComboBox, QHBoxLayout, QPushButton

from app.ui.widgets.form import FG0, FG3, Panel, _lbl, section_title
from app.ui.widgets.knob import Knob
from app.ui.widgets.vu import VUMeter


class SlotPanel(Panel):
    paramsChanged = Signal(int, float, float, float)
    modelChanged = Signal(int, str)

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

        right = _lbl(slot_name, 10, FG0, mono=True)
        self._title = section_title(Path(selected_model).name if selected_model else "No model")
        self.add_header(self._title, right)
        body = self.body_layout()

        picker = QHBoxLayout()
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(260)
        self._model_combo.currentIndexChanged.connect(self._emit_model)
        picker.addWidget(self._model_combo, 1)

        browse = QPushButton("Browse…")
        browse.setFixedHeight(28)
        browse.clicked.connect(self._browse_model)
        picker.addWidget(browse)
        body.addLayout(picker)

        knobs = QHBoxLayout()
        self._gain = Knob("GAIN", 0.65, accent=accent)
        self._temp = Knob("TEMP", 0.52, accent=accent)
        self._smooth = Knob("SMOOTH", 0.40, accent=accent)
        for knob in (self._gain, self._temp, self._smooth):
            knob.valueChanged.connect(self._emit_params)
            knobs.addWidget(knob)
        knobs.addStretch()
        body.addLayout(knobs)

        vu_row = QHBoxLayout()
        self._vu_l = VUMeter(0.0, 0.0)
        self._vu_r = VUMeter(0.0, 0.0)
        vu_row.addWidget(self._vu_l)
        vu_row.addWidget(self._vu_r)
        vu_row.addWidget(_lbl("spectrogram pending", 10, FG3, mono=True))
        vu_row.addStretch()
        body.addLayout(vu_row)

        self.set_available_models(model_paths, selected_model)

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
        self.paramsChanged.emit(self._index, self._gain.value, self._temp.value, self._smooth.value)

    def _emit_model(self, _):
        current = self.current_model()
        self._title.setText(Path(current).name if current else "No model")
        self.modelChanged.emit(self._index, current or "")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "", "TorchScript (*.ts)")
        if not path:
            return

        exists = False
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == path:
                self._model_combo.setCurrentIndex(i)
                exists = True
                break

        if not exists:
            self._model_combo.addItem(Path(path).name, path)
            self._model_combo.setCurrentIndex(self._model_combo.count() - 1)

    def set_vu(self, level: float, peak: float):
        self._vu_l.set_levels(level, peak)
        self._vu_r.set_levels(level * 0.92, min(1.0, peak * 0.95))
