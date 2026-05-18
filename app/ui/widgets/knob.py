"""Knob — thin wrapper around QDial that keeps the previous API."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDial, QLabel, QVBoxLayout, QWidget

from app.ui.widgets.form import ACID, FG0, FG2, MONO


class Knob(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label: str, value: float = 0.5, size: int = 40, accent: str = ACID, parent=None):
        super().__init__(parent)
        self._accent = accent
        self._value = max(0.0, min(1.0, value))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._dial = QDial()
        self._dial.setRange(0, 1000)
        self._dial.setNotchesVisible(False)
        self._dial.setFixedSize(size, size)
        self._dial.setValue(int(self._value * 1000))
        self._dial.valueChanged.connect(self._on_dial_changed)
        self._dial.setStyleSheet(f"QDial {{ background:transparent; }}")
        layout.addWidget(self._dial, 0, Qt.AlignHCenter)

        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(f"color:{FG2}; font-size:9px; {MONO} letter-spacing:1px; background:transparent;")
        layout.addWidget(self._label)

        self._value_lbl = QLabel(f"{self._value:.2f}")
        self._value_lbl.setAlignment(Qt.AlignCenter)
        self._value_lbl.setStyleSheet(f"color:{FG0}; font-size:10px; {MONO} background:transparent;")
        layout.addWidget(self._value_lbl)

    def _on_dial_changed(self, raw: int):
        value = raw / 1000.0
        self._value = value
        self._value_lbl.setText(f"{value:.2f}")
        self.valueChanged.emit(value)

    def set_available(self, available: bool):
        self._dial.setEnabled(available)
        alpha = "ff" if available else "44"
        self._label.setStyleSheet(
            f"color:{FG2}{alpha}; font-size:9px; {MONO} letter-spacing:1px; background:transparent;"
        )
        self._value_lbl.setStyleSheet(
            f"color:{FG0}{alpha}; font-size:10px; {MONO} background:transparent;"
        )

    @property
    def value(self) -> float:
        return self._value

    def set_value(self, value: float):
        """Set value programmatically without emitting valueChanged."""
        value = max(0.0, min(1.0, value))
        self._value = value
        self._dial.blockSignals(True)
        self._dial.setValue(int(value * 1000))
        self._dial.blockSignals(False)
        self._value_lbl.setText(f"{value:.2f}")
