"""Training config constants and the extra-config chip widget."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

from app.ui.widgets.form import ACID, BG1, FG1, FG3, LINE1, MONO, _lbl

EXTRA_CONFIGS = [
    ("causal", "causal convolutions"),
    ("noise",  "noise synthesizer V2"),
]
DEFAULT_ON = {"noise", "causal"}


class _ConfigChip(QWidget):
    toggled = Signal(bool)

    def __init__(self, key: str, desc: str, on: bool = False, parent=None):
        super().__init__(parent)
        self.key = key
        self._on = on
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(36)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumWidth(80)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(6)
        self._key_lbl = _lbl(key, size=11, color=ACID if on else FG1, mono=True)
        self._desc_lbl = _lbl(f"· {desc}", size=10, color=FG3)
        layout.addWidget(self._key_lbl)
        layout.addWidget(self._desc_lbl)

    @property
    def is_on(self) -> bool:
        return self._on

    def mousePressEvent(self, event):
        self._on = not self._on
        self._key_lbl.setStyleSheet(
            f"color:{ACID if self._on else FG1}; font-size:11px; {MONO} background:transparent;"
        )
        self.update()
        self.toggled.emit(self._on)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        if self._on:
            p.setPen(QPen(QColor("#3a4f1a"), 1))
            p.setBrush(QColor("#1e2d0a"))
        else:
            p.setPen(QPen(QColor(LINE1), 1))
            p.setBrush(QColor(BG1))
        p.drawRoundedRect(r, 3, 3)
        p.end()
