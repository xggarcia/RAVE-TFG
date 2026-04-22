from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from app.ui.widgets.form import ACID, AMBER, BG1, LINE0


class VUMeter(QWidget):
    def __init__(self, level: float = 0.0, peak: float = 0.0, parent=None):
        super().__init__(parent)
        self._level = max(0.0, min(1.0, level))
        self._peak = max(0.0, min(1.0, peak))
        self.setMinimumSize(8, 44)

    def set_levels(self, level: float, peak: float):
        self._level = max(0.0, min(1.0, level))
        self._peak = max(0.0, min(1.0, peak))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        r = self.rect().adjusted(1, 1, -1, -1)
        p.setPen(QColor(LINE0))
        p.setBrush(QColor(BG1))
        p.drawRoundedRect(r, 2, 2)

        h = r.height()
        filled = int(h * self._level)
        fill_rect = r.adjusted(1, h - filled, -1, -1)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACID if self._level < 0.8 else AMBER))
        p.drawRoundedRect(fill_rect, 1, 1)

        py = r.y() + h - int(h * self._peak)
        p.setPen(QColor(AMBER))
        p.drawLine(r.x() + 1, py, r.right() - 1, py)

        p.end()
