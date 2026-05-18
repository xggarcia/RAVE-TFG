"""VUMeter — vertical QProgressBar with a peak indicator overlaid."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QProgressBar

from app.ui.widgets.form import ACID, AMBER, BG1, LINE0


_QSS = (
    f"QProgressBar {{ background:{BG1}; border:1px solid {LINE0}; border-radius:2px; }}"
    f"QProgressBar::chunk {{ background:{ACID}; }}"
)


class VUMeter(QProgressBar):
    def __init__(self, level: float = 0.0, peak: float = 0.0, parent=None):
        super().__init__(parent)
        self.setOrientation(Qt.Vertical)
        self.setRange(0, 1000)
        self.setTextVisible(False)
        self.setMinimumSize(8, 44)
        self.setStyleSheet(_QSS)

        self._peak = max(0.0, min(1.0, peak))
        self.setValue(int(max(0.0, min(1.0, level)) * 1000))

    def set_levels(self, level: float, peak: float):
        level = max(0.0, min(1.0, level))
        self._peak = max(0.0, min(1.0, peak))
        # Switch chunk color to amber when hot.
        chunk_color = AMBER if level >= 0.8 else ACID
        self.setStyleSheet(
            f"QProgressBar {{ background:{BG1}; border:1px solid {LINE0}; border-radius:2px; }}"
            f"QProgressBar::chunk {{ background:{chunk_color}; }}"
        )
        self.setValue(int(level * 1000))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._peak <= 0.0:
            return
        p = QPainter(self)
        r = self.rect().adjusted(1, 1, -1, -1)
        h = r.height()
        py = r.y() + h - int(h * self._peak)
        p.setPen(QColor(AMBER))
        p.drawLine(r.x() + 1, py, r.right() - 1, py)
        p.end()
