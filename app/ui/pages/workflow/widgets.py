from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from app.ui.widgets.form import ACID, AMBER, BG0, BG1, BG2, FG0, FG2, FG3, LINE0, MAG


@dataclass
class StageState:
    status: str = "pending"
    meta: str = ""
    progress: float | None = None


class StageRow(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._status = "pending"
        self._meta = ""
        self._progress = None
        self.setMinimumHeight(74)

    def update_state(self, state: StageState):
        self._status = state.status
        self._meta = state.meta
        self._progress = state.progress
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG2))

        x = 18
        y = self.height() // 2 - 12
        color = ACID if self._status == "done" else AMBER if self._status == "running" else MAG if self._status == "failed" else FG3

        p.setPen(QPen(QColor(color), 1.2))
        p.setBrush(QColor(BG1))
        p.drawEllipse(x, y, 24, 24)
        p.setPen(QPen(QColor(color), 2))
        if self._status == "done":
            p.drawLine(x + 7, y + 13, x + 11, y + 17)
            p.drawLine(x + 11, y + 17, x + 18, y + 8)
        elif self._status == "running":
            p.setBrush(QColor(color))
            p.drawEllipse(x + 10, y + 10, 4, 4)
        else:
            p.drawLine(x + 8, y + 12, x + 16, y + 12)

        p.setPen(QColor(FG0))
        p.drawText(56, y + 13, self._title)
        p.setPen(QColor(FG2))
        p.drawText(56, y + 30, self._meta)

        if self._progress is not None:
            bar_x = 56
            bar_y = y + 38
            bar_w = max(10, self.width() - 80)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(BG0))
            p.drawRoundedRect(bar_x, bar_y, bar_w, 5, 2, 2)
            p.setBrush(QColor(color))
            p.drawRoundedRect(bar_x, bar_y, int(bar_w * max(0.0, min(1.0, self._progress))), 5, 2, 2)

        p.setPen(QPen(QColor(LINE0), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()