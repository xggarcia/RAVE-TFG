from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from app.ui.widgets.form import ACID, BG0, BG1, FG0, LINE0, MAG


class PhasePad(QWidget):
    xyChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setMouseTracking(True)
        self._x = 0.5
        self._y = 0.5
        self._trail: list[QPointF] = []

    def mousePressEvent(self, event):
        self._set_from_pos(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._set_from_pos(event.position())

    def _set_from_pos(self, pos: QPointF):
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        self._x = max(0.0, min(1.0, pos.x() / w))
        self._y = max(0.0, min(1.0, pos.y() / h))
        self._trail.append(QPointF(pos))
        if len(self._trail) > 20:
            self._trail.pop(0)
        self.xyChanged.emit(self._x, self._y)
        self.update()

    @property
    def xy(self) -> tuple[float, float]:
        return self._x, self._y

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)

        p.setPen(QColor(LINE0))
        p.setBrush(QColor(BG0))
        p.drawRoundedRect(r, 4, 4)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACID + "44"))
        p.drawEllipse(r.left(), r.top(), r.width() // 2, r.height() // 2)
        p.setBrush(QColor("#40d0d844"))
        p.drawEllipse(r.center().x(), r.top(), r.width() // 2, r.height() // 2)
        p.setBrush(QColor(MAG + "44"))
        p.drawEllipse(r.left(), r.center().y(), r.width() // 2, r.height() // 2)

        p.setPen(QPen(QColor(LINE0), 1, Qt.DashLine))
        p.drawLine(r.center().x(), r.top(), r.center().x(), r.bottom())
        p.drawLine(r.left(), r.center().y(), r.right(), r.center().y())

        if len(self._trail) > 1:
            p.setPen(QPen(QColor(FG0 + "77"), 1))
            for i in range(1, len(self._trail)):
                p.drawLine(self._trail[i - 1], self._trail[i])

        cx = r.left() + int(r.width() * self._x)
        cy = r.top() + int(r.height() * self._y)
        p.setPen(QPen(QColor(FG0), 2))
        p.setBrush(QColor(BG1))
        p.drawEllipse(cx - 8, cy - 8, 16, 16)

        p.end()
