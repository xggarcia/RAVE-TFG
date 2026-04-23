from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from app.ui.widgets.form import ACID, BG2, FG0, FG2, FG3, LINE1, MONO


class Knob(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label: str, value: float = 0.5, size: int = 40, accent: str = ACID, parent=None):
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self._size = size
        self._accent = accent

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._dial = _Dial(self._value, size=size, accent=accent)
        self._dial.valueChanged.connect(self._on_dial_changed)
        layout.addWidget(self._dial, 0, Qt.AlignHCenter)

        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(f"color:{FG2}; font-size:9px; {MONO} letter-spacing:1px; background:transparent;")
        layout.addWidget(self._label)

        self._value_lbl = QLabel(f"{self._value:.2f}")
        self._value_lbl.setAlignment(Qt.AlignCenter)
        self._value_lbl.setStyleSheet(f"color:{FG0}; font-size:10px; {MONO} background:transparent;")
        layout.addWidget(self._value_lbl)

    def _on_dial_changed(self, value: float):
        self._value = value
        self._value_lbl.setText(f"{value:.2f}")
        self.valueChanged.emit(value)

    def set_available(self, available: bool):
        self._dial.set_available(available)
        alpha = "ff" if available else "44"
        self._label.setStyleSheet(
            f"color:{FG2}{alpha}; font-size:9px; {MONO} letter-spacing:1px; background:transparent;"
        )
        self._value_lbl.setStyleSheet(
            f"color:{FG0}{alpha}; font-size:10px; {MONO} background:transparent;"
        )
        self._dial.setEnabled(available)

    @property
    def value(self) -> float:
        return self._value


class _Dial(QWidget):
    valueChanged = Signal(float)

    def __init__(self, value: float, size: int, accent: str, parent=None):
        super().__init__(parent)
        self._value = value
        self._accent = accent
        self._available = True
        self.setFixedSize(size, size)

    def set_available(self, available: bool):
        self._available = available
        self.update()

    def mousePressEvent(self, event):
        self._set_from_pos(event.position().y())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._set_from_pos(event.position().y())

    def _set_from_pos(self, y: float):
        h = max(1.0, float(self.height() - 1))
        value = 1.0 - max(0.0, min(1.0, y / h))
        self._value = value
        self.valueChanged.emit(value)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        r = self.rect().adjusted(1, 1, -1, -1)
        cx = r.center().x()
        cy = r.center().y()
        radius = min(r.width(), r.height()) // 2 - 1

        alpha = 255 if self._available else 60
        arc_color = QColor(self._accent)
        arc_color.setAlpha(alpha)
        body_color = QColor(BG2)
        body_color.setAlpha(alpha)
        border_color = QColor(LINE1)
        border_color.setAlpha(alpha)
        needle_color = QColor(FG3)
        needle_color.setAlpha(alpha)

        p.setPen(QPen(border_color, 1))
        p.setBrush(body_color)
        p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        p.setPen(QPen(arc_color, 2))
        start = 225 * 16
        span = int(-(270 * self._value) * 16)
        p.drawArc(cx - radius + 2, cy - radius + 2, (radius - 2) * 2, (radius - 2) * 2, start, span)

        angle = (225 - 270 * self._value) * 3.14159265 / 180.0
        px = cx + int((radius - 6) * __import__("math").cos(angle))
        py = cy - int((radius - 6) * __import__("math").sin(angle))
        p.setPen(QPen(needle_color, 1.5))
        p.drawLine(cx, cy, px, py)

        p.end()
