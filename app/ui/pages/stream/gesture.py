"""Gesture draw widget for the streaming page."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath
from PySide6.QtWidgets import QWidget

from app.ui.widgets.form import ACID, BG1, LINE0


class _GestureDrawWidget(QWidget):
    curveChanged = Signal(list)

    def __init__(self, points=None, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(86)
        self.setMaximumHeight(86)
        self.setMouseTracking(True)
        self._drawing = False
        self._points = list(points) if points else [(0.0, 0.5), (1.0, 0.5)]

    def clear_curve(self):
        self._points = [(0.0, 0.5), (1.0, 0.5)]
        self.curveChanged.emit(self.get_curve())
        self.update()

    def set_curve(self, points: list[tuple[float, float]]):
        try:
            pts = [(float(x), float(y)) for x, y in points]
        except Exception:
            return
        self._points = pts if pts else [(0.0, 0.5), (1.0, 0.5)]
        self.update()

    def get_curve(self):
        pts = []
        for p in self._points:
            try:
                x = max(0.0, min(1.0, float(p[0])))
                y = max(0.0, min(1.0, float(p[1])))
                pts.append((x, y))
            except Exception:
                continue
        if not pts:
            pts = [(0.0, 0.5), (1.0, 0.5)]
        pts.sort(key=lambda v: v[0])
        if pts[0][0] > 0.0:
            pts.insert(0, (0.0, pts[0][1]))
        if pts[-1][0] < 1.0:
            pts.append((1.0, pts[-1][1]))
        return [(float(x), float(y)) for x, y in pts]

    def _to_norm(self, x_px: int, y_px: int):
        w = max(1, self.width() - 1)
        h = max(1, self.height() - 1)
        x = max(0.0, min(1.0, float(x_px) / w))
        y = max(0.0, min(1.0, 1.0 - (float(y_px) / h)))
        return (x, y)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._drawing = True
        x, y = self._to_norm(int(event.position().x()), int(event.position().y()))
        self._points = [(x, y)]
        self.curveChanged.emit(self.get_curve())
        self.update()

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        x, y = self._to_norm(int(event.position().x()), int(event.position().y()))
        if self._points:
            px, py = self._points[-1]
            if abs(px - x) < 0.002 and abs(py - y) < 0.002:
                return
        self._points.append((x, y))
        self.curveChanged.emit(self.get_curve())
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()

        p.setPen(QPen(QColor(LINE0), 1))
        p.setBrush(QColor(BG1))
        p.drawRoundedRect(r.adjusted(0, 0, -1, -1), 4, 4)

        p.setPen(QPen(QColor(LINE0), 1, Qt.DashLine))
        y_mid = int(r.top() + r.height() * 0.5)
        p.drawLine(r.left() + 4, y_mid, r.right() - 4, y_mid)
        for frac in (0.25, 0.5, 0.75):
            x = int(r.left() + r.width() * frac)
            p.drawLine(x, r.top() + 4, x, r.bottom() - 4)

        curve = self.get_curve()
        if curve:
            path = QPainterPath()
            x0 = r.left() + curve[0][0] * r.width()
            y0 = r.bottom() - curve[0][1] * r.height()
            path.moveTo(x0, y0)
            for x, y in curve[1:]:
                px = r.left() + x * r.width()
                py = r.bottom() - y * r.height()
                path.lineTo(px, py)

            p.setPen(QPen(QColor(ACID), 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        p.end()
