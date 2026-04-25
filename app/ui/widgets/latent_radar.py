import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from app.ui.widgets.form import ACID, AMBER, LINE0

_MARGIN = 14
_HIT_PX = 24


class LatentRadarWidget(QWidget):
    """Polygon radar chart for per-latent-dim scale editing (drag endpoints)."""

    dimSelected  = Signal(int)        # dim index clicked
    scaleChanged = Signal(int, float) # dim index, new scale 0-2

    def __init__(self, n_dims: int = 16, parent=None):
        super().__init__(parent)
        self._n_dims  = max(1, n_dims)
        self._scales  = [1.0] * self._n_dims
        self._active  = 0
        self._dragging = False
        self.setFixedSize(200, 200)
        self.setMouseTracking(True)

    # ── public API ───────────────────────────────────────────────────────────

    def set_dims(self, n: int):
        if n < 1:
            return
        self._n_dims  = n
        self._scales  = [1.0] * n
        self._active  = 0
        self.update()

    def set_scale(self, dim: int, scale: float):
        if 0 <= dim < self._n_dims:
            self._scales[dim] = max(0.0, min(2.0, scale))
            self.update()

    def set_scales(self, scales: list[float], active_dim: int | None = None):
        if not scales:
            return
        self._n_dims = len(scales)
        self._scales = [max(0.0, min(2.0, float(v))) for v in scales]
        if active_dim is not None:
            self._active = max(0, min(self._n_dims - 1, int(active_dim)))
        else:
            self._active = max(0, min(self._n_dims - 1, self._active))
        self.update()

    def set_active_dim(self, dim: int):
        if 0 <= dim < self._n_dims:
            self._active = dim
            self.update()

    def get_scales(self) -> list[float]:
        return list(self._scales)

    @property
    def active_dim(self) -> int:
        return self._active

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _cx_cy_r(self):
        cx, cy = self.width() / 2.0, self.height() / 2.0
        return cx, cy, min(cx, cy) - _MARGIN

    def _angle(self, dim: int) -> float:
        return (2.0 * math.pi * dim / self._n_dims) - math.pi / 2.0

    def _endpoint(self, dim: int, scale: float, cx: float, cy: float, max_r: float):
        a = self._angle(dim)
        r = (scale / 2.0) * max_r
        return cx + r * math.cos(a), cy + r * math.sin(a)

    def _nearest_dim(self, mx: float, my: float):
        cx, cy, max_r = self._cx_cy_r()
        best_dim, best_dist = 0, float("inf")
        for dim in range(self._n_dims):
            x, y = self._endpoint(dim, self._scales[dim], cx, cy, max_r)
            d = math.hypot(mx - x, my - y)
            if d < best_dist:
                best_dist, best_dim = d, dim
        return best_dim, best_dist

    def _scale_at_mouse(self, dim: int, mx: float, my: float) -> float:
        cx, cy, max_r = self._cx_cy_r()
        a = self._angle(dim)
        proj = (mx - cx) * math.cos(a) + (my - cy) * math.sin(a)
        return max(0.0, min(2.0, (proj / max_r) * 2.0))

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, max_r = self._cx_cy_r()

        # Guide rings — dashed at scale=1.0 (r = max_r/2)
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = max_r * frac
            pen = QPen(QColor(LINE0), 1)
            if frac == 0.5:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # Spokes
        p.setPen(QPen(QColor(LINE0), 1))
        for dim in range(self._n_dims):
            a = self._angle(dim)
            p.drawLine(
                QPointF(cx, cy),
                QPointF(cx + max_r * math.cos(a), cy + max_r * math.sin(a)),
            )

        # Filled polygon
        if self._n_dims >= 2:
            pts = [
                QPointF(*self._endpoint(d, self._scales[d], cx, cy, max_r))
                for d in range(self._n_dims)
            ]
            fill = QColor(ACID)
            fill.setAlpha(40)
            p.setBrush(QBrush(fill))
            p.setPen(QPen(QColor(ACID), 1))
            p.drawPolygon(QPolygonF(pts))

        # Dim endpoint dots
        for dim in range(self._n_dims):
            x, y = self._endpoint(dim, self._scales[dim], cx, cy, max_r)
            if dim == self._active:
                p.setPen(QPen(QColor(AMBER), 2))
                p.setBrush(QBrush(QColor(AMBER)))
                p.drawEllipse(QPointF(x, y), 5.0, 5.0)
            else:
                p.setPen(QPen(QColor(ACID), 1))
                p.setBrush(QBrush(QColor(ACID)))
                p.drawEllipse(QPointF(x, y), 3.0, 3.0)

        p.end()

    # ── mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        dim, dist = self._nearest_dim(event.position().x(), event.position().y())
        if dist < _HIT_PX:
            self._active   = dim
            self._dragging = True
            self.dimSelected.emit(dim)
            self.update()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        scale = self._scale_at_mouse(
            self._active, event.position().x(), event.position().y()
        )
        self._scales[self._active] = scale
        self.scaleChanged.emit(self._active, scale)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
