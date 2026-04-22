from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QColor, QLinearGradient

BG2   = "#28302b"
BG3   = "#2f3832"
FG1   = "#c6cdc3"
LINE0 = "#333c36"
ACID  = "#a8e63d"


class _TrafficDot(QWidget):
    def __init__(self, color: str, border: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._border = QColor(border)
        self.setFixedSize(12, 12)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(self._border)
        p.setBrush(self._color)
        p.drawEllipse(1, 1, 10, 10)
        p.end()


class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self._drag_pos: QPoint | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(0)

        # Traffic lights
        dots = QWidget()
        dl = QHBoxLayout(dots)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(7)
        dl.addWidget(_TrafficDot("#ae3e32", "#7a2820"))  # close (red)
        dl.addWidget(_TrafficDot("#cc9a14", "#9a7210"))  # minimize (amber)
        dl.addWidget(_TrafficDot("#4eb840", "#369028"))  # zoom (green)
        layout.addWidget(dots)

        # Centered title
        title_lbl = QLabel("RAVE-TFG")
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet(
            f"color:{ACID}; font-family:'JetBrains Mono','Consolas',monospace; "
            f"font-size:12px; font-weight:600; letter-spacing:3px;"
        )
        layout.addWidget(title_lbl, 1)

        # Spacer to balance traffic lights
        spacer = QWidget()
        spacer.setFixedWidth(60)
        layout.addWidget(spacer)

        self.setStyleSheet(
            f"TitleBar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {BG3}, stop:1 {BG2}); "
            f"border-bottom: 1px solid {LINE0}; }}"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
