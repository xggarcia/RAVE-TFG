from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen

BG0   = "#1e2320"
FG3   = "#717870"
LINE1 = "#3f4b42"
ACID  = "#a8e63d"
MONO  = "font-family:'JetBrains Mono','Consolas',monospace;"


class PlaceholderPage(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        name_lbl = QLabel(title.upper())
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(
            f"color:{FG3}; font-size:11px; {MONO} letter-spacing:3px;"
        )
        layout.addWidget(name_lbl)

        hint = QLabel("Coming in a future phase")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color:{FG3}; font-size:10px; {MONO} margin-top:6px;")
        layout.addWidget(hint)

    def paintEvent(self, event):
        p = QPainter(self)
        r = self.rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(BG0))
        p.drawRect(r)
        # diagonal stripe pattern
        p.setPen(QPen(QColor(LINE1), 1))
        spacing = 20
        for i in range(-r.height(), r.width() + r.height(), spacing):
            p.drawLine(i, 0, i + r.height(), r.height())
        p.end()
