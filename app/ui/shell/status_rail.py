from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter

LINE0 = "#333c36"
BG1   = "#232926"
FG2   = "#96a092"
FG3   = "#717870"
ACID  = "#a8e63d"


class _Dot(QWidget):
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(6, 6)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(0, 0, 6, 6)
        p.end()


class StatusRail(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self.setStyleSheet(
            f"StatusRail {{ background:{BG1}; border-top: 1px solid {LINE0}; }}"
        )
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(12, 0, 12, 0)
        self._layout.setSpacing(0)

        self._set_defaults()

    def _set_defaults(self):
        self._status_lbl = self._add_item("Idle", FG3)
        self._add_separator()
        self._add_item("GPU OK", ACID)
        self._add_separator()
        self._add_item("~/rave-tfg/workspace", FG3)
        self._layout.addStretch()
        hint = QLabel("Ctrl+K  command")
        hint.setStyleSheet(
            f"color:{FG3}; font-family:'JetBrains Mono','Consolas',monospace; font-size:10px;"
        )
        self._layout.addWidget(hint)

    def _add_item(self, text: str, color: str):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{color}; font-family:'JetBrains Mono','Consolas',monospace; "
            f"font-size:10px; letter-spacing:1px;"
        )
        self._layout.addWidget(lbl)
        return lbl

    def _add_separator(self):
        sep = QWidget()
        sep.setFixedSize(1, 14)
        sep.setStyleSheet(f"background:{LINE0};")
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(12, 0, 12, 0)
        hl.addWidget(sep)
        self._layout.addWidget(w)

    def set_status(self, text: str, color: str = FG3):
        """Update the first status item programmatically."""
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"color:{color}; font-family:'JetBrains Mono','Consolas',monospace; "
            f"font-size:10px; letter-spacing:1px;"
        )
