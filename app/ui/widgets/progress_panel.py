"""ProgressPanel — log stream + progress bar shown while a job runs."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QProgressBar, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QTextCharFormat, QColor as QC

BG0   = "#1e2320"
BG2   = "#28302b"
BG3   = "#2f3832"
FG0   = "#f0f4ee"
FG1   = "#c6cdc3"
FG2   = "#96a092"
FG3   = "#717870"
LINE0 = "#333c36"
LINE1 = "#3f4b42"
ACID  = "#a8e63d"
AMBER = "#e8c040"
MAG   = "#e0406a"
MONO  = "font-family:'JetBrains Mono','Consolas',monospace;"


class ProgressPanel(QWidget):
    """Shows while a background job runs: status dot, progress bar, log stream."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 16, 0, 0)
        root.setSpacing(0)

        # Card
        card = QWidget()
        card.setStyleSheet(f"background:{BG2}; border:1px solid {LINE0}; border-radius:6px;")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(38)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {BG3}, stop:1 {BG2});"
            f"border-bottom:1px solid {LINE0};"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.setSpacing(8)

        self._dot = _BlinkDot()
        hl.addWidget(self._dot)
        self._title_lbl = QLabel("Running…")
        self._title_lbl.setStyleSheet(f"color:{FG0}; font-size:12px; font-weight:600; background:transparent;")
        hl.addWidget(self._title_lbl)
        hl.addStretch()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color:{FG2}; font-size:10px; {MONO} background:transparent;")
        hl.addWidget(self._status_lbl)

        cl.addWidget(header)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate by default
        self._bar.setFixedHeight(3)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            f"QProgressBar {{ background:{BG0}; border:none; border-radius:0; }}"
            f"QProgressBar::chunk {{ background:{ACID}; }}"
        )
        cl.addWidget(self._bar)

        # Log area
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(200)
        self._log.setStyleSheet(
            f"QPlainTextEdit {{ background:{BG0}; color:{FG1}; {MONO} font-size:11px;"
            f"border:none; padding:10px; }}"
        )
        cl.addWidget(self._log)

        # Footer: result message
        self._result = QLabel("")
        self._result.setStyleSheet(f"color:{FG2}; font-size:12px; background:transparent; padding:8px 14px;")
        self._result.hide()
        cl.addWidget(self._result)

        root.addWidget(card)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, title: str = "Running…"):
        self._title_lbl.setText(title)
        self._status_lbl.setText("")
        self._log.clear()
        self._result.hide()
        self._bar.setRange(0, 0)
        self._dot.set_color(ACID)
        self.show()

    @Slot(str)
    def append_log(self, line: str):
        self._log.appendPlainText(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    @Slot(str, str)
    def append_log_level(self, level: str, line: str):
        color = AMBER if level == "WARN" else MAG if level == "ERR" else ACID
        self.append_log(f"[{level}] {line}")

    def set_status(self, text: str):
        self._status_lbl.setText(text)

    def set_progress(self, value: int, maximum: int):
        self._bar.setRange(0, maximum)
        self._bar.setValue(value)

    def finish(self, success: bool, message: str):
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        color = ACID if success else MAG
        self._dot.set_color(color)
        self._result.setText(message)
        self._result.setStyleSheet(f"color:{color}; font-size:12px; background:transparent; padding:8px 14px;")
        self._result.show()


class _BlinkDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(ACID)
        self.setFixedSize(8, 8)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(0, 0, 8, 8)
        p.end()
