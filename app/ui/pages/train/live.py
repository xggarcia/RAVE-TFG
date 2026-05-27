"""Live training panel: log stream + error panel."""
from PySide6.QtCore import Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.form import (
    ACID,
    AMBER,
    BG0,
    BG1,
    BG2,
    BG3,
    FG1,
    FG3,
    LINE0,
    MAG,
    MONO,
    Panel,
    _lbl,
    section_title,
)
from app.ui.pages.train.monitor import ErrorPanel

LOG_COLORS = {"INFO": ACID, "WARN": AMBER, "ERR": MAG}


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(38)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {BG3}, stop:1 {BG2}); border-bottom:1px solid {LINE0};"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.addWidget(section_title("Log stream"))
        hl.addStretch()
        layout.addWidget(header)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"QTextEdit {{ background:{BG0}; color:{FG1}; {MONO} font-size:10.5px;"
            f"border:none; padding:10px; }}"
        )
        self._log.setMinimumHeight(300)
        layout.addWidget(self._log, 1)

        footer = QWidget()
        footer.setFixedHeight(30)
        footer.setStyleSheet(f"background:{BG1}; border-top:1px solid {LINE0};")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 0, 10, 0)
        self._line_count = _lbl("0 lines", size=10, color=FG3, mono=True)
        fl.addWidget(self._line_count)
        fl.addStretch()
        self._autoscroll = QCheckBox("Autoscroll")
        self._autoscroll.setChecked(True)
        self._autoscroll.setStyleSheet(f"color:{FG1}; font-size:10px; background:transparent;")
        fl.addWidget(self._autoscroll)
        layout.addWidget(footer)

        self._count = 0

    @Slot(str, str)
    def append(self, level: str, message: str):
        color = LOG_COLORS.get(level, FG1)
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt_level = QTextCharFormat()
        fmt_level.setForeground(QColor(color))
        cursor.setCharFormat(fmt_level)
        cursor.insertText(f"[{level}] ")

        fmt_msg = QTextCharFormat()
        fmt_msg.setForeground(QColor(FG1))
        cursor.setCharFormat(fmt_msg)
        cursor.insertText(f"{message}\n")

        self._count += 1
        self._line_count.setText(f"{self._count} lines")

        if self._autoscroll.isChecked():
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def clear(self):
        self._log.clear()
        self._count = 0
        self._line_count.setText("0 lines")


class LivePanel(QWidget):
    """Log stream + error panel, shown once training starts."""

    def __init__(self, max_steps: int = 0, parent=None):
        super().__init__(parent)
        self.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(16)

        log_panel_wrap = Panel()
        self._log_panel = LogPanel()
        log_panel_wrap._root.addWidget(self._log_panel)
        layout.addWidget(log_panel_wrap)

        self._error = ErrorPanel()
        layout.addWidget(self._error)

    def reset(self, max_steps: int = 0):
        self._log_panel.clear()
        self._error.hide()

    @Slot(dict)
    def update_progress(self, state: dict):
        pass

    @Slot(str, str)
    def append_log(self, level: str, message: str):
        self._log_panel.append(level, message)

    @Slot(str, str)
    def show_failure(self, short: str, traceback: str):
        if "out of memory" in traceback.lower() or "OutOfMemoryError" in traceback:
            self._error.show_oom(traceback)
        else:
            self._error.show_error(short, traceback)
