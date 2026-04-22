"""Live training panel: stat strip + loss chart + log stream."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QCheckBox, QPushButton, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QTextCharFormat, QTextCursor

from app.ui.widgets.form import Panel, section_title, _lbl, BG0, BG1, BG2, BG3, FG0, FG1, FG2, FG3, LINE0, LINE1, ACID, AMBER, MAG, BLUE, MONO

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

_STAT_KEYS = [
    ("Step",     "—", FG0),
    ("of",       "—", FG2),
    ("it/s",     "—", ACID),
    ("Loss",     "—", ACID),
    ("Val loss", "—", FG0),
    ("ETA",      "—", AMBER),
]

LOG_COLORS = {"INFO": ACID, "WARN": AMBER, "ERR": MAG}


class _StatCell(QWidget):
    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(_lbl(label, size=9, color=FG3, mono=True, spacing="1px"))
        self._val = _lbl("—", size=18, color=color, mono=True, bold=True)
        layout.addWidget(self._val)

    def set_value(self, v: str):
        self._val.setText(v)


class StatStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG2}; border-bottom:1px solid {LINE0};")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(0)
        self._cells: dict[str, _StatCell] = {}
        for i, (key, _, color) in enumerate(_STAT_KEYS):
            cell = _StatCell(key, color)
            self._cells[key] = cell
            layout.addWidget(cell, 1)
            if i < len(_STAT_KEYS) - 1:
                sep = QWidget()
                sep.setFixedSize(1, 40)
                sep.setStyleSheet(f"background:{LINE0};")
                layout.addWidget(sep)
                layout.addSpacing(0)

    def update(self, state: dict):
        self._cells["Step"].set_value(f"{state.get('step', 0):,}")
        self._cells["of"].set_value(f"{state.get('max_steps', 0):,}")
        self._cells["it/s"].set_value(f"{state.get('it_s', 0.0):.2f}")
        self._cells["Loss"].set_value(f"{state.get('loss', 0.0):.4f}")
        self._cells["Val loss"].set_value(
            f"{state.get('val_loss', 0.0):.4f}" if state.get('val_loss') else "—"
        )
        self._cells["ETA"].set_value(state.get("eta", "—"))


class ChartPanel(QWidget):
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
        hl.addWidget(section_title("Loss curves"))
        hl.addStretch()
        legend = _lbl("● train  ● val", size=10, color=FG2, mono=True)
        legend.setStyleSheet(f"color:{FG2}; font-size:10px; {MONO} background:transparent;")
        hl.addWidget(legend)
        layout.addWidget(header)

        if _HAS_PG:
            self._plot = pg.PlotWidget(background=BG0)
            self._plot.setMinimumHeight(180)
            self._plot.showGrid(x=False, y=True, alpha=0.15)
            self._plot.getAxis("bottom").setPen(pg.mkPen(LINE0))
            self._plot.getAxis("left").setPen(pg.mkPen(LINE0))
            self._plot.getAxis("bottom").setTextPen(pg.mkPen(FG3))
            self._plot.getAxis("left").setTextPen(pg.mkPen(FG3))
            self._train_curve = self._plot.plot(pen=pg.mkPen(color=ACID, width=1.5))
            self._val_curve   = self._plot.plot(pen=pg.mkPen(color=BLUE, width=1.5))
            self._steps: list[int]   = []
            self._losses: list[float] = []
            self._val_steps: list[int]   = []
            self._val_losses: list[float] = []
            layout.addWidget(self._plot)
        else:
            fallback = _lbl("pyqtgraph not installed — install it to see the loss chart.",
                             size=11, color=FG3)
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setContentsMargins(14, 40, 14, 40)
            layout.addWidget(fallback)

    def add_point(self, step: int, loss: float, val_loss: float | None = None):
        if not _HAS_PG:
            return
        self._steps.append(step)
        self._losses.append(loss)
        self._train_curve.setData(self._steps, self._losses)
        if val_loss is not None and val_loss > 0:
            self._val_steps.append(step)
            self._val_losses.append(val_loss)
            self._val_curve.setData(self._val_steps, self._val_losses)

    def clear(self):
        if not _HAS_PG:
            return
        self._steps.clear(); self._losses.clear()
        self._val_steps.clear(); self._val_losses.clear()
        self._train_curve.setData([], [])
        self._val_curve.setData([], [])


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


class ErrorPanel(QWidget):
    """Shown when training fails — OOM message + suggested fixes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(0)

        # Error banner
        self._banner = QWidget()
        self._banner.setStyleSheet(
            f"background: rgba(62,16,32,0.9); border:1px solid {MAG}; border-radius:6px;"
        )
        bl = QHBoxLayout(self._banner)
        bl.setContentsMargins(16, 12, 16, 12)
        bl.setSpacing(12)
        bl.addWidget(_lbl("⚠", size=16, color=MAG))
        self._err_col = QVBoxLayout()
        self._err_title = _lbl("Error", size=13, color=FG0, bold=True)
        self._err_desc  = _lbl("", size=12, color=FG1, wrap=True)
        self._err_col.addWidget(self._err_title)
        self._err_col.addWidget(self._err_desc)
        bl.addLayout(self._err_col, 1)
        layout.addWidget(self._banner)
        layout.addSpacing(16)

        # Two-col: traceback + suggested fixes
        cols = QHBoxLayout()
        cols.setSpacing(20)

        tb_panel = Panel()
        tb_panel.add_header(section_title("Stack trace"))
        self._tb_edit = QTextEdit()
        self._tb_edit.setReadOnly(True)
        self._tb_edit.setFixedHeight(260)
        self._tb_edit.setStyleSheet(
            f"QTextEdit {{ background:{BG0}; color:{FG1}; {MONO} font-size:10px;"
            f"border:none; padding:14px; white-space:pre; }}"
        )
        tb_panel._root.addWidget(self._tb_edit)
        cols.addWidget(tb_panel, 1)

        fixes_panel = Panel()
        fixes_panel.setFixedWidth(300)
        fixes_panel.add_header(section_title("Suggested fixes"))
        self._fixes_layout = QVBoxLayout()
        self._fixes_layout.setContentsMargins(0, 0, 0, 0)
        self._fixes_layout.setSpacing(0)
        fixes_container = QWidget()
        fixes_container.setStyleSheet("background:transparent;")
        fixes_container.setLayout(self._fixes_layout)
        fixes_panel._root.addWidget(fixes_container)
        cols.addWidget(fixes_panel)

        layout.addLayout(cols)

    def show_oom(self, traceback: str):
        self._err_title.setText("CUDA out of memory")
        self._err_desc.setText(
            "The GPU ran out of memory. The last checkpoint is intact. "
            "Try reducing batch size or disabling causal convolutions."
        )
        self._tb_edit.setPlainText(traceback)
        self._populate_fixes([
            ("Retry with batch_size = 4",         "safest · ~2× slower"),
            ("Enable gradient checkpointing",     "trades compute for memory"),
            ("Disable causal conv",               "frees ~900 MB · changes model"),
            ("Move to CPU offload",               "requires restart · much slower"),
        ])
        self.show()

    def show_error(self, short: str, traceback: str):
        self._err_title.setText(short)
        self._err_desc.setText(traceback[:200] if traceback else "")
        self._tb_edit.setPlainText(traceback)
        self._populate_fixes([])
        self.show()

    def _populate_fixes(self, fixes: list[tuple[str, str]]):
        while self._fixes_layout.count():
            item = self._fixes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (action, sub) in enumerate(fixes):
            row = QWidget()
            row.setStyleSheet(
                f"border-bottom:{'1px solid '+LINE0 if i < len(fixes)-1 else 'none'};"
                f"background:transparent;"
            )
            rl = QVBoxLayout(row)
            rl.setContentsMargins(10, 10, 10, 10)
            rl.setSpacing(2)
            rl.addWidget(_lbl(action, size=12, color=FG0, bold=True))
            rl.addWidget(_lbl(sub, size=10, color=FG3, mono=True))
            self._fixes_layout.addWidget(row)


class LivePanel(QWidget):
    """Stats + chart + log, shown once training starts."""

    def __init__(self, max_steps: int = 0, parent=None):
        super().__init__(parent)
        self._max_steps = max_steps
        self.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(16)

        # Stat strip (inside a Panel)
        stat_panel = Panel()
        self._stats = StatStrip()
        stat_panel._root.addWidget(self._stats)
        layout.addWidget(stat_panel)

        # Chart + log side by side
        cols = QHBoxLayout()
        cols.setSpacing(20)

        # Left: chart panel
        left_panel = Panel()
        self._chart = ChartPanel()
        left_panel._root.addWidget(self._chart)
        cols.addWidget(left_panel, 1)

        # Right: log panel
        log_panel_wrap = Panel()
        log_panel_wrap.setFixedWidth(360)
        self._log_panel = LogPanel()
        log_panel_wrap._root.addWidget(self._log_panel)
        cols.addWidget(log_panel_wrap)

        layout.addLayout(cols)

        self._error = ErrorPanel()
        layout.addWidget(self._error)

    def reset(self, max_steps: int):
        self._max_steps = max_steps
        self._chart.clear()
        self._log_panel.clear()
        self._error.hide()

    @Slot(dict)
    def update_progress(self, state: dict):
        state["max_steps"] = self._max_steps
        self._stats.update(state)
        self._chart.add_point(
            state.get("step", 0),
            state.get("loss", 0.0),
            state.get("val_loss") or None,
        )

    @Slot(str, str)
    def append_log(self, level: str, message: str):
        self._log_panel.append(level, message)

    @Slot(str, str)
    def show_failure(self, short: str, traceback: str):
        if "out of memory" in traceback.lower() or "OutOfMemoryError" in traceback:
            self._error.show_oom(traceback)
        else:
            self._error.show_error(short, traceback)
