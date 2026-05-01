"""Loss chart and error panel widgets for the live training view."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.form import (
    ACID,
    BG0,
    BG2,
    BG3,
    BLUE,
    FG0,
    FG1,
    FG2,
    FG3,
    LINE0,
    MAG,
    MONO,
    Panel,
    _lbl,
    section_title,
)

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False


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
            self._steps: list[int]    = []
            self._losses: list[float] = []
            self._val_steps: list[int]    = []
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


class ErrorPanel(QWidget):
    """Shown when training fails — OOM message + suggested fixes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 16, 0, 0)
        layout.setSpacing(0)

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
