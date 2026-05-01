"""Train page helper widgets: resume banner, step list, step items."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from app.ui.widgets.form import (
    ACID,
    AMBER,
    BG1,
    BG2,
    BG3,
    FG0,
    FG1,
    FG2,
    FG3,
    LINE0,
    LINE1,
    MONO,
    _lbl,
    section_title,
)

CONFIGS = ["v2_small", "v2", "v3", "v3_small"]

PIPELINE_STEPS = [
    ("do_all",     "DO ALL",    "preprocess → train → export"),
    ("preprocess", "Preprocess","dataset preparation only"),
    ("train_only", "Train",     "model optimization only"),
    ("export",     "Export",    "torchscript export only"),
]


class _ResumeBanner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        self.setStyleSheet(
            f"background: rgba(80,60,20,0.5); border:1px solid {AMBER}; border-radius:4px;"
        )
        hl = QHBoxLayout(self)
        hl.setContentsMargins(14, 10, 14, 10)
        hl.setSpacing(12)
        hl.addWidget(_lbl("⚠", size=13, color=AMBER))
        self._msg = _lbl("Resumable checkpoint found", size=12, color=FG0)
        hl.addWidget(self._msg, 1)
        self._fresh_btn = QPushButton("Start fresh")
        self._fresh_btn.setFixedHeight(28)
        self._resume_btn = QPushButton("Resume")
        self._resume_btn.setProperty("role", "primary")
        self._resume_btn.setFixedHeight(28)
        hl.addWidget(self._fresh_btn)
        hl.addWidget(self._resume_btn)
        self.setFixedHeight(52)

    def show_checkpoint(self, path: str):
        self._msg.setText(f"Resumable checkpoint found  {path}")
        self.show()

    @property
    def resume_btn(self) -> QPushButton:
        return self._resume_btn

    @property
    def fresh_btn(self) -> QPushButton:
        return self._fresh_btn


class _TrainStepItem(QWidget):
    clicked = Signal(str)

    def __init__(self, index: int, step_id: str, label: str, sub: str, parent=None):
        super().__init__(parent)
        self._active = False
        self._id = step_id
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(52)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(10, 8, 10, 8)
        hl.setSpacing(10)

        self._num = QWidget()
        self._num.setFixedSize(24, 24)
        self._num_lbl = _lbl(str(index + 1).zfill(2), size=10, color=FG2, mono=True, bold=True)
        self._num_lbl.setAlignment(Qt.AlignCenter)
        nl = QVBoxLayout(self._num)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(self._num_lbl)
        hl.addWidget(self._num)

        col = QVBoxLayout()
        col.setSpacing(2)
        self._lbl = _lbl(label, size=12, color=FG1)
        self._sub = _lbl(sub, size=10, color=FG3, mono=True)
        col.addWidget(self._lbl)
        col.addWidget(self._sub)
        hl.addLayout(col)

    def set_active(self, active: bool):
        self._active = active
        color = FG0 if active else FG1
        num_bg = ACID if active else BG2
        num_color = "#1e2320" if active else FG2
        self._lbl.setStyleSheet(f"color:{color}; font-size:12px; background:transparent;")
        self._num_lbl.setStyleSheet(f"color:{num_color}; font-size:10px; {MONO} font-weight:600; background:transparent;")
        self._num.setStyleSheet(f"background:{num_bg}; border-radius:12px;")
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        if self._active:
            p.setPen(QPen(QColor(LINE1), 1))
            p.setBrush(QColor(BG3))
            p.drawRoundedRect(r, 3, 3)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACID))
            p.drawRoundedRect(-1, 6, 3, r.height() - 12, 2, 2)
        p.end()

    def enterEvent(self, e):
        if not self._active:
            self.setStyleSheet(f"background:{BG3}; border-radius:3px;")

    def leaveEvent(self, e):
        self.setStyleSheet("")
        self.update()

    def mousePressEvent(self, e):
        self.clicked.emit(self._id)


class _TrainStepList(QWidget):
    navigate = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, _TrainStepItem] = {}
        self._active = ""
        self.setFixedWidth(260)
        self.setStyleSheet(f"background:{BG1}; border-right:1px solid {LINE0};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(2)
        layout.addWidget(section_title("Pipeline"))
        layout.addSpacing(4)

        for i, (sid, label, sub) in enumerate(PIPELINE_STEPS):
            item = _TrainStepItem(i, sid, label, sub)
            item.clicked.connect(self.navigate.emit)
            self._items[sid] = item
            layout.addWidget(item)

        layout.addStretch()

    def set_active(self, step_id: str):
        if self._active and self._active in self._items:
            self._items[self._active].set_active(False)
        self._active = step_id
        if step_id in self._items:
            self._items[step_id].set_active(True)
