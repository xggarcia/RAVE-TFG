"""Dataset creation wizard — split-view with 7 sub-flow detail panels."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QStackedWidget, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    PageHeader, section_title, _lbl,
    BG0, BG1, BG2, BG3, FG0, FG1, FG2, FG3, LINE0, LINE1, ACID, ACIDBG, ACIDDIM, MONO,
)
from app.ui.pages._dataset_details import (
    DoAllDetail, FirstDownloadDetail, FinalDownloadDetail,
    NormalizeDetail, MergeDetail, ConvertDetail,
)
from app.ui.pages._dataset_preview import PreviewSelectDetail

STEPS = [
    ("do_all",    "DO ALL",               "full pipeline · query → preview → select → download"),
    ("first",     "First download only",  "query CSV · bulk grab"),
    ("preview",   "Select from previews", "audition · accept / reject"),
    ("final",     "Final download only",  "selected IDs CSV → audio"),
    ("normalize", "Normalize volume",     "target dBFS"),
    ("merge",     "Merge selected CSVs",  "combine selections"),
    ("convert",   "Convert format / SR",  "44.1 kHz · mono · WAV subtype"),
]


class _StepItem(QWidget):
    clicked = Signal(str)

    def __init__(self, index: int, step_id: str, label: str, sub: str, parent=None):
        super().__init__(parent)
        self._active = False
        self._index = index
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
            # acid left indicator
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


class _StepList(QWidget):
    navigate = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, _StepItem] = {}
        self._active = ""
        self.setFixedWidth(260)
        self.setStyleSheet(f"background:{BG1}; border-right:1px solid {LINE0};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(2)
        layout.addWidget(section_title("Pipeline"))
        layout.addSpacing(4)

        for i, (sid, label, sub) in enumerate(STEPS):
            item = _StepItem(i, sid, label, sub)
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


class DatasetPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(PageHeader(
            crumbs=["Data & Training", "Dataset Creation"],
            title="Dataset creation",
            desc="Build a training dataset from a source corpus. Run the whole pipeline, or pick a single step.",
        ))

        # Body: step list + detail stack
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._step_list = _StepList()
        self._step_list.navigate.connect(self._navigate)
        body.addWidget(self._step_list)

        # Detail stack in a scroll area
        self._stack = QStackedWidget()
        self._detail_widgets: dict[str, QWidget] = {}

        do_all = DoAllDetail()
        preview = PreviewSelectDetail()
        do_all.previewReady.connect(
            lambda folder, save_path: (
                preview.load_folder(folder, save_path),
                self._navigate("preview"),
            )
        )
        preview.selectionSaved.connect(
            lambda _path: (
                self._navigate("do_all"),
                do_all._run(),
            )
        )

        panels = [
            ("do_all",    do_all),
            ("first",     FirstDownloadDetail()),
            ("preview",   preview),
            ("final",     FinalDownloadDetail()),
            ("normalize", NormalizeDetail()),
            ("merge",     MergeDetail()),
            ("convert",   ConvertDetail()),
        ]
        for sid, widget in panels:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")
            wrapper = QWidget()
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(24, 24, 24, 24)
            wl.addWidget(widget)
            scroll.setWidget(wrapper)
            self._stack.addWidget(scroll)
            self._detail_widgets[sid] = scroll

        body.addWidget(self._stack, 1)
        root_widget = QWidget()
        root_widget.setLayout(body)
        root.addWidget(root_widget, 1)

        self._navigate("do_all")

    def _navigate(self, step_id: str):
        if step_id not in self._detail_widgets:
            return
        self._stack.setCurrentWidget(self._detail_widgets[step_id])
        self._step_list.set_active(step_id)
