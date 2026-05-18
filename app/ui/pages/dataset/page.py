"""Dataset creation wizard — split-view with 7 sub-flow detail panels."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.tokens import ACID, BG0, BG1, BG3, FG0, FG2, FG3, LINE0, MONO
from app.ui.widgets.form import PageHeader, section_title
from app.ui.pages.dataset.details import DoAllDetail
from app.ui.pages.dataset.step_panels import (
    ConvertDetail,
    FinalDownloadDetail,
    FirstDownloadDetail,
    MergeDetail,
    NormalizeDetail,
)
from app.ui.pages.dataset.preview import PreviewSelectDetail

STEPS = [
    ("do_all",    "DO ALL",               "full pipeline · query → preview → select → download"),
    ("first",     "First download only",  "query CSV · bulk grab"),
    ("preview",   "Select from previews", "audition · accept / reject"),
    ("final",     "Final download only",  "selected IDs CSV → audio"),
    ("normalize", "Normalize volume",     "target dBFS"),
    ("merge",     "Merge selected CSVs",  "combine selections"),
    ("convert",   "Convert format / SR",  "44.1 kHz · mono · WAV subtype"),
]


_LIST_QSS = f"""
QListWidget {{
    background: {BG1};
    border: none;
    border-right: 1px solid {LINE0};
    outline: none;
    padding: 8px 4px;
}}
QListWidget::item {{
    color: {FG0};
    padding: 8px 12px;
    margin: 1px 6px;
    border-radius: 3px;
}}
QListWidget::item:hover {{
    background: {BG3};
}}
QListWidget::item:selected {{
    background: {BG3};
    color: {FG0};
    border-left: 3px solid {ACID};
}}
"""


class DatasetPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(PageHeader(
            crumbs=["Core Workflow", "Dataset Creation"],
            title="Dataset creation",
            desc="Build a training dataset from a source corpus. Run the whole pipeline, or pick a single step.",
        ))

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._step_list = QListWidget()
        self._step_list.setFixedWidth(260)
        self._step_list.setStyleSheet(_LIST_QSS)
        for i, (sid, label, sub) in enumerate(STEPS):
            item = QListWidgetItem(f"{i + 1:02d}  {label}\n      {sub}")
            item.setData(Qt.UserRole, sid)
            self._step_list.addItem(item)
        self._step_list.currentRowChanged.connect(self._on_row_changed)
        body.addWidget(self._step_list)

        self._stack = QStackedWidget()
        self._step_index: dict[str, int] = {}

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
            self._step_index[sid] = self._stack.addWidget(scroll)

        body.addWidget(self._stack, 1)
        root_widget = QWidget()
        root_widget.setLayout(body)
        root.addWidget(root_widget, 1)

        self._navigate("do_all")

    def _navigate(self, step_id: str):
        if step_id not in self._step_index:
            return
        self._stack.setCurrentIndex(self._step_index[step_id])
        for row in range(self._step_list.count()):
            if self._step_list.item(row).data(Qt.UserRole) == step_id:
                self._step_list.setCurrentRow(row)
                break

    def _on_row_changed(self, row: int):
        if row < 0:
            return
        item = self._step_list.item(row)
        if item is None:
            return
        sid = item.data(Qt.UserRole)
        if sid in self._step_index:
            self._stack.setCurrentIndex(self._step_index[sid])
