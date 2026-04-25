from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame, QPushButton, QLineEdit,
)
from PySide6.QtCore import Qt

from app.ui.widgets.form import (
    PageHeader, Panel, Field, FileInput, Toggle, section_title, _lbl,
    BG0, BG1, FG0, FG1, FG2, FG3, LINE1, ACID, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.export_worker import ExportWorker

RUN_META = [
    ("Name",        "—"),
    ("Config",      "—"),
    ("Last step",   "—"),
    ("Best val",    "—"),
    ("Checkpoint",  "—"),
    ("Trained",     "—"),
]


class ExportPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: ExportWorker | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._export_btn = QPushButton("⬆  Export")
        self._export_btn.setProperty("role", "primary")
        self._export_btn.setFixedHeight(30)
        self._export_btn.clicked.connect(self._start)

        root.addWidget(PageHeader(
            crumbs=["Data & Training", "Export Model"],
            title="Export model",
            desc="Convert a training run into a deployable TorchScript .ts file.",
            actions=[self._export_btn],
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(20)

        cols = QHBoxLayout()
        cols.setSpacing(20)
        cols.addWidget(self._build_params(), 1)
        cols.addWidget(self._build_run_info(), 0)
        cl.addLayout(cols)

        self._progress = ProgressPanel()
        cl.addWidget(self._progress)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_params(self) -> QWidget:
        panel = Panel()
        panel.add_header(section_title("Parameters"))
        body = panel.body_layout()

        self._run_input = FileInput(directory=True, placeholder="~/runs/model/version_0/")
        body.addWidget(Field("Run folder", hint="Auto-detected from ~/runs. Pick a run to export.", inline=True).add(self._run_input))

        self._streaming = Toggle(on=True)
        body.addWidget(Field("Streaming", hint="Compiles a streaming-capable model (required for real-time use).", inline=True).add(self._streaming))

        self._name_edit = QLineEdit("model_streaming.ts")
        self._name_edit.setStyleSheet(
            f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
            f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
        )
        body.addWidget(Field("Output name", inline=True).add(self._name_edit))

        self._dest_input = FileInput(directory=True, placeholder="~/exports/")
        body.addWidget(Field("Destination", inline=True).add(self._dest_input))

        body.addStretch()
        return panel

    def _build_run_info(self) -> QWidget:
        panel = Panel()
        panel.setFixedWidth(300)
        panel.add_header(section_title("Detected run"))
        body = panel.body_layout()

        for key, val in RUN_META:
            row = QHBoxLayout()
            row.addWidget(_lbl(key, size=10, color=FG3, mono=True, spacing="1px"))
            row.addStretch()
            row.addWidget(_lbl(val, size=11, color=FG0, mono=True))
            body.addLayout(row)

        hint = _lbl("Select a run folder to see details.", size=10, color=FG3)
        hint.setWordWrap(True)
        body.addWidget(hint)
        body.addStretch()
        return panel

    def _start(self):
        if self._worker and self._worker.isRunning():
            return

        self._export_btn.setEnabled(False)
        self._progress.start("Exporting model…")

        self._worker = ExportWorker(
            run_path=self._run_input.path,
            output_name=self._name_edit.text(),
            destination=self._dest_input.path,
            streaming=self._streaming.is_on,
        )
        self._worker.log.connect(self._progress.append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, success: bool, message: str):
        self._progress.finish(success, message)
        self._export_btn.setEnabled(True)
