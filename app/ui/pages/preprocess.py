from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QPushButton, QLabel, QLineEdit, QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt

from app.ui.widgets.form import (
    PageHeader, Panel, Field, FileInput, RadioGroup, Toggle, section_title, _lbl,
    BG0, BG1, BG2, FG0, FG1, FG2, FG3, LINE0, ACID, AMBER, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.preprocess_worker import PreprocessWorker


class PreprocessPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PreprocessWorker | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Actions
        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setProperty("role", "primary")
        self._start_btn.setFixedHeight(30)
        self._start_btn.clicked.connect(self._start)

        root.addWidget(PageHeader(
            crumbs=["Data & Training", "Preprocess"],
            title="Preprocess dataset",
            desc="Convert an audio folder into an LMDB training dataset.",
            actions=[self._start_btn],
        ))

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(20)

        # Two-column layout
        cols = QHBoxLayout()
        cols.setSpacing(20)
        cols.addWidget(self._build_params(), 1)
        cols.addWidget(self._build_scan_preview(), 0)
        cl.addLayout(cols)

        # Progress panel (hidden until started)
        self._progress = ProgressPanel()
        cl.addWidget(self._progress)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    # ── Parameters panel ──────────────────────────────────────────────────────

    def _build_params(self) -> QWidget:
        panel = Panel()
        panel.add_header(section_title("Parameters"))
        body = panel.body_layout()

        # Audio folder
        self._audio_input = FileInput(directory=True, placeholder="~/datasets/audio/")
        body.addWidget(Field("Audio folder", hint="Recursively scans for .wav, .flac, .mp3", inline=True).add(self._audio_input))

        # Channels
        self._channels = RadioGroup(
            options=[{"value": "1", "label": "Mono (1)"}, {"value": "2", "label": "Stereo (2)"}],
            value="1",
        )
        body.addWidget(Field("Channels", inline=True).add(self._channels))

        # Max DB size
        size_row = QWidget()
        size_row.setStyleSheet("background:transparent;")
        sl = QHBoxLayout(size_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        self._max_db = QLineEdit("8")
        self._max_db.setFixedWidth(70)
        self._max_db.setStyleSheet(
            f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
            f"border:1px solid #3f4b42; border-radius:4px; padding:6px 10px;"
        )
        self._db_unit = QComboBox()
        self._db_unit.addItems(["GB", "MB"])
        self._db_unit.setFixedWidth(60)
        sl.addWidget(self._max_db)
        sl.addWidget(self._db_unit)
        sl.addStretch()
        body.addWidget(Field("Max DB size", hint="Caps the LMDB file at this size; 0 = unlimited", inline=True).add(size_row))

        # Lazy loading
        self._lazy = Toggle(on=False)
        body.addWidget(Field("Lazy loading", hint="Stream audio at train time. Lower disk use, slower training.", inline=True).add(self._lazy))

        # Output path
        self._output_input = FileInput(directory=True, placeholder="~/preprocessed/")
        body.addWidget(Field("Output path", inline=True).add(self._output_input))

        body.addStretch()
        return panel

    # ── Scan preview panel ────────────────────────────────────────────────────

    def _build_scan_preview(self) -> QWidget:
        panel = Panel()
        panel.setFixedWidth(300)
        panel.add_header(section_title("Scan preview"))
        body = panel.body_layout()

        rows = [
            ("Files",        "—"),
            ("Total",        "—"),
            ("Avg length",   "—"),
            ("Sample rate",  "—"),
            ("Channels",     "—"),
            ("Disk",         "—"),
        ]
        for key, val in rows:
            row = QHBoxLayout()
            row.addWidget(_lbl(key, size=10, color=FG3, mono=True, spacing="1px"))
            row.addStretch()
            val_lbl = _lbl(val, size=11, color=FG0, mono=True)
            row.addWidget(val_lbl)
            body.addLayout(row)

        hint = _lbl("Select an audio folder and click Start to populate.", size=10, color=FG3)
        hint.setWordWrap(True)
        body.addWidget(hint)
        body.addStretch()
        return panel

    # ── Actions ───────────────────────────────────────────────────────────────

    def _start(self):
        if self._worker and self._worker.isRunning():
            return

        audio = self._audio_input.path
        if not audio:
            self._progress.start("Waiting…")
            self._progress.append_log("Please select an audio folder first.")
            self._progress.finish(False, "No folder selected.")
            return

        channels = 1 if self._channels.value.startswith("M") or self._channels.value == "1" else 2
        try:
            max_db = int(self._max_db.text())
        except ValueError:
            max_db = 8

        self._start_btn.setEnabled(False)
        self._progress.start("Preprocessing dataset…")

        self._worker = PreprocessWorker(
            audio_path=audio,
            channels=channels,
            lazy=self._lazy.is_on,
            max_db_size=max_db,
        )
        self._worker.log.connect(self._progress.append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, success: bool, message: str):
        self._progress.finish(success, message)
        self._start_btn.setEnabled(True)
