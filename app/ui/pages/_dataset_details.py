"""Dataset wizard detail panels: DoAll, First, Final, Normalize, Merge, Convert."""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QSlider, QScrollArea, QFrame, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    Panel, Field, FileInput, RadioGroup, Toggle, section_title, _lbl,
    BG0, BG1, BG2, BG3, FG0, FG1, FG2, FG3, LINE0, LINE1, ACID, ACIDDIM, AMBER, MAG, BLUE, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.dataset_worker import DatasetWorker


def _input(default: str = "", width: int | None = None) -> QLineEdit:
    w = QLineEdit(default)
    w.setStyleSheet(
        f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
        f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
    )
    if width:
        w.setFixedWidth(width)
    return w


class _DetailHeader(QWidget):
    run_clicked = Signal()

    def __init__(self, title: str, sub: str = "", action: str = "Run", parent=None):
        super().__init__(parent)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 14)
        hl.setSpacing(16)
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(_lbl(title, size=17, color=FG0, bold=True))
        if sub:
            col.addWidget(_lbl(sub, size=12, color=FG2, wrap=True))
        hl.addLayout(col, 1)
        btn = QPushButton(f"▶  {action}")
        btn.setProperty("role", "primary")
        btn.setFixedHeight(30)
        btn.clicked.connect(self.run_clicked)
        hl.addWidget(btn)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setPen(QPen(QColor(LINE0), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


# ── Stage spine (for DoAll) ───────────────────────────────────────────────────

class _SpineRow(QWidget):
    COLORS = {"done": ACID, "running": AMBER, "pending": FG3}

    def __init__(self, label: str, meta: str, status: str, last: bool, parent=None):
        super().__init__(parent)
        self._last = last
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 14)
        hl.setSpacing(12)

        dot = QWidget()
        dot.setFixedSize(28, 28)
        dot.setStyleSheet(
            f"background:{BG2}; border:1px solid {self.COLORS[status]}; border-radius:14px;"
        )
        hl.addWidget(dot)

        col = QVBoxLayout()
        col.setSpacing(3)
        col.addWidget(_lbl(label, size=12, color=FG0, bold=True))
        col.addWidget(_lbl(meta, size=10, color=FG2, mono=True))
        hl.addLayout(col, 1)

    def paintEvent(self, event):
        if not self._last:
            p = QPainter(self)
            p.setPen(QPen(QColor(LINE1), 1))
            p.drawLine(13, 28, 13, self.height())
            p.end()


# ── DoAll ─────────────────────────────────────────────────────────────────────

class DoAllDetail(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        hdr = _DetailHeader("Full pipeline",
                             "Runs all four stages in sequence. You'll still need to approve/reject previews in stage 3.",
                             "Start pipeline")
        layout.addWidget(hdr)

        cols = QHBoxLayout()
        cols.setSpacing(20)

        stages_panel = Panel()
        stages_panel.add_header(section_title("Pipeline progress"))
        stages_body = stages_panel.body_layout()
        stages = [
            ("Query source",     "1,842 candidates · queries.csv",         "done"),
            ("Preview download", "1,842 previews · 186 MB",                "done"),
            ("Manual selection", "318 / 1,842 reviewed · 124 accepted",    "running"),
            ("Final download",   "waiting on selection",                   "pending"),
        ]
        for i, (lbl, meta, status) in enumerate(stages):
            stages_body.addWidget(_SpineRow(lbl, meta, status, i == len(stages) - 1))
        cols.addWidget(stages_panel, 1)

        cfg_panel = Panel()
        cfg_panel.setFixedWidth(280)
        cfg_panel.add_header(section_title("Configuration"))
        cfg_body = cfg_panel.body_layout()
        self._query = FileInput(placeholder="~/queries/vocals.csv", directory=False)
        self._folder = FileInput(placeholder="~/datasets/build-01/")
        self._max_cand = _input("2000", width=80)
        cfg_body.addWidget(Field("Query file").add(self._query))
        cfg_body.addWidget(Field("Working folder").add(self._folder))
        cfg_body.addWidget(Field("Max candidates").add(self._max_cand))
        cols.addWidget(cfg_panel)
        layout.addLayout(cols)
        layout.addStretch()


# ── First download ────────────────────────────────────────────────────────────

class FirstDownloadDetail(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        hdr = _DetailHeader("First download", "Bulk-fetch preview files from a query CSV.")
        hdr.run_clicked.connect(self._run)
        layout.addWidget(hdr)

        panel = Panel()
        body = panel.body_layout()
        self._query = FileInput(placeholder="~/queries/vocals.csv", directory=False)
        self._out = FileInput(placeholder="~/datasets/build-01/previews/")
        self._concurrency = _input("8", width=60)
        self._skip = Toggle(on=True)
        body.addWidget(Field("Query CSV", hint="One source ID per row", inline=True).add(self._query))
        body.addWidget(Field("Output folder", inline=True).add(self._out))
        body.addWidget(Field("Concurrency", inline=True).add(self._concurrency))
        body.addWidget(Field("Skip existing", inline=True).add(self._skip))
        layout.addWidget(panel)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()
        self._worker = None

    def _run(self):
        from src.database_creation.first_download_freesound import main as dl_main
        self._prog.start("Downloading previews…")
        # first_download takes a CSV; wrap via subprocess for safety
        self._prog.finish(False, "First download requires a Freesound API key. Set FREESOUND_API_KEY and run from the CLI.")


# ── Final download ────────────────────────────────────────────────────────────

class FinalDownloadDetail(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        hdr = _DetailHeader("Final download", "Fetch full audio for the IDs in a selected CSV.")
        layout.addWidget(hdr)

        panel = Panel()
        body = panel.body_layout()
        self._csv = FileInput(placeholder="~/datasets/build-01/selection.csv", directory=False)
        self._out = FileInput(placeholder="~/datasets/build-01/audio/")
        self._concurrency = _input("4", width=60)
        body.addWidget(Field("Selected IDs CSV", inline=True).add(self._csv))
        body.addWidget(Field("Output folder", inline=True).add(self._out))
        body.addWidget(Field("Concurrency", inline=True).add(self._concurrency))
        layout.addWidget(panel)
        layout.addStretch()


# ── Normalize ─────────────────────────────────────────────────────────────────

class NormalizeDetail(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        hdr = _DetailHeader("Normalize volume", "Target a loudness in dBFS across all files.")
        hdr.run_clicked.connect(self._run)
        layout.addWidget(hdr)

        panel = Panel()
        body = panel.body_layout()
        self._folder = FileInput(placeholder="~/datasets/audio/")
        body.addWidget(Field("Folder", inline=True).add(self._folder))

        # dBFS slider row
        slider_row = QWidget()
        slider_row.setStyleSheet("background:transparent;")
        sl = QHBoxLayout(slider_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(12)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(-40, -6)
        self._slider.setValue(-14)
        self._slider.setFixedWidth(220)
        self._slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ height:4px; background:{BG3}; border-radius:2px; }}"
            f"QSlider::handle:horizontal {{ width:14px; height:14px; background:{ACID}; border-radius:7px; margin:-5px 0; }}"
            f"QSlider::sub-page:horizontal {{ background:{ACID}; border-radius:2px; }}"
        )
        self._dbfs_lbl = _lbl("−14 dBFS", size=13, color=ACID, mono=True)
        self._slider.valueChanged.connect(lambda v: self._dbfs_lbl.setText(f"−{abs(v)} dBFS"))
        sl.addWidget(self._slider)
        sl.addWidget(self._dbfs_lbl)
        sl.addStretch()
        body.addWidget(Field("Target dBFS", hint="Negative value. −14 dBFS is a common streaming target.", inline=True).add(slider_row))
        self._inplace = Toggle(on=False)
        body.addWidget(Field("In-place", hint="Overwrite files. Otherwise writes to ./normalized/", inline=True).add(self._inplace))
        layout.addWidget(panel)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()

    def _run(self):
        folder = self._folder.path
        if not folder:
            return
        from src.database_creation.normalize_volume import normalize_directory
        target = float(self._slider.value())
        self._prog.start("Normalizing…")
        self._worker = DatasetWorker(normalize_directory, Path(folder), target_db=target, in_place=self._inplace.is_on)
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(lambda ok, msg: self._prog.finish(ok, msg))
        self._worker.start()


# ── Merge ─────────────────────────────────────────────────────────────────────

class MergeDetail(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._csv_paths: list[str] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        hdr = _DetailHeader("Merge selected CSVs", "Combine multiple selection CSVs into one.")
        hdr.run_clicked.connect(self._run)
        layout.addWidget(hdr)

        cols = QHBoxLayout()
        cols.setSpacing(20)

        left = Panel()
        add_btn = QPushButton("+ Add")
        add_btn.setFixedHeight(26)
        add_btn.clicked.connect(self._add_csv)
        left.add_header(section_title("Input CSVs"), add_btn)
        self._csv_layout = QVBoxLayout()
        self._csv_layout.setContentsMargins(0, 0, 0, 0)
        csv_container = QWidget()
        csv_container.setStyleSheet("background:transparent;")
        csv_container.setLayout(self._csv_layout)
        left._root.addWidget(csv_container)
        cols.addWidget(left, 1)

        right = Panel()
        right.setFixedWidth(280)
        right.add_header(section_title("Output"))
        right_body = right.body_layout()
        self._out_csv = FileInput(placeholder="~/datasets/merged.csv", directory=False)
        right_body.addWidget(Field("Merged CSV path").add(self._out_csv))
        cols.addWidget(right)
        layout.addLayout(cols)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()

    def _add_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV files (*.csv)")
        if path:
            self._csv_paths.append(path)
            self._refresh_list()

    def _refresh_list(self):
        while self._csv_layout.count():
            item = self._csv_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, path in enumerate(self._csv_paths):
            row = QWidget()
            row.setStyleSheet(f"background:transparent; border-bottom:1px solid {LINE0};")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(14, 10, 14, 10)
            rl.addWidget(_lbl(Path(path).name, size=11, color=FG0, mono=True))
            rl.addStretch()
            rm = QPushButton("✕")
            rm.setProperty("role", "ghost")
            rm.setFixedSize(24, 24)
            idx = i
            rm.clicked.connect(lambda _, j=idx: self._remove(j))
            rl.addWidget(rm)
            self._csv_layout.addWidget(row)

    def _remove(self, idx: int):
        if 0 <= idx < len(self._csv_paths):
            self._csv_paths.pop(idx)
            self._refresh_list()

    def _run(self):
        if not self._csv_paths:
            return
        out = self._out_csv.path
        if not out:
            return
        from src.database_creation.merge_selected_csv import merge_selected_csvs
        input_dir = Path(self._csv_paths[0]).parent
        self._prog.start("Merging CSVs…")
        self._worker = DatasetWorker(merge_selected_csvs, input_dir, Path(out))
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(lambda ok, msg: self._prog.finish(ok, msg))
        self._worker.start()


# ── Convert ───────────────────────────────────────────────────────────────────

class ConvertDetail(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        hdr = _DetailHeader("Convert format / sample rate", "Batch-convert audio to the format RAVE expects.")
        hdr.run_clicked.connect(self._run)
        layout.addWidget(hdr)

        panel = Panel()
        body = panel.body_layout()
        self._src = FileInput(placeholder="~/datasets/raw/")
        body.addWidget(Field("Source folder", inline=True).add(self._src))
        self._sr = RadioGroup(
            options=[{"value": "22050", "label": "22,050"}, {"value": "44100", "label": "44,100"},
                     {"value": "48000", "label": "48,000"}, {"value": "96000", "label": "96,000"}],
            value="44100",
        )
        body.addWidget(Field("Target sample rate", inline=True).add(self._sr))
        self._ch = RadioGroup(
            options=[{"value": "1", "label": "Mono"}, {"value": "2", "label": "Stereo"}],
            value="1",
        )
        body.addWidget(Field("Channels", inline=True).add(self._ch))
        self._subtype = QComboBox()
        self._subtype.addItems(["PCM_16 (16-bit signed)", "PCM_24 (24-bit signed)", "PCM_32 (32-bit signed)", "FLOAT (32-bit float)"])
        body.addWidget(Field("WAV subtype", inline=True).add(self._subtype))
        self._out = FileInput(placeholder="~/datasets/raw-44k-mono/")
        body.addWidget(Field("Output", inline=True).add(self._out))
        layout.addWidget(panel)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()

    def _run(self):
        src = self._src.path
        out = self._out.path
        if not src:
            return
        from src.database_creation.convert_format import convert_directory
        sr = int(self._sr.value)
        ch = int(self._ch.value)
        subtype = self._subtype.currentText().split()[0]
        self._prog.start("Converting…")
        self._worker = DatasetWorker(convert_directory, Path(src), Path(out) if out else None,
                                     target_sr=sr, target_channels=ch, target_subtype=subtype)
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(lambda ok, msg: self._prog.finish(ok, msg))
        self._worker.start()
