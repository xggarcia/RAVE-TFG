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
    previewReady = Signal(str, str)   # (previews_folder, save_csv_path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._stage = 0  # 0=idle, 1=preview_dl, 2=awaiting_review, 3=final_dl, 4=done
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        hdr = _DetailHeader(
            "Full pipeline",
            "Stage 1: download previews. Stage 2: review in the Preview step. "
            "Stage 3: download final audio, normalize and convert.",
            "Start pipeline",
        )
        hdr.run_clicked.connect(self._run)
        layout.addWidget(hdr)

        cols = QHBoxLayout()
        cols.setSpacing(20)

        stages_panel = Panel()
        stages_panel.add_header(section_title("Pipeline stages"))
        stages_body = stages_panel.body_layout()
        self._stage_rows_do: list[_SpineRow] = []
        stages_init = [
            ("1 · Preview download",  "fetch Freesound preview candidates",    "pending"),
            ("2 · Manual selection",  "review in the Preview step on the left", "pending"),
            ("3 · Final download",    "download selected audio",                "pending"),
            ("4 · Normalize",         "target RMS loudness",                    "pending"),
            ("5 · Convert",           "WAV mono 44 100 Hz PCM_16",              "pending"),
        ]
        for i, (lbl, meta, status) in enumerate(stages_init):
            row = _SpineRow(lbl, meta, status, i == len(stages_init) - 1)
            self._stage_rows_do.append(row)
            stages_body.addWidget(row)
        cols.addWidget(stages_panel, 1)

        cfg_panel = Panel()
        cfg_panel.setFixedWidth(280)
        cfg_panel.add_header(section_title("Configuration"))
        cfg_body = cfg_panel.body_layout()
        self._query = FileInput(placeholder="~/queries/vocals.csv", directory=False)
        self._folder = FileInput(placeholder="~/datasets/build-01/")
        self._selection_csv_name = _input("selection", width=220)
        self._api_key_do = _input("", width=220)
        cfg_body.addWidget(Field("Query file").add(self._query))
        cfg_body.addWidget(Field("Working folder").add(self._folder))
        cfg_body.addWidget(Field(
            "Selection CSV name",
            hint="Saved to / read from the working folder as <name>.csv.",
        ).add(self._selection_csv_name))
        cfg_body.addWidget(Field(
            "API key",
            hint="Reads FREESOUND_API_KEY from .env if blank.",
        ).add(self._api_key_do))
        cols.addWidget(cfg_panel)
        layout.addLayout(cols)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()

    def _run(self):
        import os
        folder = self._folder.path
        query  = self._query.path
        sel_name = self._selection_csv_name.text().strip() or "selection"

        if not folder:
            self._prog.start("Missing working folder")
            self._prog.finish(False, "Set a working folder in the configuration panel.")
            return

        sel_csv = Path(folder) / f"{sel_name}.csv"
        api_key = self._api_key_do.text().strip() or os.getenv("FREESOUND_API_KEY", "").strip()

        if query and api_key and not sel_csv.exists():
            # Stage 1: download previews into {folder}/previews/
            from src.database_creation.first_download_freesound import (
                read_jobs_from_csv, download_sounds_freesound,
            )
            _query = query
            _folder = folder
            _key = api_key

            def _run_stage1():
                jobs = read_jobs_from_csv(Path(_query))
                if not jobs:
                    print("No download jobs found in the query CSV.")
                    return 0, 0
                previews_dir = Path(_folder) / "previews"
                previews_dir.mkdir(parents=True, exist_ok=True)
                total = 0
                for job in jobs:
                    job.output_dir = previews_dir
                    if _key:
                        job.api_key = _key
                    total += download_sounds_freesound(job)
                return total, len(jobs)

            _previews_dir = str(Path(folder) / "previews")
            _sel_csv = str(sel_csv)

            def _on_stage1_done(ok: bool, msg: str):
                if ok:
                    self._prog.info(
                        "Stage 1 done — switching to Preview step…",
                        f"{msg}\n\nAccept/reject files in the Preview step, then press Save selection.\n"
                        f"Selection will be saved to:  {_sel_csv}",
                    )
                    self.previewReady.emit(_previews_dir, _sel_csv)
                else:
                    self._prog.finish(False, msg)

            self._prog.start("Stage 1 — downloading previews…")
            self._worker = DatasetWorker(_run_stage1)
            self._worker.log.connect(self._prog.append_log)
            self._worker.finished.connect(_on_stage1_done)
            self._worker.start()
            return

        if sel_csv.exists() and api_key:
            # Stages 3-5: final download → normalize → convert
            from src.database_creation.download_csv import download_from_csv
            from src.database_creation.normalize_volume import normalize_directory
            from src.database_creation.convert_format import convert_directory

            final_dir = Path(folder) / "audio"
            _csv = sel_csv
            _key = api_key

            def _run_stages():
                ok, total = download_from_csv(_csv, final_dir, _key, skip_existing=True)
                print(f"\nDownloaded {ok}/{total} files.")
                ok2, n2 = normalize_directory(final_dir)
                print(f"Normalized {ok2}/{n2} files.")
                ok3, n3 = convert_directory(final_dir)
                print(f"Converted {ok3}/{n3} files.")
                return ok3, n3

            self._prog.start("Stages 3-5: downloading, normalizing, converting…")
            self._worker = DatasetWorker(_run_stages)
            self._worker.log.connect(self._prog.append_log)
            self._worker.finished.connect(lambda ok, msg: self._prog.finish(ok, msg))
            self._worker.start()
            return

        # Guidance when required fields are missing
        missing = []
        if not sel_csv.exists():
            missing.append(f"Selection CSV not found at: {sel_csv}  (run Preview step first)")
        if not api_key:
            missing.append("Freesound API key (set FREESOUND_API_KEY in .env or enter above)")
        self._prog.start("Missing required fields")
        self._prog.finish(False, "Please fill in:\n• " + "\n• ".join(missing))


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
        hdr.run_clicked.connect(self._run)
        layout.addWidget(hdr)

        panel = Panel()
        body = panel.body_layout()
        self._csv = FileInput(placeholder="~/datasets/build-01/selection.csv", directory=False)
        self._out = FileInput(placeholder="~/datasets/build-01/audio/")
        body.addWidget(Field("Selected IDs CSV", inline=True).add(self._csv))
        body.addWidget(Field("Output folder", inline=True).add(self._out))
        self._api_key_input = _input("", width=260)
        body.addWidget(Field(
            "API key",
            hint="Reads FREESOUND_API_KEY from .env if left blank.",
            inline=True,
        ).add(self._api_key_input))
        self._skip = Toggle(on=True)
        body.addWidget(Field("Skip existing", inline=True).add(self._skip))
        layout.addWidget(panel)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()
        self._worker = None

    def _run(self):
        import os
        csv_path = self._csv.path
        out_path = self._out.path
        if not csv_path or not out_path:
            return
        api_key = (self._api_key_input.text() if self._api_key_input else "").strip()
        if not api_key:
            api_key = os.getenv("FREESOUND_API_KEY", "").strip()
        if not api_key:
            self._prog.start("Missing API key")
            self._prog.finish(False, "Set FREESOUND_API_KEY in .env or enter it above.")
            return
        from src.database_creation.download_csv import download_from_csv
        self._prog.start("Downloading audio…")
        self._worker = DatasetWorker(
            download_from_csv, Path(csv_path), Path(out_path), api_key,
            self._skip.is_on,
        )
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(lambda ok, msg: self._prog.finish(ok, msg))
        self._worker.start()


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
        body.addWidget(Field(
            "Note",
            hint="Files are normalized in-place and saved as 16-bit WAV.",
            inline=True,
        ).add(_lbl("Always in-place", size=11, color=FG3)))
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
        self._worker = DatasetWorker(normalize_directory, Path(folder), target_db=target)
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
        body.addWidget(Field(
            "Note",
            hint="Files are converted in-place (original non-WAV files are removed).",
            inline=True,
        ).add(_lbl("Always in-place", size=11, color=FG3)))
        layout.addWidget(panel)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()

    def _run(self):
        src = self._src.path
        if not src:
            return
        from src.database_creation.convert_format import convert_directory
        sr = int(self._sr.value)
        ch = int(self._ch.value)
        subtype = self._subtype.currentText().split()[0]
        self._prog.start("Converting…")
        self._worker = DatasetWorker(convert_directory, Path(src),
                                     target_sr=sr, target_channels=ch, target_subtype=subtype)
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(lambda ok, msg: self._prog.finish(ok, msg))
        self._worker.start()
