"""Dataset wizard detail panels: DoAll orchestrator + shared helpers."""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QScrollArea, QFrame,
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    Panel, Field, FileInput, Toggle, section_title, _lbl,
    BG0, BG1, BG2, FG0, FG2, FG3, LINE0, LINE1, ACID, AMBER, MONO,
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

