"""Dataset wizard detail panels: DoAll orchestrator + shared helpers."""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QInputDialog, QMessageBox,
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    Panel, Field, FileInput, section_title, _lbl, BG1,
    BG2, FG0, FG2, FG3, LINE0, LINE1, ACID, AMBER, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.dataset_worker import DatasetWorker


def ensure_freesound_oauth(parent: QWidget, client_id: str, client_secret: str) -> bool:
    """Guarantee a valid Freesound OAuth token is cached on disk.

    Returns True if a valid token is available (cached, refreshed, or freshly
    obtained via a browser + QInputDialog flow). Returns False on cancel or
    failure (a QMessageBox is shown on failure).
    """
    from src.database import freesound_auth as fa

    if not client_id or not client_secret:
        QMessageBox.warning(
            parent, "Missing credentials",
            "Both API key (client_id) and Client secret are required to download originals.",
        )
        return False

    def _code_provider(auth_url: str) -> str:
        code, ok = QInputDialog.getText(
            parent,
            "Freesound authorization",
            "A browser tab has been opened. Authorize the app on Freesound\n"
            f"and paste the authorization code below.\n\nURL: {auth_url}",
        )
        return code.strip() if ok else ""

    try:
        token = fa.get_access_token(client_id, client_secret, code_provider=_code_provider)
        return bool(token)
    except Exception as exc:
        QMessageBox.critical(parent, "Authorization failed", str(exc))
        return False


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
        self._client_id_do = _input("", width=220)
        self._client_secret_do = _input("", width=220)
        self._client_secret_do.setEchoMode(QLineEdit.Password)
        cfg_body.addWidget(Field("Query file").add(self._query))
        cfg_body.addWidget(Field("Working folder").add(self._folder))
        cfg_body.addWidget(Field(
            "Selection CSV name",
            hint="Saved to / read from the working folder as <name>.csv.",
        ).add(self._selection_csv_name))
        cfg_body.addWidget(Field(
            "API key",
            hint="Token for read-only endpoints (search, sound info).",
        ).add(self._api_key_do))
        cfg_body.addWidget(Field(
            "Client ID",
            hint="OAuth2 client_id. Leave blank to reuse the API key.",
        ).add(self._client_id_do))
        cfg_body.addWidget(Field(
            "Client secret",
            hint="Required to download originals (OAuth2). \nWithout it, only previews are fetched.",
        ).add(self._client_secret_do))
        cols.addWidget(cfg_panel)
        layout.addLayout(cols)

        self._prog = ProgressPanel()
        layout.addWidget(self._prog)
        layout.addStretch()

    def _run(self):
        folder = self._folder.path
        query  = self._query.path
        sel_name = self._selection_csv_name.text().strip() or "selection"

        if not folder:
            self._prog.start("Missing working folder")
            self._prog.finish(False, "Set a working folder in the configuration panel.")
            return

        sel_csv = Path(folder) / f"{sel_name}.csv"
        api_key = self._api_key_do.text().strip()
        client_id = self._client_id_do.text().strip() or api_key
        client_secret = self._client_secret_do.text().strip()

        if query and api_key and not sel_csv.exists():
            # Stage 1: download previews into {folder}/previews/
            from src.database.first_download_freesound import (
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
            if not client_secret:
                self._prog.start("Missing client secret")
                self._prog.finish(False, "Set Client secret to download originals (OAuth2 is required).")
                return
            if not ensure_freesound_oauth(self, client_id, client_secret):
                self._prog.start("Authorization required")
                self._prog.finish(False, "Freesound authorization was cancelled or failed.")
                return

            from src.database.download_csv import download_from_csv
            from src.database.normalize_volume import normalize_directory
            from src.database.convert_format import convert_directory

            final_dir = Path(folder) / "audio"
            _csv = sel_csv
            _key = api_key
            _cid = client_id
            _secret = client_secret

            def _run_stages():
                ok, total = download_from_csv(
                    _csv, final_dir, _key,
                    skip_existing=True, client_secret=_secret, client_id=_cid,
                )
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
            missing.append("Freesound API key (enter it above)")
        self._prog.start("Missing required fields")
        self._prog.finish(False, "Please fill in:\n• " + "\n• ".join(missing))

