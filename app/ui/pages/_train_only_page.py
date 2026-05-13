"""Train-only sub-page: configuration form + live training panel."""
import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.form import (
    AMBER,
    BG0,
    BG1,
    FG0,
    FG2,
    FG3,
    LINE0,
    LINE1,
    MONO,
    Field,
    FileInput,
    PageHeader,
    Panel,
    RadioGroup,
    _lbl,
    section_title,
)
from app.ui.pages._train_form_widgets import CONFIGS, _ResumeBanner
from app.ui.pages._train_live import LivePanel
from app.ui.pages._workflow_config import DEFAULT_ON, EXTRA_CONFIGS, _ConfigChip
from app.workers.train_worker import TrainWorker


class _TrainOnlyPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: TrainWorker | None = None
        self._resume_path: str | None = None
        self._use_resume = False
        self._chips: list[_ConfigChip] = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._reset_btn = QPushButton("Reset form")
        self._reset_btn.setFixedHeight(30)
        self._reset_btn.clicked.connect(self._reset_form)

        self._start_btn = QPushButton("▶  Start training")
        self._start_btn.setProperty("role", "primary")
        self._start_btn.setFixedHeight(30)
        self._start_btn.clicked.connect(self._start_or_stop)

        root.addWidget(PageHeader(
            crumbs=["Data & Training", "Train Model"],
            title="Train model",
            desc="Train a RAVE model from a preprocessed dataset. Resume detection picks up the latest checkpoint.",
            actions=[self._reset_btn, self._start_btn],
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(16)

        self._banner = _ResumeBanner()
        self._banner.resume_btn.clicked.connect(lambda: self._set_resume(True))
        self._banner.fresh_btn.clicked.connect(lambda: self._set_resume(False))
        cl.addWidget(self._banner)

        cols = QHBoxLayout()
        cols.setSpacing(20)
        cols.addWidget(self._build_params(), 1)
        cols.addWidget(self._build_estimate(), 0)
        cl.addLayout(cols)

        self._live = LivePanel()
        cl.addWidget(self._live)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_params(self) -> QWidget:
        panel = Panel()
        panel.add_header(section_title("Model configuration"))
        body = panel.body_layout()

        def _input(default: str) -> QLineEdit:
            w = QLineEdit(default)
            w.setStyleSheet(
                f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
                f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
            )
            return w

        grid = QWidget()
        grid.setStyleSheet("background:transparent;")
        gl = QHBoxLayout(grid)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(14)

        left_col = QVBoxLayout()
        left_col.setSpacing(14)
        right_col = QVBoxLayout()
        right_col.setSpacing(14)

        self._name_edit = _input("my_model")
        self._name_edit.textChanged.connect(self._check_resume)
        left_col.addWidget(Field("Model name").add(self._name_edit))

        self._config_box = QComboBox()
        self._config_box.addItems(CONFIGS)
        right_col.addWidget(Field("Config").add(self._config_box))

        self._db_input = FileInput(placeholder="~/preprocessed/data.lmdb", directory=False)
        left_col.addWidget(Field("Dataset path").add(self._db_input))

        self._channels = RadioGroup(
            options=[{"value": "1", "label": "1"}, {"value": "2", "label": "2"}],
            value="1",
        )
        right_col.addWidget(Field("Channels").add(self._channels))

        self._val_edit   = _input("10000")
        self._save_edit  = _input("25000")
        self._steps_edit = _input("500000")
        self._batch_edit = _input("8")
        left_col.addWidget(Field("Val every").add(self._val_edit))
        right_col.addWidget(Field("Save every").add(self._save_edit))
        left_col.addWidget(Field("Max steps").add(self._steps_edit))
        right_col.addWidget(Field("Batch size").add(self._batch_edit))

        gl.addLayout(left_col)
        gl.addLayout(right_col)
        body.addWidget(grid)

        extra_lbl = _lbl("Extra configs", size=10, color=FG2, mono=True, spacing="1px")
        body.addWidget(extra_lbl)
        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)
        for key, desc in EXTRA_CONFIGS:
            chip = _ConfigChip(key, desc, on=(key in DEFAULT_ON))
            self._chips.append(chip)
            chips_row.addWidget(chip)
        chips_row.addStretch()
        body.addLayout(chips_row)

        body.addStretch()
        return panel

    def _build_estimate(self) -> QWidget:
        panel = Panel()
        panel.setFixedWidth(280)
        panel.add_header(section_title("Estimate"))
        body = panel.body_layout()

        body.addWidget(_lbl("~18h 32m", size=22, color=AMBER, bold=True, mono=True))
        body.addWidget(_lbl("On RTX 4090 @ batch=8. Scales ~inversely with batch size.", size=11, color=FG2, wrap=True))

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{LINE0};")
        body.addWidget(sep)

        for k, v in [("VRAM estimate", "~11.8 GB"), ("Checkpoint size", "~520 MB × 20"), ("Disk needed", "~10.4 GB")]:
            row = QHBoxLayout()
            row.addWidget(_lbl(k, size=10, color=FG3, mono=True, spacing="1px"))
            row.addStretch()
            row.addWidget(_lbl(v, size=11, color=FG0, mono=True))
            body.addLayout(row)

        body.addStretch()
        return panel

    def _check_resume(self):
        name = self._name_edit.text().strip()
        if not name:
            self._banner.hide()
            return
        try:
            sys.path.insert(0, ".")
            from src.core.train import find_latest_run
            path = find_latest_run("models/user_model/checkpoints", name)
            if path:
                self._resume_path = path
                self._banner.show_checkpoint(path)
            else:
                self._resume_path = None
                self._banner.hide()
        except Exception:
            self._banner.hide()

    def _set_resume(self, use: bool):
        self._use_resume = use
        self._banner.hide()

    def _start_or_stop(self):
        if self._worker:
            self._worker.stop()
            self._start_btn.setText("▶  Start training")
            self._start_btn.setProperty("role", "primary")
            self._start_btn.style().unpolish(self._start_btn)
            self._start_btn.style().polish(self._start_btn)
            self._worker = None
            return

        try:
            max_steps  = int(self._steps_edit.text())
            batch_size = int(self._batch_edit.text())
            val_every  = int(self._val_edit.text())
            save_every = int(self._save_edit.text())
            channels   = int(self._channels.value)
        except ValueError:
            return

        extra = [c.key for c in self._chips if c.is_on]
        ckpt  = (self._resume_path if self._use_resume else None)

        from src.core.train import _detect_gpu_flag
        cmd = [
            "rave", "train",
            "--config", self._config_box.currentText(),
            *[arg for c in extra for arg in ("--config", c)],
            "--db_path", self._db_input.path or "preprocessed_data",
            "--out_path", "models/user_model/checkpoints",
            "--name",    self._name_edit.text() or "my_model",
            "--channels", str(channels),
            "--val_every", str(val_every),
            "--save_every", str(save_every),
            "--max_steps",  str(max_steps),
            "--batch", str(batch_size),
        ]
        for gpu_id in _detect_gpu_flag():
            cmd.extend(["--gpu", gpu_id])
        if ckpt:
            cmd.extend(["--ckpt", ckpt])

        self._live.reset(max_steps)
        self._live.show()

        self._worker = TrainWorker(cmd, max_steps)
        self._worker.progress.connect(self._live.update_progress)
        self._worker.log.connect(self._live.append_log)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

        self._start_btn.setText("■  Stop")
        self._start_btn.setProperty("role", "danger")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)

    def _reset_form(self):
        self._name_edit.setText("my_model")
        self._config_box.setCurrentIndex(0)
        self._db_input.set_path("")
        self._val_edit.setText("10000")
        self._save_edit.setText("25000")
        self._steps_edit.setText("500000")
        self._batch_edit.setText("8")
        for chip in self._chips:
            chip._on = chip.key in DEFAULT_ON
            chip.update()
        self._banner.hide()
        self._live.hide()

    @Slot(str, str)
    def _on_failed(self, short: str, traceback: str):
        self._live.show_failure(short, traceback)
        self._reset_start_btn()

    @Slot(dict)
    def _on_finished(self, summary: dict):
        self._live.append_log("INFO", f"Training finished. Steps: {summary.get('steps_trained', '?')}")
        self._reset_start_btn()

    def _reset_start_btn(self):
        self._worker = None
        self._start_btn.setText("▶  Start training")
        self._start_btn.setProperty("role", "primary")
        self._start_btn.style().unpolish(self._start_btn)
        self._start_btn.style().polish(self._start_btn)
