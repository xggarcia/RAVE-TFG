"""Train model page — form, resume banner, live training panel."""
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QPushButton, QLineEdit, QComboBox, QSizePolicy, QStackedWidget,
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    PageHeader, Panel, Field, FileInput, RadioGroup, section_title, _lbl,
    BG0, BG1, BG2, BG3, FG0, FG1, FG2, FG3, LINE0, LINE1, ACID, AMBER, MAG, MONO,
)
from app.ui.pages._train_live import LivePanel
from app.workers.train_worker import TrainWorker
from app.ui.pages.workflow import WorkflowPage
from app.ui.pages.preprocess import PreprocessPage
from app.ui.pages.export import ExportPage

CONFIGS = ["v2_small", "v2", "v3", "v3_small"]
from app.ui.pages.workflow import _ConfigChip, EXTRA_CONFIGS, DEFAULT_ON  # noqa: E402


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


PIPELINE_STEPS = [
    ("do_all", "DO ALL", "preprocess → train → export"),
    ("preprocess", "Preprocess", "dataset preparation only"),
    ("train_only", "Train", "model optimization only"),
    ("export", "Export", "torchscript export only"),
]


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

    # ── Form panels ───────────────────────────────────────────────────────────

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

        self._val_edit     = _input("10000")
        self._save_edit    = _input("25000")
        self._steps_edit   = _input("500000")
        self._batch_edit   = _input("8")
        left_col.addWidget(Field("Val every").add(self._val_edit))
        right_col.addWidget(Field("Save every").add(self._save_edit))
        left_col.addWidget(Field("Max steps").add(self._steps_edit))
        right_col.addWidget(Field("Batch size").add(self._batch_edit))

        gl.addLayout(left_col)
        gl.addLayout(right_col)
        body.addWidget(grid)

        # Extra configs row
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

    # ── Resume detection ──────────────────────────────────────────────────────

    def _check_resume(self):
        name = self._name_edit.text().strip()
        if not name:
            self._banner.hide()
            return
        try:
            sys.path.insert(0, ".")
            from src.train import find_latest_run
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

    # ── Start / stop ──────────────────────────────────────────────────────────

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

        # Build command (mirrors src/train.py:TrainModel)
        from src.train import _detect_gpu_flag
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


class TrainPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._step_list = _TrainStepList()
        self._step_list.navigate.connect(self._navigate)
        body.addWidget(self._step_list)

        self._stack = QStackedWidget()
        self._detail_widgets: dict[str, QWidget] = {
            "do_all": WorkflowPage(),
            "preprocess": PreprocessPage(),
            "train_only": _TrainOnlyPage(),
            "export": ExportPage(),
        }
        for sid, widget in self._detail_widgets.items():
            self._stack.addWidget(widget)

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
