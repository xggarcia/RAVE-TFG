from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

EXTRA_CONFIGS = [
    ("noise",    "noise injection"),
    ("causal",   "causal convolutions"),
    ("snake",    "snake activations"),
    ("hinge",    "hinge discriminator"),
    ("descript", "descript discriminator"),
]
DEFAULT_ON = {"noise", "causal"}

from app.ui.pages._train_live import LivePanel
from app.ui.widgets.form import (
    ACID,
    AMBER,
    BG0,
    BG1,
    BG2,
    FG0,
    FG1,
    FG2,
    FG3,
    LINE0,
    LINE1,
    MAG,
    MONO,
    Field,
    FileInput,
    PageHeader,
    Panel,
    RadioGroup,
    Toggle,
    _lbl,
    section_title,
)
from app.ui.pages._workflow_widgets import StageRow, StageState


class _ConfigChip(QWidget):
    def __init__(self, key: str, desc: str, on: bool = False, parent=None):
        super().__init__(parent)
        self.key = key
        self._on = on
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(36)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumWidth(80)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(6)
        self._key_lbl = _lbl(key, size=11, color=ACID if on else FG1, mono=True)
        self._desc_lbl = _lbl(f"· {desc}", size=10, color=FG3)
        layout.addWidget(self._key_lbl)
        layout.addWidget(self._desc_lbl)

    @property
    def is_on(self) -> bool:
        return self._on

    def mousePressEvent(self, event):
        self._on = not self._on
        self._key_lbl.setStyleSheet(
            f"color:{ACID if self._on else FG1}; font-size:11px; {MONO} background:transparent;"
        )
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        if self._on:
            p.setPen(QPen(QColor("#3a4f1a"), 1))
            p.setBrush(QColor("#1e2d0a"))
        else:
            p.setPen(QPen(QColor(LINE1), 1))
            p.setBrush(QColor(BG1))
        p.drawRoundedRect(r, 3, 3)
        p.end()
from app.workers.export_worker import ExportWorker
from app.workers.preprocess_worker import PreprocessWorker
from app.workers.train_worker import TrainWorker


class WorkflowPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pre: PreprocessWorker | None = None
        self._train: TrainWorker | None = None
        self._exp: ExportWorker | None = None
        self._stop_requested = False
        self._active_stage = "preprocess"
        self._states = {
            "preprocess": StageState(meta="Waiting to start"),
            "train": StageState(meta="Will start after preprocess"),
            "export": StageState(meta="Will start after training"),
        }
        self._build()
        self._refresh_timeline()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setProperty("role", "primary")
        self._start_btn.setFixedHeight(30)
        self._start_btn.clicked.connect(self._start_workflow)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedHeight(30)
        self._pause_btn.setEnabled(False)
        self._pause_btn.setToolTip("Pause is not available for RAVE subprocesses.")

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("role", "danger")
        self._stop_btn.setFixedHeight(30)
        self._stop_btn.clicked.connect(self._stop_workflow)
        self._stop_btn.setEnabled(False)

        root.addWidget(
            PageHeader(
                crumbs=["Data & Training", "Full Workflow"],
                title="Full workflow",
                desc="Preprocess, train, and export in one unattended run.",
                actions=[self._pause_btn, self._stop_btn, self._start_btn],
            )
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(20)

        cl.addWidget(self._build_config())
        cl.addWidget(self._build_timeline())
        cl.addWidget(self._build_detail())
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_config(self) -> QWidget:
        self._chips: list[_ConfigChip] = []

        panel = Panel()
        panel.add_header(section_title("Run configuration"), _lbl("active", 10, ACID, mono=True))
        body = panel.body_layout()

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self._audio_input = FileInput(placeholder="~/datasets/vocals-v2/", directory=True)
        self._model_name = QLineEdit("vox-phase3-v2")
        self._model_name.setStyleSheet(
            f"background:{BG1}; color:{FG0}; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px; {MONO}"
        )
        self._config = QComboBox()
        self._config.addItems(["v2_small", "v2", "v3", "v3_small"])
        self._config.setCurrentText("v2")
        self._channels = RadioGroup(
            options=[{"value": "1", "label": "Mono (1)"}, {"value": "2", "label": "Stereo (2)"}],
            value="1",
        )
        self._max_steps = QLineEdit("500000")
        self._max_steps.setStyleSheet(
            f"background:{BG1}; color:{FG0}; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px; {MONO}"
        )
        self._batch_size = QLineEdit("8")
        self._batch_size.setStyleSheet(
            f"background:{BG1}; color:{FG0}; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px; {MONO}"
        )
        self._destination = FileInput(placeholder="models/user_model/exported_model", directory=True)
        self._val_every = QLineEdit("10000")
        self._val_every.setStyleSheet(
            f"background:{BG1}; color:{FG0}; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px; {MONO}"
        )
        self._save_every = QLineEdit("25000")
        self._save_every.setStyleSheet(
            f"background:{BG1}; color:{FG0}; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px; {MONO}"
        )
        self._export_run = FileInput(placeholder="auto-detect from training", directory=True)
        self._streaming = Toggle(on=True)
        self._export_name = QLineEdit("model_streaming.ts")
        self._export_name.setStyleSheet(
            f"background:{BG1}; color:{FG0}; border:1px solid {LINE1}; border-radius:4px; padding:6px 10px; {MONO}"
        )

        widgets = [
            Field("Audio folder", inline=False).add(self._audio_input),
            Field("Model name", inline=False).add(self._model_name),
            Field("Config", inline=False).add(self._config),
            Field("Channels", inline=False).add(self._channels),
            Field("Max steps", inline=False).add(self._max_steps),
            Field("Batch size", inline=False).add(self._batch_size),
            Field("Val every", inline=False).add(self._val_every),
            Field("Save every", inline=False).add(self._save_every),
            Field("Export destination", inline=False).add(self._destination),
            Field("Export run folder", hint="Leave blank to auto-detect after training.", inline=False).add(self._export_run),
            Field("Streaming", hint="Required for real-time use.", inline=False).add(self._streaming),
            Field("Output name", inline=False).add(self._export_name),
        ]
        for i, widget in enumerate(widgets):
            grid.addWidget(widget, i // 4, i % 4)

        body.addLayout(grid)

        # Extra configs chips
        body.addWidget(_lbl("Extra configs", size=10, color=FG2, mono=True, spacing="1px"))
        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)
        for key, desc in EXTRA_CONFIGS:
            chip = _ConfigChip(key, desc, on=(key in DEFAULT_ON))
            self._chips.append(chip)
            chips_row.addWidget(chip)
        chips_row.addStretch()
        body.addLayout(chips_row)

        return panel

    def _build_timeline(self) -> QWidget:
        panel = Panel()
        self._timeline_right = _lbl("0 of 3", 10, FG2, mono=True)
        panel.add_header(section_title("Stages"), self._timeline_right)
        body = panel.body_layout()
        body.setSpacing(0)

        self._timeline_rows: dict[str, QWidget] = {}
        for stage_id, title in [("preprocess", "Preprocess"), ("train", "Train"), ("export", "Export")]:
            row = StageRow(title)
            self._timeline_rows[stage_id] = row
            body.addWidget(row)
        return panel

    def _build_detail(self) -> QWidget:
        shell = QWidget()
        lay = QHBoxLayout(shell)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)

        self._live = LivePanel()
        lay.addWidget(self._live, 1)

        side = Panel()
        side.setFixedWidth(360)
        self._detail_title = _lbl("Workflow log", 10, FG2, mono=True)
        side.add_header(self._detail_title)
        self._detail_log = QTextEdit()
        self._detail_log.setReadOnly(True)
        self._detail_log.setMinimumHeight(320)
        self._detail_log.setStyleSheet(
            f"QTextEdit {{ background:{BG0}; color:{FG1}; border:none; padding:10px; {MONO} font-size:10.5px; }}"
        )
        side._root.addWidget(self._detail_log)
        lay.addWidget(side)

        return shell

    def _log(self, level: str, message: str):
        color = ACID if level == "INFO" else AMBER if level == "WARN" else MAG
        self._detail_log.append(f"<span style='color:{color}'>[{level}]</span> <span style='color:{FG1}'>{message}</span>")

    def _refresh_timeline(self):
        done = sum(1 for s in self._states.values() if s.status == "done")
        self._timeline_right.setText(f"{done} of 3")
        for key, row in self._timeline_rows.items():
            row.update_state(self._states[key])

    def _set_state(self, stage: str, status: str, meta: str = "", progress: float | None = None):
        st = self._states[stage]
        st.status = status
        st.meta = meta
        st.progress = progress
        self._active_stage = stage
        self._detail_title.setText(f"Active stage: {stage}")
        self._refresh_timeline()

    def _workflow_values(self) -> dict | None:
        audio = self._audio_input.path
        if not audio:
            self._log("ERR", "Select an audio folder before starting the workflow.")
            return None
        try:
            max_steps  = int(self._max_steps.text())
            batch      = int(self._batch_size.text())
            val_every  = int(self._val_every.text())
            save_every = int(self._save_every.text())
        except ValueError:
            self._log("ERR", "Max steps, batch size, val every and save every must be integers.")
            return None
        name = self._model_name.text().strip() or "my_model"
        return {
            "audio": audio,
            "name": name,
            "config": self._config.currentText(),
            "channels": 1 if self._channels.value.startswith("Mono") else 2,
            "max_steps": max_steps,
            "batch": batch,
            "val_every": val_every,
            "save_every": save_every,
            "extra_configs": [c.key for c in self._chips if c.is_on],
            "export_dest": self._destination.path or "models/user_model/exported_model",
            "export_run": self._export_run.path or "",
            "streaming": self._streaming.is_on,
            "export_name": self._export_name.text().strip() or f"{name}_streaming.ts",
        }

    def _start_workflow(self):
        if self._pre or self._train or self._exp:
            return
        values = self._workflow_values()
        if not values:
            return

        self._values = values
        self._stop_requested = False
        self._detail_log.clear()
        self._live.reset(values["max_steps"])
        self._live.hide()

        for stage in self._states:
            self._states[stage] = StageState(status="pending", meta="Pending")
        self._set_state("preprocess", "running", "Preparing dataset", 0.0)
        self._set_state("train", "pending", "Waiting for preprocess")
        self._set_state("export", "pending", "Waiting for training")

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._run_preprocess()

    def _stop_workflow(self):
        self._stop_requested = True
        self._log("WARN", "Stop requested. Active stage will halt when possible.")
        if self._train:
            self._train.stop()

    def _run_preprocess(self):
        self._pre = PreprocessWorker(
            audio_path=self._values["audio"],
            channels=self._values["channels"],
            lazy=False,
            max_db_size=8,
        )
        self._pre.log.connect(lambda m: self._log("INFO", m))
        self._pre.finished.connect(self._on_preprocess_finished)
        self._pre.start()

    @Slot(bool, str)
    def _on_preprocess_finished(self, success: bool, message: str):
        self._pre = None
        if not success:
            self._set_state("preprocess", "failed", message)
            self._finish_workflow()
            return
        self._set_state("preprocess", "done", message, 1.0)
        if self._stop_requested:
            self._set_state("train", "stopped", "Stopped before training")
            self._set_state("export", "stopped", "Stopped before export")
            self._finish_workflow()
            return
        self._set_state("train", "running", "Starting training", 0.0)
        self._run_train()

    def _run_train(self):
        from src.train import _detect_gpu_flag

        extra = self._values["extra_configs"]
        cmd = [
            "rave", "train",
            "--config", self._values["config"],
            *[arg for c in extra for arg in ("--config", c)],
            "--db_path", "preprocessed_data",
            "--out_path", "models/user_model/checkpoints",
            "--name", self._values["name"],
            "--channels", str(self._values["channels"]),
            "--val_every", str(self._values["val_every"]),
            "--save_every", str(self._values["save_every"]),
            "--max_steps", str(self._values["max_steps"]),
            "--batch", str(self._values["batch"]),
        ]
        for gpu_id in _detect_gpu_flag():
            cmd.extend(["--gpu", gpu_id])

        self._live.show()
        self._train = TrainWorker(cmd, self._values["max_steps"])
        self._train.progress.connect(self._on_train_progress)
        self._train.log.connect(self._on_train_log)
        self._train.failed.connect(self._on_train_failed)
        self._train.finished.connect(self._on_train_finished)
        self._train.start()

    @Slot(dict)
    def _on_train_progress(self, state: dict):
        self._live.update_progress(state)
        step = state.get("step", 0)
        max_steps = max(1, self._values["max_steps"])
        progress = min(1.0, step / max_steps)
        self._set_state("train", "running", f"step {step:,} / {max_steps:,}", progress)

    @Slot(str, str)
    def _on_train_log(self, level: str, message: str):
        self._live.append_log(level, message)
        self._log(level, message)

    @Slot(str, str)
    def _on_train_failed(self, short: str, traceback: str):
        self._train = None
        self._live.show_failure(short, traceback)
        self._set_state("train", "failed", short)
        self._finish_workflow()

    @Slot(dict)
    def _on_train_finished(self, summary: dict):
        self._train = None
        steps = summary.get("steps_trained", 0)
        self._set_state("train", "done", f"trained to step {steps:,}", 1.0)
        if self._stop_requested:
            self._set_state("export", "stopped", "Stopped before export")
            self._finish_workflow()
            return
        self._set_state("export", "running", "Exporting model", 0.0)
        self._run_export()

    def _run_export(self):
        self._exp = ExportWorker(
            run_path=self._values["export_run"],
            output_name=self._values["export_name"],
            destination=self._values["export_dest"],
            streaming=self._values["streaming"],
        )
        self._exp.log.connect(lambda m: self._log("INFO", m))
        self._exp.finished.connect(self._on_export_finished)
        self._exp.start()

    @Slot(bool, str)
    def _on_export_finished(self, success: bool, message: str):
        self._exp = None
        if success:
            self._set_state("export", "done", message, 1.0)
        else:
            self._set_state("export", "failed", message)
        self._finish_workflow()

    def _finish_workflow(self):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
