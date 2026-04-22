"""Phase-aware training page."""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QPushButton, QLineEdit, QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    PageHeader, Panel, Field, FileInput, RadioGroup, section_title, _lbl,
    BG0, BG1, BG2, FG0, FG1, FG2, FG3, LINE0, LINE1, ACID, AMBER, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.dataset_worker import DatasetWorker

CONFIGS = ["v2_small", "v2", "v3", "v3_small"]


def _input(default: str, width: int | None = None) -> QLineEdit:
    w = QLineEdit(default)
    w.setStyleSheet(
        f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
        f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
    )
    if width:
        w.setFixedWidth(width)
    return w


class _PhaseRow(QWidget):
    def __init__(self, index: int, name: str, files: int, parent=None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self.setStyleSheet(f"background:{BG1}; border:1px solid {LINE0}; border-radius:3px;")
        hl = QHBoxLayout(self)
        hl.setContentsMargins(10, 8, 8, 8)
        hl.setSpacing(10)

        num_lbl = _lbl(str(index + 1).zfill(2), size=11, color=ACID, mono=True, bold=True)
        num_lbl.setFixedWidth(24)
        hl.addWidget(num_lbl)

        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(_lbl(f"{name}/", size=11, color=FG0, mono=True))
        col.addWidget(_lbl(f"{files} files", size=10, color=FG3, mono=True))
        hl.addLayout(col, 1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(2)
        up_btn = QPushButton("▲")
        up_btn.setFixedSize(22, 16)
        up_btn.setProperty("role", "ghost")
        dn_btn = QPushButton("▼")
        dn_btn.setFixedSize(22, 16)
        dn_btn.setProperty("role", "ghost")
        for b in (up_btn, dn_btn):
            b.setStyleSheet(
                f"QPushButton {{ background:transparent; color:{FG2}; border:none; font-size:8px; }}"
                f"QPushButton:hover {{ color:{FG0}; }}"
            )
        btn_col.addWidget(up_btn)
        btn_col.addWidget(dn_btn)
        hl.addLayout(btn_col)

        self._up = up_btn
        self._dn = dn_btn

    def connect_move(self, up_fn, dn_fn):
        self._up.clicked.connect(up_fn)
        self._dn.clicked.connect(dn_fn)


class _PhaseList(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._phases: list[tuple[str, int]] = []  # (name, file_count)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self.setStyleSheet("background:transparent;")

    def load(self, base_path: str):
        self._phases.clear()
        if not os.path.isdir(base_path):
            return
        audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
        for name in sorted(os.listdir(base_path)):
            full = os.path.join(base_path, name)
            if not os.path.isdir(full):
                continue
            n = sum(1 for f in os.listdir(full) if os.path.splitext(f)[1].lower() in audio_exts)
            self._phases.append((name, n))
        self._refresh()

    def _refresh(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (name, files) in enumerate(self._phases):
            row = _PhaseRow(i, name, files)
            row.connect_move(
                lambda _, j=i: self._move(j, -1),
                lambda _, j=i: self._move(j, +1),
            )
            self._layout.addWidget(row)
        self._layout.addStretch()

    def _move(self, idx: int, direction: int):
        j = idx + direction
        if 0 <= j < len(self._phases):
            self._phases[idx], self._phases[j] = self._phases[j], self._phases[idx]
            self._refresh()

    @property
    def ordered_names(self) -> list[str]:
        return [name for name, _ in self._phases]


class PhasePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: DatasetWorker | None = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"background:{BG0};")

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setProperty("role", "primary")
        self._start_btn.setFixedHeight(30)
        self._start_btn.clicked.connect(self._start)

        root.addWidget(PageHeader(
            crumbs=["Data & Training", "Phase-aware Training"],
            title="Phase-aware training",
            desc="Train a single model across multiple phase-tagged audio folders. Phase order controls the curriculum.",
            actions=[self._start_btn],
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        content.setStyleSheet(f"background:{BG0};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(16)

        cols = QHBoxLayout()
        cols.setSpacing(20)
        cols.addWidget(self._build_params(), 1)
        cols.addWidget(self._build_phase_panel(), 0)
        cl.addLayout(cols)

        self._prog = ProgressPanel()
        cl.addWidget(self._prog)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _build_params(self) -> QWidget:
        panel = Panel()
        panel.add_header(section_title("Parameters"))
        body = panel.body_layout()

        self._base = FileInput(placeholder="~/datasets/song-phases/")
        self._base.path_changed.connect(self._on_base_changed)
        body.addWidget(Field("Base folder", hint="Folder containing phase subfolders (intro/, verse/, …)", inline=True).add(self._base))

        self._name = _input("phases-model-v1")
        body.addWidget(Field("Model name", inline=True).add(self._name))

        self._config = QComboBox()
        self._config.addItems(CONFIGS)
        body.addWidget(Field("Config", inline=True).add(self._config))

        self._channels = RadioGroup(
            options=[{"value": "1", "label": "1"}, {"value": "2", "label": "2"}],
            value="1",
        )
        body.addWidget(Field("Channels", inline=True).add(self._channels))

        self._steps = _input("300000")
        self._batch = _input("8")
        body.addWidget(Field("Max steps", inline=True).add(self._steps))
        body.addWidget(Field("Batch size", inline=True).add(self._batch))
        body.addStretch()
        return panel

    def _build_phase_panel(self) -> QWidget:
        panel = Panel()
        panel.setFixedWidth(320)
        self._phase_count_lbl = _lbl("0 folders", size=10, color=FG2, mono=True)
        panel.add_header(section_title("Detected phases"), self._phase_count_lbl)

        body_w = QWidget()
        body_w.setStyleSheet("background:transparent;")
        bl = QVBoxLayout(body_w)
        bl.setContentsMargins(14, 10, 14, 14)
        bl.setSpacing(8)
        bl.addWidget(_lbl("Custom phase order — drag to reorder", size=11, color=FG2))
        self._phase_list = _PhaseList()
        bl.addWidget(self._phase_list)
        panel._root.addWidget(body_w, 1)
        return panel

    def _on_base_changed(self, path: str):
        self._phase_list.load(path)
        n = len(self._phase_list.ordered_names)
        self._phase_count_lbl.setText(f"{n} folders")

    def _start(self):
        if self._worker and self._worker.isRunning():
            return
        base = self._base.path
        if not base:
            return
        ordered = self._phase_list.ordered_names or None
        self._start_btn.setEnabled(False)
        self._prog.start("Running phase-aware training…")

        from src.phase_workflow import phase_train_workflow
        self._worker = DatasetWorker(
            phase_train_workflow,
            audio_base_path=base,
            phase_labels=ordered,
            model_name=self._name.text() or "phases-model-v1",
            config=self._config.currentText(),
            channels=int(self._channels.value),
            max_steps=int(self._steps.text() or 300000),
            batch_size=int(self._batch.text() or 8),
        )
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    @Slot(bool, str)
    def _on_done(self, ok: bool, msg: str):
        self._prog.finish(ok, msg)
        self._start_btn.setEnabled(True)
        self._worker = None
