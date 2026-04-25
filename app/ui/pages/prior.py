"""Train prior page."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QPushButton, QLineEdit, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.widgets.form import (
    PageHeader, Panel, Field, FileInput, section_title, _lbl,
    BG0, BG1, BG2, BG3, FG0, FG1, FG2, FG3, LINE0, LINE1,
    ACID, ACIDDIM, ACIDBG, AMBER, MAG, BLUE, MONO,
)
from app.ui.widgets.progress_panel import ProgressPanel
from app.workers.dataset_worker import DatasetWorker

CONFIGS = [
    ("decoder_only",   "decoder_only",  "transformer decoder"),
    ("recurrent",      "recurrent",     "LSTM · recommended"),
    ("encoder_decoder","enc+dec",       "translation-style"),
]


class _ConfigCard(QWidget):
    def __init__(self, value: str, label: str, desc: str, active: bool = False, parent=None):
        super().__init__(parent)
        self.value = value
        self._active = active
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(52)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)
        self._lbl = _lbl(label, size=11, color=ACID if active else FG0, mono=True)
        self._desc = _lbl(desc, size=10, color=FG3)
        layout.addWidget(self._lbl)
        layout.addWidget(self._desc)

    def set_active(self, active: bool):
        self._active = active
        self._lbl.setStyleSheet(
            f"color:{ACID if active else FG0}; font-size:11px; {MONO} background:transparent;"
        )
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        if self._active:
            p.setPen(QPen(QColor(ACIDDIM), 1))
            p.setBrush(QColor(ACIDBG))
        else:
            p.setPen(QPen(QColor(LINE1), 1))
            p.setBrush(QColor(BG1))
        p.drawRoundedRect(r, 4, 4)
        p.end()


class PriorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: DatasetWorker | None = None
        self._config = "recurrent"
        self._cards: list[_ConfigCard] = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._start_btn = QPushButton("▶  Start training")
        self._start_btn.setProperty("role", "primary")
        self._start_btn.setFixedHeight(30)
        self._start_btn.clicked.connect(self._start)

        root.addWidget(PageHeader(
            crumbs=["Data & Training", "Train Prior"],
            title="Train prior (advanced)",
            desc="Train an autoregressive prior over the latent space of a trained RAVE model.",
            actions=[self._start_btn],
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(24, 24, 24, 24)
        cl.setSpacing(16)

        # Info banner
        banner = QWidget()
        banner.setStyleSheet(
            f"background: rgba(30,48,80,0.9); border:1px solid {BLUE}; border-radius:4px;"
        )
        bl = QHBoxLayout(banner)
        bl.setContentsMargins(14, 12, 14, 12)
        bl.setSpacing(10)
        bl.addWidget(_lbl("ℹ", size=13, color=BLUE))
        info_txt = _lbl(
            "Priors are optional. Train one when you want the Streaming GUI to synthesize "
            "without an input signal, or to generate plausible latent trajectories instead of pure noise.",
            size=12, color=FG1, wrap=True,
        )
        bl.addWidget(info_txt, 1)
        cl.addWidget(banner)

        # Parameters panel
        panel = Panel()
        panel.add_header(section_title("Parameters"))
        body = panel.body_layout()

        self._ckpt = FileInput(placeholder="~/exports/model_streaming.ts", directory=False)
        body.addWidget(Field("RAVE checkpoint", hint="The trained model whose latent space we're modeling").add(self._ckpt))

        self._audio = FileInput(placeholder="~/preprocessed/data.lmdb", directory=True)
        body.addWidget(Field("Audio dataset").add(self._audio))

        self._name_edit = _make_input("my_prior")
        body.addWidget(Field("Prior name").add(self._name_edit))

        # Config cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        for value, label, desc in CONFIGS:
            card = _ConfigCard(value, label, desc, active=(value == self._config))
            card.mousePressEvent = lambda e, v=value, c=card: self._select_config(v)
            self._cards.append(card)
            cards_row.addWidget(card)
        body.addWidget(Field("Config").add(_cards_widget(cards_row)))

        self._steps_edit = _make_input("200000")
        self._batch_edit  = _make_input("16")

        row = QHBoxLayout()
        row.setSpacing(14)
        lf = QVBoxLayout(); lf.addWidget(Field("Max steps").add(self._steps_edit))
        rf = QVBoxLayout(); rf.addWidget(Field("Batch size").add(self._batch_edit))
        row.addLayout(lf); row.addLayout(rf)
        body.addLayout(row)

        cl.addWidget(panel)

        self._prog = ProgressPanel()
        cl.addWidget(self._prog)
        cl.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def _select_config(self, value: str):
        self._config = value
        for card in self._cards:
            card.set_active(card.value == value)

    def _start(self):
        if self._worker and self._worker.isRunning():
            return
        ckpt  = self._ckpt.path
        audio = self._audio.path
        if not ckpt or not audio:
            self._prog.start("Missing inputs")
            self._prog.finish(False, "Select a RAVE checkpoint and audio dataset first.")
            return

        self._start_btn.setEnabled(False)
        self._prog.start("Training prior…")

        from src.train_prior import TrainPrior
        self._worker = DatasetWorker(
            TrainPrior,
            rave_model_path=ckpt,
            audio_path=audio,
            prior_name=self._name_edit.text() or "my_prior",
            config=self._config,
        )
        self._worker.log.connect(self._prog.append_log)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    @Slot(bool, str)
    def _on_done(self, ok: bool, msg: str):
        self._prog.finish(ok, msg)
        self._start_btn.setEnabled(True)
        self._worker = None


def _make_input(default: str) -> QLineEdit:
    w = QLineEdit(default)
    w.setStyleSheet(
        f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
        f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
    )
    return w


def _cards_widget(layout: QHBoxLayout) -> QWidget:
    w = QWidget()
    w.setStyleSheet("background:transparent;")
    w.setLayout(layout)
    return w
