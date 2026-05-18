"""Workflow config constants, chip widget, and config panel builder."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QWidget,
)

from app.ui.widgets.form import (
    ACID,
    BG1,
    FG0,
    FG1,
    FG2,
    FG3,
    LINE1,
    MONO,
    Field,
    FileInput,
    Panel,
    RadioGroup,
    Toggle,
    _lbl,
    section_title,
)

if TYPE_CHECKING:
    from app.ui.pages.workflow.page import WorkflowPage

EXTRA_CONFIGS = [
    ("noise",    "noise injection"),
    ("causal",   "causal convolutions"),
    ("snake",    "snake activations"),
    ("hinge",    "hinge discriminator"),
    ("descript", "descript discriminator"),
]
DEFAULT_ON = {"noise", "causal"}


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


def build_config_panel(page: "WorkflowPage") -> QWidget:
    """Build the run-configuration panel; sets form-field attributes on *page*."""
    page._chips: list[_ConfigChip] = []

    panel = Panel()
    panel.add_header(section_title("Run configuration"), _lbl("active", 10, ACID, mono=True))
    body = panel.body_layout()

    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(10)

    _line = (
        f"background:{BG1}; color:{FG0}; border:1px solid {LINE1};"
        f" border-radius:4px; padding:6px 10px; {MONO}"
    )
    page._audio_input  = FileInput(placeholder="~/datasets/vocals-v2/", directory=True)
    page._model_name   = QLineEdit("vox-phase3-v2")
    page._model_name.setStyleSheet(_line)
    page._config       = QComboBox()
    page._config.addItems(["v2_small", "v2", "v3", "v3_small"])
    page._config.setCurrentText("v2")
    page._channels     = RadioGroup(
        options=[{"value": "1", "label": "Mono (1)"}, {"value": "2", "label": "Stereo (2)"}],
        value="1",
    )
    page._max_steps    = QLineEdit("500000")
    page._max_steps.setStyleSheet(_line)
    page._batch_size   = QLineEdit("8")
    page._batch_size.setStyleSheet(_line)
    page._destination  = FileInput(placeholder="models/user_model/exported_model", directory=True)
    page._val_every    = QLineEdit("10000")
    page._val_every.setStyleSheet(_line)
    page._save_every   = QLineEdit("25000")
    page._save_every.setStyleSheet(_line)
    page._export_run   = FileInput(placeholder="auto-detect from training", directory=True)
    page._streaming    = Toggle(on=True)
    page._export_name  = QLineEdit("model_streaming.ts")
    page._export_name.setStyleSheet(_line)

    widgets = [
        Field("Audio folder", inline=False).add(page._audio_input),
        Field("Model name", inline=False).add(page._model_name),
        Field("Config", inline=False).add(page._config),
        Field("Channels", inline=False).add(page._channels),
        Field("Max steps", inline=False).add(page._max_steps),
        Field("Batch size", inline=False).add(page._batch_size),
        Field("Val every", inline=False).add(page._val_every),
        Field("Save every", inline=False).add(page._save_every),
        Field("Export destination", inline=False).add(page._destination),
        Field("Export run folder", hint="Leave blank to auto-detect after training.", inline=False).add(page._export_run),
        Field("Streaming", hint="Required for real-time use.", inline=False).add(page._streaming),
        Field("Output name", inline=False).add(page._export_name),
    ]
    for i, widget in enumerate(widgets):
        grid.addWidget(widget, i // 4, i % 4)

    body.addLayout(grid)

    body.addWidget(_lbl("Extra configs", size=10, color=FG2, mono=True, spacing="1px"))
    chips_row = QHBoxLayout()
    chips_row.setSpacing(8)
    for key, desc in EXTRA_CONFIGS:
        chip = _ConfigChip(key, desc, on=(key in DEFAULT_ON))
        page._chips.append(chip)
        chips_row.addWidget(chip)
    chips_row.addStretch()
    body.addLayout(chips_row)

    return panel
