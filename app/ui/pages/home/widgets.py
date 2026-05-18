"""Home page helper widgets and data constants."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

BG0   = "#1e2320"
BG1   = "#232926"
BG2   = "#28302b"
BG3   = "#2f3832"
BG4   = "#37413a"
FG0   = "#f0f4ee"
FG1   = "#c6cdc3"
FG2   = "#96a092"
FG3   = "#717870"
LINE0 = "#333c36"
LINE1 = "#3f4b42"
LINE2 = "#4e5c51"
ACID  = "#a8e63d"
AMBER = "#e8c040"
MAG   = "#e0406a"
BLUE  = "#5090d8"
MONO  = "font-family:'JetBrains Mono','Consolas',monospace;"

SECTIONS = [
    {
        "title": "Generate & Stream",
        "desc": "Run inference on trained models. Real-time multi-model streaming.",
        "hue": ACID,
        "tiles": [
            {"id": "generate", "label": "Generate audio", "sub": "offline render"},
            {"id": "stream",   "label": "Streaming GUI",  "sub": "realtime · multi-model"},
        ],
    },
    {
        "title": "Core Workflow",
        "desc": "Main pipeline: preprocess, train, and export.",
        "hue": AMBER,
        "tiles": [
            {"id": "dataset", "label": "Dataset creation", "sub": "wizard · 7 steps"},
            {"id": "train",   "label": "Train model",      "sub": "pipeline: preprocess → train → export"},
        ],
    },
    {
        "title": "Training Extras",
        "desc": "Optional advanced modules outside the main workflow.",
        "hue": BLUE,
        "tiles": [
            {"id": "anchors", "label": "Phase anchors", "sub": "existing model"},
        ],
    },
    {
        "title": "Maintenance",
        "desc": "Workspace hygiene. Destructive operations live here.",
        "hue": MAG,
        "tiles": [
            {"id": "clean", "label": "Clean user data", "sub": "preproc · ckpt · exports · outputs"},
        ],
    },
]



def _lbl(text, size=12, color=FG0, bold=False, mono=False, spacing=None):
    l = QLabel(text)
    ff = MONO if mono else "font-family:'Inter','Segoe UI',sans-serif;"
    fw = "font-weight:600;" if bold else ""
    ls = f"letter-spacing:{spacing};" if spacing else ""
    l.setStyleSheet(f"color:{color}; font-size:{size}px; {ff} {fw} {ls} background:transparent;")
    return l


class _TileCard(QWidget):
    clicked = Signal(str)

    def __init__(self, tile: dict, hue: str, parent=None):
        super().__init__(parent)
        self._id = tile["id"]
        self._hue = QColor(hue)
        self._hover = False
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(72)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        layout.addWidget(_lbl(tile["label"], size=12, color=FG0, bold=True))
        layout.addWidget(_lbl(tile["sub"], size=10, color=FG3, mono=True))

    def mousePressEvent(self, e):
        self.clicked.emit(self._id)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        border = QColor(LINE2 if self._hover else LINE0)
        bg = QColor(BG3 if self._hover else BG2)
        p.setPen(QPen(border, 1))
        p.setBrush(bg)
        p.drawRoundedRect(r, 4, 4)
        p.setPen(Qt.NoPen)
        p.setBrush(self._hue)
        p.drawEllipse(r.right() - 16, r.top() + 10, 6, 6)
        p.end()


class _SectionGroup(QWidget):
    navigate = Signal(str)

    def __init__(self, sec: dict, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        accent = QWidget()
        accent.setFixedSize(2, 18)
        accent.setStyleSheet(f"background:{sec['hue']}; border-radius:1px;")
        header.addWidget(accent)
        header.addWidget(_lbl(sec["title"], size=13, color=FG0, bold=True))
        header.addWidget(_lbl(f"— {sec['desc']}", size=10, color=FG3, mono=True))
        header.addStretch()
        layout.addLayout(header)

        grid = QWidget()
        grid_layout = QGridLayout(grid)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(8)
        cols = 4
        for i, tile in enumerate(sec["tiles"]):
            card = _TileCard(tile, sec["hue"])
            card.clicked.connect(self.navigate)
            grid_layout.addWidget(card, i // cols, i % cols)

        layout.addWidget(grid)
