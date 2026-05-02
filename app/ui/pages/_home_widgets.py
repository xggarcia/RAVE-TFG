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

RECENT = [
    {"kind": "train",      "name": "vox-phase3-v2",      "meta": "step 184,200 / 300,000", "progress": 0.61, "status": "running", "time": "now"},
    {"kind": "generate",   "name": "pad_ambient_03.wav",  "meta": "8.4s · demo_guitar.ts",  "status": "done", "time": "12m ago"},
    {"kind": "export",     "name": "vox-phase2",          "meta": "streaming · 44.1 kHz",   "status": "done", "time": "2h ago"},
    {"kind": "preprocess", "name": "field-recordings-v2", "meta": "1,247 files · 3.8 GB",   "status": "done", "time": "yesterday"},
]

def _get_gpu_stat():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            mem = torch.cuda.get_device_properties(0).total_mem
            total_gb = mem / (1024 ** 3)
            return {"label": "GPU", "value": name, "sub": f"CUDA {cap[0]}.{cap[1]} · {total_gb:.0f} GB"}
    except Exception:
        pass
    return {"label": "GPU", "value": "CPU only", "sub": "No CUDA device"}

STATS = [
    {"label": "Models trained", "value": "14",       "sub": "+3 this week"},
    {"label": "Active runs",    "value": "1",        "sub": "vox-phase3-v2 · 61%"},
    {"label": "Datasets",       "value": "7",        "sub": "12.4 GB total"},
    _get_gpu_stat(),
]


def _lbl(text, size=12, color=FG0, bold=False, mono=False, spacing=None):
    l = QLabel(text)
    ff = MONO if mono else "font-family:'Inter','Segoe UI',sans-serif;"
    fw = "font-weight:600;" if bold else ""
    ls = f"letter-spacing:{spacing};" if spacing else ""
    l.setStyleSheet(f"color:{color}; font-size:{size}px; {ff} {fw} {ls} background:transparent;")
    return l


class _ProgressBar(QWidget):
    def __init__(self, value: float, parent=None):
        super().__init__(parent)
        self._value = value
        self.setFixedHeight(3)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(BG1))
        p.drawRoundedRect(r, 1, 1)
        w = int(r.width() * self._value)
        if w:
            p.setBrush(QColor(ACID))
            p.drawRoundedRect(0, 0, w, r.height(), 1, 1)
        p.end()


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


class _StatCard(QWidget):
    def __init__(self, label: str, value: str, sub: str, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(72)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        layout.addWidget(_lbl(label, size=10, color=FG2, mono=True, spacing="1px"))
        layout.addWidget(_lbl(value, size=18, color=FG0, bold=True))
        layout.addWidget(_lbl(sub, size=10, color=FG3, mono=True))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(QPen(QColor(LINE0), 1))
        p.setBrush(QColor(BG2))
        p.drawRoundedRect(r, 4, 4)
        p.end()


class _RecentRow(QWidget):
    def __init__(self, entry: dict, last: bool, parent=None):
        super().__init__(parent)
        self._last = last
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.addWidget(_lbl(entry["name"], size=12, color=FG0, bold=True))
        top.addStretch()
        top.addWidget(_lbl(entry["time"], size=10, color=FG3, mono=True))
        layout.addLayout(top)

        layout.addWidget(_lbl(entry["meta"], size=10, color=FG2, mono=True))

        if entry.get("progress") is not None:
            layout.addWidget(_ProgressBar(entry["progress"]))

    def paintEvent(self, event):
        p = QPainter(self)
        if not self._last:
            p.setPen(QPen(QColor(LINE0), 1))
            p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
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
