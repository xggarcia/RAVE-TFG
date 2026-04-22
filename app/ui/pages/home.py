from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QPushButton, QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen

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

SECTIONS = [
    {
        "title": "Generate & Stream",
        "desc": "Run inference on trained models. Real-time multi-model streaming.",
        "hue": ACID,
        "tiles": [
            {"id": "generate", "label": "Generate audio",  "sub": "offline render"},
            {"id": "stream",   "label": "Streaming GUI",   "sub": "realtime · multi-model"},
        ],
    },
    {
        "title": "Data & Training",
        "desc": "Preprocess datasets, train models and priors, export for streaming.",
        "hue": AMBER,
        "tiles": [
            {"id": "workflow",   "label": "Full workflow",        "sub": "preprocess → train → export"},
            {"id": "preprocess", "label": "Preprocess dataset",   "sub": "audio → lmdb"},
            {"id": "train",      "label": "Train model",          "sub": "resume aware"},
            {"id": "export",     "label": "Export model",         "sub": "ts · streaming"},
            {"id": "prior",      "label": "Train prior",          "sub": "advanced"},
            {"id": "phase",      "label": "Phase-aware training",  "sub": "multi-phase"},
            {"id": "anchors",    "label": "Phase anchors",         "sub": "existing model"},
            {"id": "dataset",    "label": "Dataset creation",      "sub": "wizard · 7 steps"},
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
    {"kind": "train",      "name": "vox-phase3-v2",       "meta": "step 184,200 / 300,000", "progress": 0.61, "status": "running", "time": "now"},
    {"kind": "generate",   "name": "pad_ambient_03.wav",   "meta": "8.4s · demo_guitar.ts",  "status": "done", "time": "12m ago"},
    {"kind": "export",     "name": "vox-phase2",           "meta": "streaming · 44.1 kHz",   "status": "done", "time": "2h ago"},
    {"kind": "preprocess", "name": "field-recordings-v2",  "meta": "1,247 files · 3.8 GB",   "status": "done", "time": "yesterday"},
]

STATS = [
    {"label": "Models trained", "value": "14",       "sub": "+3 this week"},
    {"label": "Active runs",    "value": "1",        "sub": "vox-phase3-v2 · 61%"},
    {"label": "Datasets",       "value": "7",        "sub": "12.4 GB total"},
    {"label": "GPU",            "value": "RTX 4090", "sub": "CUDA 12.1 · 6.2/24 GB"},
]

MONO = "font-family:'JetBrains Mono','Consolas',monospace;"


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
        # hue accent dot top-right
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
        color = ACID if entry["status"] == "running" else FG2
        top.addWidget(_lbl(entry["name"], size=12, color=FG0, bold=True))
        top.addStretch()
        top.addWidget(_lbl(entry["time"], size=10, color=FG3, mono=True))
        layout.addLayout(top)

        layout.addWidget(_lbl(entry["meta"], size=10, color=FG2, mono=True))

        if entry.get("progress") is not None:
            bar = _ProgressBar(entry["progress"])
            layout.addWidget(bar)

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

        # Header row
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

        # Tile grid
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


class HomePage(QWidget):
    navigate = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"HomePage {{ background:{BG0}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        content.setStyleSheet(f"background:{BG0};")
        vl = QVBoxLayout(content)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        vl.addWidget(self._build_hero())

        body = QWidget()
        body.setStyleSheet(f"background:{BG0};")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(28, 24, 28, 24)
        body_layout.setSpacing(24)
        body_layout.addWidget(self._build_left(), 1)
        body_layout.addWidget(self._build_right(), 0)

        vl.addWidget(body)
        vl.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── Hero ────────────────────────────────────────────────

    def _build_hero(self) -> QWidget:
        hero = QWidget()
        hero.setStyleSheet(
            f"background: qlineargradient(x1:1,y1:0,x2:0,y2:1, stop:0 #223320, stop:1 {BG1});"
            f"border-bottom:1px solid {LINE0};"
        )
        layout = QVBoxLayout(hero)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(0)

        layout.addWidget(_lbl("RAVE · REALTIME AUDIO VARIATIONAL AUTOENCODER", size=10, color=ACID, mono=True, spacing="3px"))

        top = QHBoxLayout()
        top.setSpacing(24)
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        h1 = _lbl("Train, export and stream neural audio models.", size=24, color=FG0, bold=True)
        h1.setWordWrap(True)
        left_col.addWidget(h1)
        left_col.addWidget(_lbl("Pick a task below, or resume your last session.", size=12, color=FG2))
        top.addLayout(left_col, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        resume_btn = QPushButton("Resume last")
        resume_btn.setFixedHeight(30)
        new_btn = QPushButton("New training run")
        new_btn.setProperty("role", "primary")
        new_btn.setFixedHeight(30)
        new_btn.clicked.connect(lambda: self.navigate.emit("train"))
        btn_row.addWidget(resume_btn)
        btn_row.addWidget(new_btn)
        top.addLayout(btn_row)

        layout.addSpacing(8)
        layout.addLayout(top)

        # Stat grid
        layout.addSpacing(20)
        stat_grid = QHBoxLayout()
        stat_grid.setSpacing(12)
        for s in STATS:
            stat_grid.addWidget(_StatCard(s["label"], s["value"], s["sub"]))
        layout.addLayout(stat_grid)

        return hero

    # ── Left col (sections) ─────────────────────────────────

    def _build_left(self) -> QWidget:
        left = QWidget()
        left.setStyleSheet("background:transparent;")
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        for sec in SECTIONS:
            group = _SectionGroup(sec)
            group.navigate.connect(self.navigate)
            layout.addWidget(group)

        layout.addStretch()
        return left

    # ── Right col (recent + tips) ────────────────────────────

    def _build_right(self) -> QWidget:
        right = QWidget()
        right.setFixedWidth(300)
        right.setStyleSheet("background:transparent;")
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(_lbl("RECENT ACTIVITY", size=10, color=FG2, mono=True, spacing="2px"))
        layout.addSpacing(8)

        activity_box = QWidget()
        activity_box.setStyleSheet(f"background:{BG2}; border:1px solid {LINE0}; border-radius:4px;")
        ab_layout = QVBoxLayout(activity_box)
        ab_layout.setContentsMargins(0, 0, 0, 0)
        ab_layout.setSpacing(0)
        for i, r in enumerate(RECENT):
            ab_layout.addWidget(_RecentRow(r, i == len(RECENT) - 1))
        layout.addWidget(activity_box)

        layout.addSpacing(20)
        layout.addWidget(_lbl("QUICK TIPS", size=10, color=FG2, mono=True, spacing="2px"))
        layout.addSpacing(8)

        tip_box = QWidget()
        tip_box.setStyleSheet(f"background:{BG2}; border:1px solid {LINE0}; border-radius:4px;")
        tip_layout = QVBoxLayout(tip_box)
        tip_layout.setContentsMargins(12, 12, 12, 12)
        tip_text = QLabel(
            "Training typically takes <b>15–24h</b> on cloud GPU. "
            "Runs persist between sessions."
        )
        tip_text.setWordWrap(True)
        tip_text.setStyleSheet(f"color:{FG1}; font-size:12px; background:transparent;")
        tip_layout.addWidget(tip_text)
        layout.addWidget(tip_box)

        layout.addStretch()
        return right
