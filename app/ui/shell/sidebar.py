from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QSizePolicy
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QFontDatabase

# Nav structure matching shell.jsx NAV
NAV = [
    {
        "group": "GENERATE & STREAM",
        "items": [
            {"id": "generate", "label": "Generate Audio"},
            {"id": "stream",   "label": "Streaming GUI"},
        ],
    },
    {
        "group": "DATA & TRAINING",
        "items": [
            {"id": "workflow",   "label": "Full Workflow"},
            {"id": "preprocess", "label": "Preprocess"},
            {"id": "train",      "label": "Train Model"},
            {"id": "export",     "label": "Export Model"},
            {"id": "prior",      "label": "Train Prior"},
            {"id": "phase",      "label": "Phase-aware Training"},
            {"id": "anchors",    "label": "Phase Anchors"},
            {"id": "dataset",    "label": "Dataset Creation"},
        ],
    },
    {
        "group": "MAINTENANCE",
        "items": [
            {"id": "clean", "label": "Clean User Data"},
        ],
    },
]

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
ACID  = "#a8e63d"


class NavItem(QWidget):
    clicked = Signal(str)

    def __init__(self, page_id: str, label: str, parent=None):
        super().__init__(parent)
        self._id = page_id
        self._label = label
        self._active = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_active(self, active: bool):
        self._active = active
        self.update()

    def mousePressEvent(self, event):
        self.clicked.emit(self._id)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()

        if self._active:
            p.setPen(QPen(QColor(LINE1), 1))
            p.setBrush(QColor(BG3))
            p.drawRoundedRect(r.adjusted(0, 0, -1, -1), 4, 4)
            # acid left bar
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACID))
            p.drawRoundedRect(-1, 6, 3, r.height() - 12, 2, 2)
        elif self.underMouse():
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(BG4))
            p.drawRoundedRect(r.adjusted(0, 0, -1, -1), 4, 4)

        font = self.font()
        font.setPointSizeF(9.5)
        p.setFont(font)
        p.setPen(QColor(FG0 if self._active else FG1))
        p.drawText(r.adjusted(12, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, self._label)
        p.end()

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, event):
        self.update()


class Sidebar(QWidget):
    navigate = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = "home"
        self._items: dict[str, NavItem] = {}
        self.setFixedWidth(220)
        self._build()

    def _build(self):
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(BG1))
        self.setPalette(pal)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Brand strip
        brand = QWidget()
        brand.setFixedHeight(52)
        brand.setAutoFillBackground(True)
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(14, 10, 14, 10)
        bl.setSpacing(10)

        logo = _LogoBox()
        bl.addWidget(logo)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)
        name_lbl = QLabel("RAVE-TFG")
        name_lbl.setStyleSheet(f"color:{FG0}; font-family:'JetBrains Mono','Consolas',monospace; font-size:11px; font-weight:600; letter-spacing:2px;")
        ver_lbl = QLabel("v0.4.2 · local")
        ver_lbl.setStyleSheet(f"color:{FG3}; font-family:'JetBrains Mono','Consolas',monospace; font-size:9px; letter-spacing:1px;")
        text_col.addWidget(name_lbl)
        text_col.addWidget(ver_lbl)
        bl.addLayout(text_col)
        bl.addStretch()
        root.addWidget(brand)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{LINE0};")
        root.addWidget(sep)

        # Scroll area for nav
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        nav_container = QWidget()
        nav_container.setStyleSheet("background: transparent;")
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(0)

        for group in NAV:
            grp_lbl = QLabel(group["group"])
            grp_lbl.setStyleSheet(f"color:{FG3}; font-family:'JetBrains Mono','Consolas',monospace; font-size:9px; letter-spacing:2px; text-transform:uppercase; padding: 6px 8px 4px;")
            nav_layout.addWidget(grp_lbl)

            for item in group["items"]:
                nav_item = NavItem(item["id"], item["label"])
                nav_item.clicked.connect(self._on_item_clicked)
                self._items[item["id"]] = nav_item
                nav_layout.addWidget(nav_item)

            nav_layout.addSpacing(8)

        nav_layout.addStretch()
        scroll.setWidget(nav_container)
        root.addWidget(scroll, 1)

        # GPU card
        gpu_card = _GpuCard()
        root.addWidget(gpu_card)

        # Right border
        self.setStyleSheet(f"Sidebar {{ border-right: 1px solid {LINE0}; }}")

    def _on_item_clicked(self, page_id: str):
        self.set_active(page_id)
        self.navigate.emit(page_id)

    def set_active(self, page_id: str):
        if self._active in self._items:
            self._items[self._active].set_active(False)
        self._active = page_id
        if page_id in self._items:
            self._items[page_id].set_active(True)


class _LogoBox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(ACID))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, 24, 24, 4, 4)
        # waveform path
        pen = QPen(QColor("#1e2320"), 1.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(2, 12)
        path.lineTo(4, 12)
        path.lineTo(6, 7)
        path.lineTo(8, 16)
        path.lineTo(10, 9)
        path.lineTo(12, 14)
        path.lineTo(14, 12)
        path.lineTo(16, 15)
        path.lineTo(18, 12)
        path.lineTo(20, 12)
        p.drawPath(path)
        p.end()


class _GpuCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(62)
        self.setContentsMargins(8, 0, 8, 8)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.setStyleSheet(f"_GpuCard {{ background:{BG2}; border:1px solid {LINE0}; border-radius:4px; margin:8px; }}")

        top = QHBoxLayout()
        top.setSpacing(6)
        gpu_lbl = QLabel("RTX 4090 · CUDA")
        gpu_lbl.setStyleSheet(f"color:{FG1}; font-family:'JetBrains Mono','Consolas',monospace; font-size:10px; letter-spacing:1px;")
        top.addWidget(gpu_lbl)
        top.addStretch()
        layout.addLayout(top)

        vram_row = QHBoxLayout()
        vram_lbl = QLabel("VRAM")
        vram_lbl.setStyleSheet(f"color:{FG2}; font-family:'JetBrains Mono','Consolas',monospace; font-size:10px;")
        vram_val = QLabel("6.2 / 24 GB")
        vram_val.setStyleSheet(f"color:{FG0}; font-family:'JetBrains Mono','Consolas',monospace; font-size:10px;")
        vram_row.addWidget(vram_lbl)
        vram_row.addStretch()
        vram_row.addWidget(vram_val)
        layout.addLayout(vram_row)

        self._bar = _VramBar(0.26)
        layout.addWidget(self._bar)


class _VramBar(QWidget):
    def __init__(self, fraction: float, parent=None):
        super().__init__(parent)
        self._fraction = fraction
        self.setFixedHeight(4)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(BG1))
        p.drawRoundedRect(r, 2, 2)
        fill_w = int(r.width() * self._fraction)
        if fill_w > 0:
            p.setBrush(QColor(ACID))
            p.drawRoundedRect(0, 0, fill_w, r.height(), 2, 2)
        p.end()
