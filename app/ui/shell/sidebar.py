"""Sidebar with grouped navigation. Uses native QPushButton (checkable) for items."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.tokens import ACID, BG1, BG2, BG3, BG4, FG0, FG1, FG3, LINE0, LINE1, MONO

NAV = [
    {
        "group": "GENERATE & STREAM",
        "items": [
            {"id": "generate", "label": "Generate Audio"},
            {"id": "stream",   "label": "Streaming GUI"},
        ],
    },
    {
        "group": "CORE WORKFLOW",
        "items": [
            {"id": "dataset", "label": "Dataset Creation"},
            {"id": "train",   "label": "Train Model"},
        ],
    },
    {
        "group": "TRAINING EXTRAS",
        "items": [
            # {"id": "phase",    "label": "Phase-aware Training"},  # disabled for now
            {"id": "anchors",    "label": "Phase Anchors"},
        ],
    },
    {
        "group": "MAINTENANCE",
        "items": [
            {"id": "clean", "label": "Clean User Data"},
        ],
    },
]

_NAV_BTN_QSS = f"""
QPushButton {{
    color: {FG1};
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    text-align: left;
    padding: 6px 10px;
    font-size: 12px;
}}
QPushButton:hover {{
    background: {BG4};
}}
QPushButton:checked {{
    color: {FG0};
    background: {BG3};
    border: 1px solid {LINE1};
    border-left: 3px solid {ACID};
}}
"""


class Sidebar(QWidget):
    navigate = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, QPushButton] = {}
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
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(14, 10, 14, 10)
        bl.setSpacing(10)

        logo = QLabel("◊")
        logo.setFixedSize(24, 24)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            f"background:{ACID}; color:{BG1}; border-radius:4px; "
            f"font-size:14px; font-weight:700;"
        )
        bl.addWidget(logo)

        name_btn = QPushButton("RAVE-TFG")
        name_btn.setFlat(True)
        name_btn.setCursor(Qt.PointingHandCursor)
        name_btn.setStyleSheet(
            f"color:{FG0}; {MONO} font-size:11px; font-weight:600; letter-spacing:2px; "
            f"background:transparent; border:none; padding:0;"
        )
        name_btn.clicked.connect(lambda: self._on_clicked("home"))
        bl.addWidget(name_btn)
        bl.addStretch()
        root.addWidget(brand)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{LINE0};")
        root.addWidget(sep)

        # Nav buttons (checkable, grouped so only one is checked at a time)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        nav_layout = QVBoxLayout(container)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(2)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for group in NAV:
            grp_lbl = QLabel(group["group"])
            grp_lbl.setStyleSheet(
                f"color:{FG3}; {MONO} font-size:9px; letter-spacing:2px; padding: 6px 8px 4px;"
            )
            nav_layout.addWidget(grp_lbl)

            for item in group["items"]:
                btn = QPushButton(item["label"])
                btn.setCheckable(True)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setFixedHeight(32)
                btn.setStyleSheet(_NAV_BTN_QSS)
                btn.clicked.connect(lambda _checked, pid=item["id"]: self._on_clicked(pid))
                self._group.addButton(btn)
                self._items[item["id"]] = btn
                nav_layout.addWidget(btn)

            nav_layout.addSpacing(8)

        nav_layout.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        root.addWidget(_GpuCard())

        self.setStyleSheet(f"Sidebar {{ border-right: 1px solid {LINE0}; }}")

    def _on_clicked(self, page_id: str):
        self.set_active(page_id)
        self.navigate.emit(page_id)

    def set_active(self, page_id: str):
        if page_id in self._items:
            self._items[page_id].setChecked(True)


def _get_gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            return f"{name} · CUDA {cap[0]}.{cap[1]}"
    except Exception:
        pass
    return "CPU only"


class _GpuCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
        self.setStyleSheet(f"_GpuCard {{ background:{BG2}; border:1px solid {LINE0}; border-radius:4px; margin:8px; }}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        gpu_lbl = QLabel(_get_gpu_info())
        gpu_lbl.setStyleSheet(f"color:{FG1}; {MONO} font-size:10px; letter-spacing:1px;")
        layout.addWidget(gpu_lbl)
        layout.addStretch()
