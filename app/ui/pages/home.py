"""Home dashboard page."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.pages._home_widgets import (
    ACID,
    BG0,
    BG1,
    BG2,
    FG1,
    FG2,
    LINE0,
    RECENT,
    SECTIONS,
    STATS,
    _RecentRow,
    _SectionGroup,
    _StatCard,
    _lbl,
)


class HomePage(QWidget):
    navigate = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"HomePage {{ background:{BG0}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ background:{BG0}; }}")

        content = QWidget()
        vl = QVBoxLayout(content)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        vl.addWidget(self._build_hero())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(28, 24, 28, 24)
        body_layout.setSpacing(24)
        body_layout.addWidget(self._build_left(), 1)
        body_layout.addWidget(self._build_right(), 0)

        vl.addWidget(body)
        vl.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_hero(self) -> QWidget:
        hero = QWidget()
        hero.setObjectName("homeHero")
        hero.setAttribute(Qt.WA_StyledBackground, True)
        hero.setStyleSheet(
            f"QWidget#homeHero {{"
            f"background: qlineargradient(x1:1,y1:0,x2:0,y2:1, stop:0 #223320, stop:1 {BG1});"
            f"border-bottom:1px solid {LINE0};"
            f"}}"
        )
        layout = QVBoxLayout(hero)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(0)

        layout.addWidget(_lbl("RAVE · REALTIME AUDIO VARIATIONAL AUTOENCODER", size=10, color=ACID, mono=True, spacing="3px"))

        top = QHBoxLayout()
        top.setSpacing(24)
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        h1 = _lbl("Train, export and stream neural audio models.", size=24, color="#f0f4ee", bold=True)
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

        layout.addSpacing(20)
        stat_grid = QHBoxLayout()
        stat_grid.setSpacing(12)
        for s in STATS:
            stat_grid.addWidget(_StatCard(s["label"], s["value"], s["sub"]))
        layout.addLayout(stat_grid)

        return hero

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
