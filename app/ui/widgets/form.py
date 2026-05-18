"""Shared form primitives: Field, FileInput, RadioGroup, Toggle, PageHeader, Panel."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QButtonGroup, QCheckBox, QRadioButton, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from app.ui.tokens import (
    ACID, ACIDBG, ACIDDIM, AMBER, BG0, BG1, BG2, BG3, BG4,
    BLUE, FG0, FG1, FG2, FG3, LINE0, LINE1, LINE2, MAG, MONO,
)


def _lbl(text, size=12, color=FG0, bold=False, mono=False, spacing=None, wrap=False):
    l = QLabel(text)
    ff = MONO if mono else ""
    fw = "font-weight:600;" if bold else ""
    ls = f"letter-spacing:{spacing};" if spacing else ""
    l.setStyleSheet(f"color:{color}; font-size:{size}px; {ff} {fw} {ls} background:transparent;")
    if wrap:
        l.setWordWrap(True)
    return l


# ── Panel ────────────────────────────────────────────────────────────────────

class Panel(QWidget):
    """A card with bg-2 fill, border, rounded corners; optional header."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(QPen(QColor(LINE0), 1))
        p.setBrush(QColor(BG2))
        p.drawRoundedRect(r, 6, 6)
        p.end()

    def add_header(self, left_widget: QWidget, right_widget: QWidget | None = None) -> QWidget:
        header = QWidget()
        header.setFixedHeight(38)
        header.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {BG3}, stop:1 {BG2});"
            f"border-bottom: 1px solid {LINE0}; border-radius: 0;"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.addWidget(left_widget)
        if right_widget:
            hl.addWidget(right_widget)
        self._root.addWidget(header)
        return header

    def add_body(self) -> QWidget:
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 14, 14, 14)
        bl.setSpacing(14)
        self._root.addWidget(body, 1)
        return body

    def body_layout(self) -> QVBoxLayout:
        body = self.add_body()
        return body.layout()


# ── PageHeader ───────────────────────────────────────────────────────────────

class PageHeader(QWidget):
    def __init__(self, crumbs: list[str], title: str, desc: str = "",
                 actions: list[QWidget] | None = None, parent=None):
        super().__init__(parent)
        self.setFixedHeight(90 if desc else 70)
        self.setAttribute(Qt.WA_StyledBackground, False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.setSpacing(4)

        if crumbs:
            breadcrumb = QLabel(" / ".join(crumbs))
            breadcrumb.setStyleSheet(
                f"color:{FG2}; font-size:10px; {MONO} letter-spacing:2px; background:transparent;"
            )
            layout.addWidget(breadcrumb)

        row = QHBoxLayout()
        row.setSpacing(16)
        title_col = QVBoxLayout()
        title_col.setSpacing(3)
        h1 = _lbl(title, size=20, color=FG0, bold=True)
        title_col.addWidget(h1)
        if desc:
            title_col.addWidget(_lbl(desc, size=12, color=FG2, wrap=True))
        row.addLayout(title_col, 1)

        if actions:
            btn_row = QHBoxLayout()
            btn_row.setSpacing(8)
            for btn in actions:
                btn_row.addWidget(btn)
            row.addLayout(btn_row)

        layout.addLayout(row)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG1))
        p.setPen(QPen(QColor(LINE0), 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


# ── Field ────────────────────────────────────────────────────────────────────

class Field(QWidget):
    """Label (+ optional hint) above or beside a child widget."""

    def __init__(self, label: str, hint: str = "", inline: bool = False, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        if inline:
            root = QHBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(12)
            lbl_col = QVBoxLayout()
            lbl_col.setSpacing(2)
            lbl_col.addWidget(_lbl(label, size=11, color=FG1))
            if hint:
                lbl_col.addWidget(_lbl(hint, size=10, color=FG3))
            lbl_w = QWidget()
            lbl_w.setFixedWidth(160)
            lbl_w.setStyleSheet("background:transparent;")
            lbl_w.setLayout(lbl_col)
            root.addWidget(lbl_w)
            self._content_layout = root
        else:
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(5)
            root.addWidget(_lbl(label, size=11, color=FG1))
            if hint:
                root.addWidget(_lbl(hint, size=10, color=FG3))
            self._content_layout = root

    def set_content(self, widget: QWidget):
        self._content_layout.addWidget(widget)
        return self

    def add(self, widget: QWidget) -> "Field":
        return self.set_content(widget)


# ── FileInput ────────────────────────────────────────────────────────────────

class FileInput(QWidget):
    path_changed = Signal(str)

    def __init__(self, value: str = "", placeholder: str = "Select path…",
                 directory: bool = True, save: bool = False, parent=None):
        super().__init__(parent)
        self._path = value
        self._dir = directory
        self._save = save
        self.setStyleSheet("background:transparent;")

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        self._display = QLabel(value or placeholder)
        self._display.setStyleSheet(
            f"background:{BG1}; color:{''+FG0 if value else FG3}; {MONO} font-size:12px;"
            f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
        )
        self._display.setMinimumWidth(120)
        self._display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        hl.addWidget(self._display, 1)

        browse = QPushButton("Browse…")
        browse.setFixedHeight(32)
        browse.clicked.connect(self._browse)
        hl.addWidget(browse)

        self._placeholder = placeholder

    def _browse(self):
        if self._dir:
            path = QFileDialog.getExistingDirectory(self, "Select folder", self._path or "")
        elif self._save:
            path, _ = QFileDialog.getSaveFileName(self, "Save file", self._path or "")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", self._path or "")
        if path:
            self.set_path(path)

    def set_path(self, path: str):
        self._path = path
        self._display.setText(path)
        self._display.setStyleSheet(
            f"background:{BG1}; color:{FG0}; {MONO} font-size:12px;"
            f"border:1px solid {LINE1}; border-radius:4px; padding:6px 10px;"
        )
        self.path_changed.emit(path)

    @property
    def path(self) -> str:
        return self._path


# ── Toggle ───────────────────────────────────────────────────────────────────

class Toggle(QCheckBox):
    """Native QCheckBox styled as a small toggle.

    Exposes `is_on` for backward compatibility with the previous custom widget.
    Default Qt checkbox look — no iOS-switch QPainter.
    """

    def __init__(self, on: bool = False, parent=None):
        super().__init__(parent)
        self.setChecked(on)
        self.setStyleSheet(
            f"QCheckBox {{ color:{FG1}; font-size:12px; background:transparent; spacing:6px; }}"
            f"QCheckBox::indicator {{ width:14px; height:14px; border-radius:3px; }}"
            f"QCheckBox::indicator:checked {{ background:{ACID}; border:1px solid {ACID}; }}"
            f"QCheckBox::indicator:unchecked {{ background:{BG1}; border:1px solid {LINE1}; }}"
        )

    @property
    def is_on(self) -> bool:
        return self.isChecked()


# ── RadioGroup ───────────────────────────────────────────────────────────────

_RADIO_QSS = (
    "QRadioButton {{ color:{fg1}; font-size:12px; background:transparent; }}"
    "QRadioButton::indicator {{ width:14px; height:14px; border-radius:7px; }}"
    "QRadioButton::indicator:checked {{ background:{acid}; border:2px solid {acid}; }}"
    "QRadioButton::indicator:unchecked {{ background:{bg1}; border:1px solid {line1}; }}"
)


class RadioGroup(QWidget):
    value_changed = Signal(str)

    def __init__(self, options: list[dict], value: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(_RADIO_QSS.format(fg1=FG1, acid=ACID, bg1=BG1, line1=LINE1))
        self._group = QButtonGroup(self)
        self._values: dict[QRadioButton, str] = {}

        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        for opt in options:
            btn = QRadioButton(opt["label"])
            self._values[btn] = opt["value"]
            if opt["value"] == value:
                btn.setChecked(True)
            btn.toggled.connect(
                lambda checked, v=opt["value"]: checked and self.value_changed.emit(v)
            )
            self._group.addButton(btn)
            hl.addWidget(btn)
        hl.addStretch()

    @property
    def value(self) -> str:
        for btn, v in self._values.items():
            if btn.isChecked():
                return v
        return ""


# ── Section title label ───────────────────────────────────────────────────────

def section_title(text: str) -> QLabel:
    l = QLabel(text.upper())
    l.setStyleSheet(
        f"color:{FG2}; font-size:10px; {MONO} letter-spacing:2px; background:transparent;"
    )
    return l
