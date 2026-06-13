"""Preview Select detail — audition, accept/reject, keyboard nav, sounddevice playback."""
import csv
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QKeyEvent

from app.ui.widgets.form import (
    Panel, _lbl, BG3, FG0,
    FG2, FG3, LINE0, LINE1, ACID, ACIDDIM, AMBER, MAG, MONO,
)
from app.ui.widgets.waveform import MiniWaveform

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}

try:
    import sounddevice as _sd
    import soundfile as _sf
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False


def _play_file(path: str):
    if not _HAS_AUDIO:
        return
    try:
        _sd.stop()
        data, sr = _sf.read(path, dtype="float32", always_2d=True)
        _sd.play(data, sr)
    except Exception:
        pass


def _stop_playback():
    if _HAS_AUDIO:
        try:
            _sd.stop()
        except Exception:
            pass


class _PreviewRow(QWidget):
    accept_toggled = Signal(int, bool)   # (index, is_accept)
    reject_toggled = Signal(int, bool)
    play_requested = Signal(int)

    def __init__(self, index: int, path: Path, last: bool, parent=None):
        super().__init__(parent)
        self._index = index
        self._path = path
        self._state = "pending"   # pending | accepted | rejected | playing
        self._last = last
        self.setFixedHeight(50)
        self.setCursor(Qt.ArrowCursor)
        self.setFocusPolicy(Qt.NoFocus)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(14, 8, 14, 8)
        hl.setSpacing(12)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(28, 28)
        self._play_btn.setProperty("role", "ghost")
        self._play_btn.clicked.connect(lambda: self.play_requested.emit(self._index))
        hl.addWidget(self._play_btn)

        id_lbl = _lbl(path.stem[:18], size=11, color=FG0, mono=True)
        id_lbl.setFixedWidth(120)
        hl.addWidget(id_lbl)

        self._wave = MiniWaveform(seed=hash(path.name) & 0xFFFF, width=300, height=28)
        self._wave.set_path(str(path))
        hl.addWidget(self._wave, 1)

        try:
            import soundfile as sf
            info = sf.info(str(path))
            dur = f"{info.duration:.1f}s"
        except Exception:
            dur = "—"
        hl.addWidget(_lbl(dur, size=10, color=FG3, mono=True))

        self._acc_btn = QPushButton("✓")
        self._acc_btn.setFixedSize(34, 28)
        self._acc_btn.clicked.connect(self._toggle_accept)
        self._rej_btn = QPushButton("✕")
        self._rej_btn.setFixedSize(34, 28)
        self._rej_btn.clicked.connect(self._toggle_reject)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.addWidget(self._acc_btn)
        btn_row.addWidget(self._rej_btn)
        hl.addLayout(btn_row)

        self._refresh_style()

    def set_state(self, state: str):
        self._state = state
        self._wave.set_state(state)
        self._play_btn.setText("⏸" if state == "playing" else "▶")
        self._refresh_style()
        self.update()

    def _toggle_accept(self):
        new_state = "pending" if self._state == "accepted" else "accepted"
        self.accept_toggled.emit(self._index, new_state == "accepted")

    def _toggle_reject(self):
        new_state = "pending" if self._state == "rejected" else "rejected"
        self.reject_toggled.emit(self._index, new_state == "rejected")

    def _refresh_style(self):
        acc_color = ACID if self._state == "accepted" else FG2
        rej_color = MAG  if self._state == "rejected" else FG2
        self._acc_btn.setStyleSheet(
            f"background:transparent; color:{acc_color}; border:1px solid {ACIDDIM if self._state=='accepted' else LINE1}; border-radius:4px; font-size:12px;"
        )
        self._rej_btn.setStyleSheet(
            f"background:transparent; color:{rej_color}; border:1px solid {MAG if self._state=='rejected' else LINE1}; border-radius:4px; font-size:12px;"
        )

    def paintEvent(self, event):
        p = QPainter(self)
        border_color = (ACID if self._state == "accepted" else
                        MAG  if self._state == "rejected" else
                        AMBER if self._state == "playing" else "transparent")
        if border_color != "transparent":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(border_color))
            p.drawRect(0, 0, 3, self.height())
        if self._state == "playing":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(BG3))
            p.drawRect(3, 0, self.width() - 3, self.height())
        if not self._last:
            p.setPen(QPen(QColor(LINE0), 1))
            p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


class PreviewSelectDetail(QWidget):
    selectionSaved = Signal(str)   # emits CSV path; only in pipeline mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[_PreviewRow] = []
        self._states: list[str] = []
        self._current: int = -1
        self._playing: int = -1
        self._auto_save_path: str | None = None
        self._pipeline_mode = False
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    def load_folder(self, folder_path: str, save_path: str | None = None):
        """Load all audio files from folder recursively and optionally set auto-save path."""
        self._auto_save_path = save_path
        self._pipeline_mode = save_path is not None
        paths = sorted(
            p for p in Path(folder_path).rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        self._load_paths(paths)
        if self._pipeline_mode:
            self._save_btn.setText("💾  Save & continue")

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Header row
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(16)
        hdr_col = QVBoxLayout()
        hdr_col.setSpacing(4)
        hdr_col.addWidget(_lbl("Select from previews", size=17, color=FG0, bold=True))
        hdr_col.addWidget(_lbl("Listen to preview clips, accept or reject.", size=12, color=FG2))
        hdr_row.addLayout(hdr_col, 1)
        self._save_btn = QPushButton("💾  Save selection")
        self._save_btn.setProperty("role", "primary")
        self._save_btn.setFixedHeight(30)
        self._save_btn.clicked.connect(self._save_selection)
        self._load_btn = QPushButton("📂  Load folder")
        self._load_btn.setFixedHeight(30)
        self._load_btn.clicked.connect(self._load_folder)
        hdr_row.addWidget(self._load_btn)
        hdr_row.addWidget(self._save_btn)
        layout.addLayout(hdr_row)

        # Stats bar
        self._stats = QLabel("")
        self._stats.setStyleSheet(f"color:{FG2}; font-size:11px; {MONO} background:transparent;")
        layout.addWidget(self._stats)

        # Keyboard hint
        hint = _lbl("← / → move  ·  A accept  ·  R reject  ·  Space play", size=10, color=FG3, mono=True)
        hint.setAlignment(Qt.AlignRight)
        layout.addWidget(hint)

        # List
        self._list_panel = Panel()
        self._list_layout = QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        list_container = QWidget()
        list_container.setLayout(self._list_layout)
        self._list_panel._root.addWidget(list_container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._list_panel)
        layout.addWidget(scroll, 1)

        self._refresh_stats()

    def _load_paths(self, paths: list[Path]):
        _stop_playback()
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()
        self._states = ["pending"] * len(paths)
        self._playing = -1
        self._current = 0 if paths else -1

        for i, path in enumerate(paths):
            row = _PreviewRow(i, path, last=(i == len(paths) - 1))
            row.accept_toggled.connect(self._on_accept)
            row.reject_toggled.connect(self._on_reject)
            row.play_requested.connect(self._on_play)
            self._list_layout.addWidget(row)
            self._rows.append(row)

        self._refresh_stats()
        self.setFocus()

    def _on_accept(self, idx: int, accepted: bool):
        self._states[idx] = "accepted" if accepted else "pending"
        self._rows[idx].set_state(self._states[idx])
        self._refresh_stats()

    def _on_reject(self, idx: int, rejected: bool):
        self._states[idx] = "rejected" if rejected else "pending"
        self._rows[idx].set_state(self._states[idx])
        self._refresh_stats()

    def _on_play(self, idx: int):
        if self._playing == idx:
            _stop_playback()
            self._rows[idx].set_state(
                "accepted" if self._states[idx] == "accepted" else
                "rejected" if self._states[idx] == "rejected" else "pending"
            )
            self._playing = -1
            return
        if self._playing >= 0:
            prev_state = self._states[self._playing]
            self._rows[self._playing].set_state(prev_state)
        self._playing = idx
        self._rows[idx].set_state("playing")
        _play_file(str(self._rows[idx]._path))

    def _refresh_stats(self):
        n_acc = self._states.count("accepted")
        n_rej = self._states.count("rejected")
        n_pen = self._states.count("pending")
        self._stats.setText(
            f"<span style='color:{ACID}'>{n_acc}</span> accepted  "
            f"<span style='color:{MAG}'>{n_rej}</span> rejected  "
            f"<span style='color:{FG2}'>{n_pen}</span> pending"
        )

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select previews folder")
        if not folder:
            return
        self._pipeline_mode = False
        self._auto_save_path = None
        self._save_btn.setText("💾  Save selection")
        paths = sorted(
            p for p in Path(folder).rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        self._load_paths(paths)

    def _save_selection(self):
        if not self._rows:
            return
        if self._auto_save_path:
            path = self._auto_save_path
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save selection CSV", "selection.csv", "CSV files (*.csv)")
            if not path:
                return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sound_id"])
            for row, state in zip(self._rows, self._states):
                if state == "accepted":
                    # sound_id is the parent directory name (set by first_download_freesound)
                    sound_id = row._path.parent.name
                    writer.writerow([sound_id])
        if self._pipeline_mode:
            self.selectionSaved.emit(path)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if not self._rows:
            return
        if key == Qt.Key_Right and self._current < len(self._rows) - 1:
            self._current += 1
        elif key == Qt.Key_Left and self._current > 0:
            self._current -= 1
        elif key == Qt.Key_A:
            self._on_accept(self._current, self._states[self._current] != "accepted")
        elif key == Qt.Key_R:
            self._on_reject(self._current, self._states[self._current] != "rejected")
        elif key == Qt.Key_Space:
            self._on_play(self._current)
        else:
            super().keyPressEvent(event)
