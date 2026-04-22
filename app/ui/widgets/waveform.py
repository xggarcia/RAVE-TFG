"""MiniWaveform — draws a waveform bar chart from audio data or a seeded placeholder."""
import random
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QPen

ACID  = "#a8e63d"
AMBER = "#e8c040"
MAG   = "#e0406a"
FG3   = "#717870"
BG0   = "#1e2320"


class _AudioLoader(QThread):
    loaded = Signal(list)  # list of float bar heights

    def __init__(self, path: str, n_bars: int = 60):
        super().__init__()
        self._path = path
        self._n = n_bars

    def run(self):
        try:
            import soundfile as sf
            import numpy as np
            data, _ = sf.read(self._path, dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            frames = len(mono)
            bars = []
            for i in range(self._n):
                start = int(i * frames / self._n)
                end = int((i + 1) * frames / self._n)
                chunk = mono[start:end]
                bars.append(float(np.abs(chunk).mean()) if len(chunk) else 0.0)
            if max(bars) > 0:
                m = max(bars)
                bars = [b / m for b in bars]
            self.loaded.emit(bars)
        except Exception:
            self.loaded.emit([])


class MiniWaveform(QWidget):
    """Seeded placeholder waveform, upgrades to real data when path is set."""

    def __init__(self, seed: int = 0, width: int = 300, height: int = 28, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._progress: float | None = None
        self._state = "pending"   # pending | accepted | rejected | playing
        self._loader: _AudioLoader | None = None

        r = random.Random(seed)
        self._bars: list[float] = [r.uniform(0.15, 1.0) for _ in range(60)]

    def set_path(self, path: str):
        self._loader = _AudioLoader(path)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.start()

    def _on_loaded(self, bars: list[float]):
        if bars:
            self._bars = bars
        self.update()

    def set_state(self, state: str, progress: float | None = None):
        self._state = state
        self._progress = progress
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()

        color = (ACID if self._state == "accepted" else
                 MAG  if self._state == "rejected" else
                 AMBER if self._state == "playing" else FG3)
        bar_color = QColor(color)
        bar_color.setAlphaF(0.7)

        n = len(self._bars)
        bar_w = max(1, w // n - 1)
        spacing = w / n

        for i, amp in enumerate(self._bars):
            bh = max(2, int(h * amp))
            x = int(i * spacing)
            y = (h - bh) // 2
            p.setPen(Qt.NoPen)
            p.setBrush(bar_color)
            p.drawRect(x, y, bar_w, bh)

        # Playback cursor
        if self._progress is not None and self._state == "playing":
            cx = int(self._progress * w)
            p.setPen(QPen(QColor(AMBER), 1))
            p.drawLine(cx, 0, cx, h)

        p.end()
